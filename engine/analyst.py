"""
Neural Analyst — GPT-4o powered vibe-check, deal scoring, and red-flag detection.
Falls back to a deterministic heuristic scorer when no API key is present.
"""

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from engine.sourcing import RawListing

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Output model
# ---------------------------------------------------------------------------

@dataclass
class DealScore:
    listing_id: str
    title: str
    source: str
    url: str
    asking_price: Optional[float]
    mrr: Optional[float]
    arr: Optional[float]

    # AI signals (0–10 scale)
    motivation_score: float = 0.0       # How badly does seller want out?
    handover_risk: float = 0.0          # "I am the brand" dependency risk
    boredom_multiple_likely: bool = False  # Will they accept 2–2.5x ARR?
    intent_label: str = "Unknown"       # "Burnout" | "Pivot" | "Scaling" | "Distressed"

    # Narrative fields
    acquisition_thesis: str = ""
    red_flags: list[str] = field(default_factory=list)
    green_flags: list[str] = field(default_factory=list)
    one_liner: str = ""

    # Composite rank score (higher = hotter deal)
    heat_score: float = 0.0

    seller_handle: Optional[str] = None
    age_days: Optional[int] = None
    tags: list = field(default_factory=list)

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a ruthless but fair micro-private-equity deal analyst working for a solo operator 
who acquires small digital businesses under $100k. Your job is to rapidly assess seller listings 
for acquisition signal — specifically emotional and operational distress that indicates a motivated seller.

You reason on:
1. HANDOVER RISK — Is the founder the business? Personal brand dependency, sole support person, 
   "I am the brand" signals. Score 0 (clean handover) to 10 (business dies without founder).

2. MOTIVATION FACTOR — Why are they selling? Burnout, pivot, new job, boredom, financial pressure?
   Score 0 (exploring options) to 10 (desperate to exit now).

3. BOREDOM MULTIPLE — Based on tone and context, is this seller likely to accept 2x–2.5x ARR 
   due to operational fatigue rather than holding out for 3x–4x? (true/false)

4. INTENT LABEL — Classify the primary exit driver as exactly one of:
   "Burnout" | "Pivot" | "Boredom" | "Financial" | "Opportunity" | "Unknown"

5. ACQUISITION THESIS — 2-sentence max. Why this deal is interesting for a micro-PE operator. 
   Be concrete about the asset, the moat, and the operator upside.

6. RED FLAGS — List up to 3 specific risks (founder dependency, declining metrics, tech debt signals, 
   unclear ownership, earnout demands, etc.)

7. GREEN FLAGS — List up to 3 positive signals (low churn, passive revenue, simple product, 
   fast close preference, clean cap table, etc.)

8. ONE-LINER — A single punchy sentence summarizing the deal for a deal feed (max 15 words).

Respond ONLY with valid JSON matching this exact schema:
{
  "motivation_score": <float 0-10>,
  "handover_risk": <float 0-10>,
  "boredom_multiple_likely": <bool>,
  "intent_label": "<string>",
  "acquisition_thesis": "<string>",
  "red_flags": ["<string>", ...],
  "green_flags": ["<string>", ...],
  "one_liner": "<string>"
}"""

USER_PROMPT_TEMPLATE = """Analyze this listing for a micro-PE acquisition opportunity:

SOURCE: {source}
TITLE: {title}
ASKING PRICE: {price}
MRR: {mrr}
ARR: {arr}
LISTING AGE: {age} days old
SELLER HANDLE: {handle}

LISTING TEXT:
{body}

Return the JSON analysis now."""

# ---------------------------------------------------------------------------
# GPT-4o analyst
# ---------------------------------------------------------------------------

def analyze_with_gpt(listing: RawListing, client) -> dict:
    """Call GPT-4o to analyze a listing. Returns parsed JSON dict."""
    price_str = f"${listing.asking_price:,.0f}" if listing.asking_price else "Not stated"
    mrr_str = f"${listing.mrr:,.0f}/mo" if listing.mrr else "Not stated"
    arr_str = f"${listing.arr:,.0f}/yr" if listing.arr else "Not stated"
    age_str = str(listing.age_days) if listing.age_days is not None else "Unknown"

    user_msg = USER_PROMPT_TEMPLATE.format(
        source=listing.source,
        title=listing.title,
        price=price_str,
        mrr=mrr_str,
        arr=arr_str,
        age=age_str,
        handle=listing.seller_handle or "Not provided",
        body=listing.body[:3000],
    )

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.3,
        max_tokens=800,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    return json.loads(raw)

# ---------------------------------------------------------------------------
# Heuristic fallback scorer (no API key needed)
# ---------------------------------------------------------------------------

BURNOUT_SIGNALS = [
    "burnt out", "burned out", "burnout", "tired", "exhausted", "drained",
    "honestly just", "moving on", "ready to exit", "want out", "bored",
    "mentally blocking", "new chapter", "new idea", "obsessed with",
]
PIVOT_SIGNALS = [
    "pivot", "new startup", "vc-backed", "job offer", "full-time job",
    "new venture", "consulting", "going all-in", "new direction",
]
HANDOVER_RISK_SIGNALS = [
    "i am the brand", "i am the support", "sole developer", "only developer",
    "only person", "i handle everything", "customers know me personally",
]
FAST_CLOSE_SIGNALS = [
    "quick close", "fast close", "no earnout", "asset purchase", "simple purchase",
    "cash buyer", "clean exit", "no drama",
]
FINANCIAL_SIGNALS = [
    "need cash", "financial", "bills", "debt", "struggling", "low runway",
]

def _keyword_score(text: str, keywords: list[str]) -> float:
    text_lower = text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    return min(hits * 2.5, 10.0)

def heuristic_score(listing: RawListing) -> dict:
    """Deterministic fallback when OpenAI is unavailable."""
    combined = f"{listing.title} {listing.body}".lower()

    motivation = max(
        _keyword_score(combined, BURNOUT_SIGNALS),
        _keyword_score(combined, PIVOT_SIGNALS),
        _keyword_score(combined, FINANCIAL_SIGNALS),
    )
    handover = _keyword_score(combined, HANDOVER_RISK_SIGNALS)

    # Slight boost if seller explicitly mentions handover period
    if re.search(r"\d+[- ]day handover|\d+[- ]week hand", combined):
        handover = max(handover - 1, 0)

    fast_close = _keyword_score(combined, FAST_CLOSE_SIGNALS) > 0

    # Determine intent label
    burn_score = _keyword_score(combined, BURNOUT_SIGNALS)
    pivot_score = _keyword_score(combined, PIVOT_SIGNALS)
    fin_score = _keyword_score(combined, FINANCIAL_SIGNALS)

    if burn_score >= pivot_score and burn_score >= fin_score:
        if burn_score > 3:
            intent = "Burnout"
        else:
            intent = "Boredom"
    elif pivot_score >= fin_score:
        intent = "Pivot"
    elif fin_score > 0:
        intent = "Financial"
    else:
        intent = "Unknown"

    # Build flags
    red_flags = []
    green_flags = []

    if handover > 5:
        red_flags.append("High founder dependency — likely 'I am the brand' scenario")
    if not listing.mrr:
        red_flags.append("MRR not stated — verify revenue claims before LOI")
    if listing.age_days and listing.age_days > 60:
        red_flags.append(f"Listing aged {listing.age_days} days — why hasn't it sold?")
    if "earnout" in combined and "no earnout" not in combined:
        red_flags.append("Seller may be expecting earnout structure")

    if fast_close:
        green_flags.append("Seller explicitly prefers quick/clean close")
    if listing.mrr and listing.arr and listing.asking_price:
        multiple = listing.asking_price / listing.arr if listing.arr > 0 else 99
        if multiple <= 2.5:
            green_flags.append(f"Asking multiple is {multiple:.1f}x ARR — below market ceiling")
    if "zero churn" in combined or "no churn" in combined or "churn < 3" in combined:
        green_flags.append("Very low churn mentioned — stable recurring revenue base")
    if "simple" in combined or "runs itself" in combined:
        green_flags.append("Product described as low-maintenance / self-running")

    # Build thesis
    price_str = f"${listing.asking_price:,.0f}" if listing.asking_price else "undisclosed"
    mrr_str = f"${listing.mrr:,.0f}/mo" if listing.mrr else "unknown MRR"
    thesis = (
        f"{listing.source} listing at {price_str} ({mrr_str}) with a {intent.lower()}-driven exit. "
        f"Opportunity: motivated seller likely open to quick close at a fair multiple."
    )

    one_liner = f"{intent} exit — {mrr_str} asset, seller wants out fast."

    return {
        "motivation_score": round(motivation, 1),
        "handover_risk": round(handover, 1),
        "boredom_multiple_likely": fast_close or motivation > 5,
        "intent_label": intent,
        "acquisition_thesis": thesis,
        "red_flags": red_flags[:3],
        "green_flags": green_flags[:3],
        "one_liner": one_liner,
    }

# ---------------------------------------------------------------------------
# Heat score computation
# ---------------------------------------------------------------------------

def compute_heat_score(score_data: dict, listing: RawListing) -> float:
    """
    Composite 0–100 score weighting:
    - Seller motivation (40%)
    - Low multiple (25%)
    - Fast close signals (20%)
    - Low handover risk (15%)
    """
    motivation_component = score_data["motivation_score"] * 4.0

    multiple_component = 0.0
    if listing.asking_price and listing.arr and listing.arr > 0:
        multiple = listing.asking_price / listing.arr
        if multiple <= 2.0:
            multiple_component = 25.0
        elif multiple <= 2.5:
            multiple_component = 20.0
        elif multiple <= 3.0:
            multiple_component = 12.0
        else:
            multiple_component = 5.0

    fast_close_component = 20.0 if score_data.get("boredom_multiple_likely") else 0.0

    handover_component = max(0, (10 - score_data["handover_risk"])) * 1.5

    raw = motivation_component + multiple_component + fast_close_component + handover_component
    return round(min(raw, 100.0), 1)

# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def score_listing(listing: RawListing, openai_client=None) -> DealScore:
    """
    Score a single listing. Uses GPT-4o if client is available, otherwise heuristics.
    """
    try:
        if openai_client:
            score_data = analyze_with_gpt(listing, openai_client)
        else:
            score_data = heuristic_score(listing)
    except Exception as exc:
        logger.warning("GPT scoring failed for %s: %s — falling back to heuristics", listing.id, exc)
        score_data = heuristic_score(listing)

    heat = compute_heat_score(score_data, listing)

    return DealScore(
        listing_id=listing.id,
        title=listing.title,
        source=listing.source,
        url=listing.url,
        asking_price=listing.asking_price,
        mrr=listing.mrr,
        arr=listing.arr,
        motivation_score=score_data.get("motivation_score", 0),
        handover_risk=score_data.get("handover_risk", 0),
        boredom_multiple_likely=score_data.get("boredom_multiple_likely", False),
        intent_label=score_data.get("intent_label", "Unknown"),
        acquisition_thesis=score_data.get("acquisition_thesis", ""),
        red_flags=score_data.get("red_flags", []),
        green_flags=score_data.get("green_flags", []),
        one_liner=score_data.get("one_liner", ""),
        heat_score=heat,
        seller_handle=listing.seller_handle,
        age_days=listing.age_days,
        tags=listing.tags,
    )


def score_all_listings(listings: list[RawListing], openai_client=None) -> list[DealScore]:
    """Score and rank all listings by heat score (descending)."""
    scored = []
    for listing in listings:
        try:
            ds = score_listing(listing, openai_client)
            scored.append(ds)
        except Exception as exc:
            logger.error("Skipping listing %s due to error: %s", listing.id, exc)

    scored.sort(key=lambda x: x.heat_score, reverse=True)
    return scored
