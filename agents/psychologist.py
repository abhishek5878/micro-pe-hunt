"""
Founder Psychology Agent.
Analyzes founder text for burnout, conviction, momentum trajectory, and motivation type.
Uses VADER (no downloads required) + custom India-specific keyword lexicon.
Produces scores 0–10 and a categorical motivation profile.
"""

import json
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

KB_PATH = Path(__file__).parent.parent / "data" / "sme_knowledge.json"


def _load_keywords() -> dict:
    with open(KB_PATH) as f:
        return json.load(f).get("founder_nlp_keywords", {})


# ─────────────────────────────────────────────────────────────────────────────
# VADER Loader (graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────

def _get_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        logger.warning("vaderSentiment not installed — sentiment scoring disabled")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Keyword Scoring
# ─────────────────────────────────────────────────────────────────────────────

def _keyword_score(text: str, keyword_list: list[str], weight: float = 1.0) -> tuple[float, list[str]]:
    """
    Score text against a keyword list.
    Returns (raw_score 0–10, matched_keywords).
    """
    text_lower = text.lower()
    hits = [kw for kw in keyword_list if kw.lower() in text_lower]
    score = min(len(hits) * 2.0 * weight, 10.0)
    return score, hits


def _extract_india_signals(text: str, keywords: dict) -> dict:
    """Extract India-specific cultural signals from founder text."""
    india_burnout_score, india_hits = _keyword_score(
        text, keywords.get("india_specific_burnout", []), weight=1.5
    )
    family_score, family_hits = _keyword_score(
        text, keywords.get("family_exit", []), weight=1.2
    )
    return {
        "india_cultural_burnout_score": india_burnout_score,
        "india_cultural_burnout_signals": india_hits,
        "family_exit_score": family_score,
        "family_exit_signals": family_hits,
    }


def _extract_financial_distress(text: str, keywords: dict) -> dict:
    """Detect financial distress signals."""
    score, hits = _keyword_score(text, keywords.get("financial_distress", []), weight=1.5)
    return {"financial_distress_score": score, "financial_distress_signals": hits}


# ─────────────────────────────────────────────────────────────────────────────
# Motivation Classifier
# ─────────────────────────────────────────────────────────────────────────────

MOTIVATION_PROFILES = {
    "Burnout Exit": {
        "description": "Founder is mentally exhausted and wants immediate closure. Strong discount to ask likely.",
        "negotiation_tip": "Move fast. Low-ball slightly — they'll accept. Emphasize 'close in 30 days.'",
        "price_flexibility": "high",
        "urgency": "immediate",
        "color": "#ef4444",
    },
    "Opportunity Exit": {
        "description": "Founder has a better opportunity (new job, startup, VC deal). Timeline-driven — wants clean close.",
        "negotiation_tip": "Match their timeline. Offer certainty over price. 'We close before your new role starts.'",
        "price_flexibility": "medium",
        "urgency": "within 2-3 months",
        "color": "#f59e0b",
    },
    "Family Exit": {
        "description": "Second-gen or succession-driven exit. Patriarch involvement likely. Emotional weight high.",
        "negotiation_tip": "Relationship-first approach. Meet the patriarch. Frame as 'continuing the legacy' not 'buying the business.'",
        "price_flexibility": "low",
        "urgency": "flexible but patriarch-dependent",
        "color": "#8b5cf6",
    },
    "Boredom / Drift": {
        "description": "Founder mentally checked out. Business runs on autopilot. Low urgency but open to right offer.",
        "negotiation_tip": "Don't rush. Build rapport. They're not desperate — show you'll care for the business.",
        "price_flexibility": "medium",
        "urgency": "no fixed timeline",
        "color": "#3b82f6",
    },
    "Financial Distress": {
        "description": "Revenue declining or cash flow pressure. High motivation, some desperation. Price expectations may reset.",
        "negotiation_tip": "Be careful — distress exits have hidden liabilities. Do extra diligence. Offer certainty but price conservatively.",
        "price_flexibility": "very high",
        "urgency": "urgent",
        "color": "#6b7280",
    },
    "Low Signal": {
        "description": "Insufficient founder signal to classify. Generic exit language.",
        "negotiation_tip": "Ask directly: 'What's driving the timeline?' before making any move.",
        "price_flexibility": "unknown",
        "urgency": "unknown",
        "color": "#374151",
    },
}


def classify_motivation(
    burnout_score: float,
    opportunity_score: float,
    family_score: float,
    financial_distress_score: float,
    conviction_score: float,
) -> str:
    """Return the dominant motivation profile label."""
    scores = {
        "Burnout Exit": burnout_score,
        "Opportunity Exit": opportunity_score,
        "Family Exit": family_score,
        "Financial Distress": financial_distress_score,
        "Boredom / Drift": max(0, 6 - conviction_score),  # low conviction → drift
    }
    dominant = max(scores, key=scores.get)
    if scores[dominant] < 1.5:
        return "Low Signal"
    return dominant


# ─────────────────────────────────────────────────────────────────────────────
# Momentum Trajectory (time-based signals)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_momentum_timeline(founder_signals: dict) -> list[dict]:
    """
    Build a timeline of founder momentum signals.
    founder_signals dict: {linkedin_activity, last_tweet, reddit_post, etc.}
    Returns list of {label, direction, note} for chart rendering.
    """
    timeline = []

    twitter_bio = founder_signals.get("twitter_bio", "")
    last_tweet = founder_signals.get("last_tweet", "")
    linkedin = founder_signals.get("linkedin_activity", "")
    reddit = founder_signals.get("reddit_post", "")

    # Analyse each signal channel
    vader = _get_vader()

    channels = [
        ("Twitter Bio", twitter_bio),
        ("Recent Tweet", last_tweet),
        ("LinkedIn Activity", linkedin),
        ("Reddit Post", reddit),
    ]

    for label, text in channels:
        if not text or text.strip() in ("None", "N/A", ""):
            continue
        if vader:
            scores = vader.polarity_scores(text)
            compound = scores["compound"]
            direction = "positive" if compound > 0.1 else "negative" if compound < -0.1 else "neutral"
            sentiment_score = round((compound + 1) * 5, 1)  # map -1:1 to 0:10
        else:
            direction = "neutral"
            sentiment_score = 5.0

        # Check for exit-specific signals
        exit_keywords = ["selling", "exit", "moving on", "new chapter", "done", "sold"]
        is_exit_signal = any(k in text.lower() for k in exit_keywords)

        timeline.append({
            "channel": label,
            "text": text[:100],
            "direction": direction,
            "score": sentiment_score,
            "is_exit_signal": is_exit_signal,
        })

    return timeline


# ─────────────────────────────────────────────────────────────────────────────
# Main Psychologist Agent
# ─────────────────────────────────────────────────────────────────────────────

class FounderPsychologist:
    """
    Analyzes founder/listing text for psychological acquisition signals.
    Works purely with rule-based NLP — no API key needed.
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.vader = _get_vader()
        self.keywords = _load_keywords()

    def analyze(
        self,
        listing_text: str,
        founder_signals: dict = None,
        transcript: Optional[str] = None,
    ) -> dict:
        """
        Full psychological analysis of a deal.
        Returns structured profile dict.
        """
        # Combine all text sources
        all_text = listing_text or ""
        if founder_signals:
            all_text += " " + " ".join(str(v) for v in founder_signals.values() if v)
        if transcript:
            all_text += " " + transcript

        # ── Burnout Scoring ───────────────────────────────────────────────
        burnout_h_score, burnout_h_hits = _keyword_score(all_text, self.keywords.get("burnout_high", []), weight=2.0)
        burnout_m_score, burnout_m_hits = _keyword_score(all_text, self.keywords.get("burnout_medium", []), weight=1.2)
        burnout_raw = min(burnout_h_score * 0.7 + burnout_m_score * 0.3, 10)

        # ── Conviction Scoring ────────────────────────────────────────────
        conv_h_score, conv_h_hits = _keyword_score(all_text, self.keywords.get("conviction_high", []), weight=2.0)
        conv_m_score, conv_m_hits = _keyword_score(all_text, self.keywords.get("conviction_medium", []), weight=1.2)
        conviction_raw = min(conv_h_score * 0.7 + conv_m_score * 0.3, 10)

        # ── Opportunity Exit Scoring ──────────────────────────────────────
        opp_score, opp_hits = _keyword_score(all_text, self.keywords.get("opportunity_exit", []), weight=1.5)

        # ── India-Specific Signals ────────────────────────────────────────
        india_signals = _extract_india_signals(all_text, self.keywords)
        financial_signals = _extract_financial_distress(all_text, self.keywords)

        # ── VADER Sentiment on listing text ───────────────────────────────
        vader_compound = 0.0
        if self.vader and listing_text:
            vader_scores = self.vader.polarity_scores(listing_text)
            vader_compound = vader_scores["compound"]

        # ── Adjust burnout with VADER ─────────────────────────────────────
        # Negative sentiment boosts burnout; positive sentiment reduces it
        vader_burnout_adj = max(0, -vader_compound * 2)  # negative compound → burnout boost
        burnout_final = min(burnout_raw + vader_burnout_adj, 10)
        conviction_final = min(conviction_raw + max(0, vader_compound * 1.5), 10)

        # ── Family exit score ─────────────────────────────────────────────
        family_score = india_signals["family_exit_score"]

        # ── Financial distress ────────────────────────────────────────────
        fin_distress_score = financial_signals["financial_distress_score"]

        # ── Classify motivation ───────────────────────────────────────────
        motivation_type = classify_motivation(
            burnout_final, opp_score, family_score, fin_distress_score, conviction_final
        )
        profile = MOTIVATION_PROFILES.get(motivation_type, MOTIVATION_PROFILES["Low Signal"])

        # ── Boredom multiple signal ───────────────────────────────────────
        # Will seller accept 2–2.5x ARR? → High burnout or financial distress
        boredom_multiple = burnout_final >= 6 or fin_distress_score >= 5
        fast_close_keywords = ["quick close", "fast close", "no earnout", "clean exit", "closure", "done"]
        fast_close_signal = any(k in all_text.lower() for k in fast_close_keywords)

        # ── Momentum Timeline ─────────────────────────────────────────────
        timeline = []
        if founder_signals:
            timeline = _parse_momentum_timeline(founder_signals)

        # ── Transcript Extraction ─────────────────────────────────────────
        transcript_extracts = {}
        if transcript:
            transcript_extracts = self._extract_transcript_facts(transcript)

        return {
            # Core scores (0–10)
            "burnout_score": round(burnout_final, 1),
            "conviction_score": round(conviction_final, 1),
            "opportunity_exit_score": round(opp_score, 1),
            "family_exit_score": round(family_score, 1),
            "financial_distress_score": round(fin_distress_score, 1),
            "india_cultural_score": round(india_signals["india_cultural_burnout_score"], 1),

            # Metadata
            "vader_compound": round(vader_compound, 3),
            "burnout_signals": burnout_h_hits[:5] + burnout_m_hits[:3],
            "conviction_signals": conv_h_hits[:3] + conv_m_hits[:3],
            "family_signals": india_signals["family_exit_signals"][:5],
            "distress_signals": financial_signals["financial_distress_signals"][:3],
            "opportunity_signals": opp_hits[:3],

            # Classification
            "motivation_type": motivation_type,
            "motivation_profile": profile,
            "boredom_multiple_likely": boredom_multiple,
            "fast_close_signal": fast_close_signal,

            # Charts data
            "momentum_timeline": timeline,
            "radar_data": {
                "Burnout": round(burnout_final, 1),
                "Conviction": round(conviction_final, 1),
                "Opp. Exit": round(min(opp_score, 10), 1),
                "Family Exit": round(min(family_score, 10), 1),
                "Fin. Distress": round(min(fin_distress_score, 10), 1),
            },

            # Transcript
            "transcript_extracts": transcript_extracts,
        }

    def _extract_transcript_facts(self, transcript: str) -> dict:
        """
        Extract structured facts from a voice call transcript.
        Returns: sector, location, revenue_hint, motivation, timeline, family_angle.
        """
        text_lower = transcript.lower()

        # Revenue mentions (₹ amounts)
        rev_patterns = re.findall(r'(?:₹|rs\.?|inr)\s*([\d.]+)\s*(crore|cr|lakh|l|lakhs)', text_lower)
        revenue_mentions = []
        for val, unit in rev_patterns:
            try:
                num = float(val)
                if "crore" in unit or unit == "cr":
                    revenue_mentions.append(f"₹{num}Cr")
                else:
                    revenue_mentions.append(f"₹{num}L")
            except ValueError:
                pass

        # Also detect "X crore" patterns
        word_rev = re.findall(r'([\d.]+)\s*(?:crore|cr)\b', text_lower)
        for v in word_rev:
            try:
                revenue_mentions.append(f"₹{float(v):.1f}Cr")
            except ValueError:
                pass

        # EBITDA / margin mentions
        ebitda_mentions = re.findall(r'ebitda\s*(?:is|of|around)?\s*(?:₹|rs\.?)?\s*([\d.]+)\s*(lakh|crore|cr|l)?', text_lower)

        # Timeline
        timeline_hints = re.findall(r'(?:within|in|by)\s+([\d]+)\s*(month|months|week|weeks|year|years)', text_lower)
        timeline = f"~{timeline_hints[0][0]} {timeline_hints[0][1]}" if timeline_hints else "Not specified"

        # Stakeholder mentions
        family_mentions = any(w in text_lower for w in ["father", "mother", "brother", "sister", "son", "daughter", "family", "patriarch"])

        # Sector hints
        sector_keywords = {
            "manufacturing": ["machine", "machining", "cnc", "parts", "components", "factory", "plant", "manufacturing", "engineering"],
            "saas": ["software", "saas", "app", "subscription", "mrr", "arr", "customers pay"],
            "d2c_ecommerce": ["brand", "product", "amazon", "flipkart", "nykaa", "d2c", "ecommerce"],
            "services_b2b": ["services", "consulting", "clients", "b2b", "agency"],
            "food_fmcg": ["food", "fmcg", "distribution", "retail", "consumers", "fssai"],
        }
        detected_sector = "manufacturing_general"
        for sector, words in sector_keywords.items():
            if any(w in text_lower for w in words):
                detected_sector = sector
                break

        # Motivation hints from transcript
        motivation_hints = []
        if any(w in text_lower for w in ["tired", "exhausted", "want to move on", "new idea", "new venture"]):
            motivation_hints.append("Personal pivot / burnout")
        if family_mentions:
            motivation_hints.append("Family dynamics / succession")
        if any(w in text_lower for w in ["fair price", "flexible", "right buyer", "just want closure"]):
            motivation_hints.append("Price-flexible / closure-driven")

        return {
            "revenue_mentions": list(set(revenue_mentions))[:5],
            "ebitda_mentions": [f"₹{e[0]}{e[1]}" for e in ebitda_mentions[:3]],
            "timeline": timeline,
            "has_family_angle": family_mentions,
            "detected_sector": detected_sector,
            "motivation_hints": motivation_hints,
        }

    def generate_psych_narrative(self, psych_data: dict, deal_context: str) -> str:
        """Generate a psychographic profile narrative. Uses LLM if available."""
        mtype = psych_data["motivation_type"]
        burnout = psych_data["burnout_score"]
        conviction = psych_data["conviction_score"]
        profile = psych_data["motivation_profile"]
        boredom_m = psych_data["boredom_multiple_likely"]
        fast_close = psych_data["fast_close_signal"]

        if self.llm:
            from langchain.schema import HumanMessage, SystemMessage
            system = (
                "You are a behavioral psychologist advising a micro-PE buyer. "
                "Write in crisp bullet-point style. Max 200 words. Be specific and actionable."
            )
            prompt = f"""
Analyze the founder's psychological profile for this acquisition:

DEAL CONTEXT: {deal_context[:500]}

SCORES:
- Burnout: {burnout}/10
- Conviction: {conviction}/10  
- Motivation Type: {mtype}
- Boredom Multiple Likely (accepts 2x ARR): {boredom_m}
- Fast Close Signal: {fast_close}
- Key Signals: {psych_data.get('burnout_signals', [])[:3]}
- Family Signals: {psych_data.get('family_signals', [])[:3]}

Write:
1. What's really driving this exit (2 sentences)
2. How to approach this founder (tone, channel, timing)
3. What NOT to do (1-2 specific warnings)
4. Negotiation angle based on their psychological state
"""
            try:
                resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
                return resp.content
            except Exception as e:
                logger.warning("LLM psych narrative failed: %s", e)

        # Template fallback
        tip = profile.get("negotiation_tip", "")
        price_flex = profile.get("price_flexibility", "unknown")
        urgency = profile.get("urgency", "unknown")

        return f"""
**Primary Exit Driver: {mtype}**
Burnout Score {burnout}/10 | Conviction {conviction}/10 | Price Flexibility: {price_flex.upper()}

**Reading the Founder:**
{"High burnout detected — founder has mentally exited already. The business is a psychological burden." if burnout >= 7 else "Moderate exit pressure — motivated but not desperate."}
{"Fast close explicitly signaled — prioritize certainty over price in your pitch." if fast_close else "No explicit timeline pressure — move at a comfortable pace."}
{"Boredom multiple likely — seller will probably accept 2–2.5x ARR for a clean exit." if boredom_m else "Seller likely expects closer to market multiples — don't try to lowball."}

**Approach Recommendation:**
{tip}

**Urgency:** {urgency}
""".strip()
