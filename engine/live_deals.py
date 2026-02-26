"""
Live Deal Sourcing Engine — multi-platform acquisition intelligence.

Sources:
  1. Acquire.com (auth)  — Playwright login, 60+ individual listing detail pages
  2. Empire Flippers     — Firecrawl LLM extract, structured data
  3. Flippa              — Firecrawl scrape + regex, 8 category pages
  4. SideProjectors      — RSS + page scrape, public listings
  5. Reddit              — PRAW OAuth, 8 subreddits, strict exit-signal filter
  6. Firecrawl search    — Web search for fresh deal signals

Returns normalised LiveDeal dicts ready for the agent pipeline and UI.
"""

import logging
import os
import re
import time
from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

USD_TO_INR = 83.5

# ─────────────────────────────────────────────────────────────────────────────
# Normalised deal schema
# ─────────────────────────────────────────────────────────────────────────────

def _make_deal(
    *,
    source: str,
    title: str,
    url: str,
    asking_price_usd: Optional[float] = None,
    monthly_profit_usd: Optional[float] = None,
    monthly_revenue_usd: Optional[float] = None,
    multiple: Optional[float] = None,
    niche: str = "",
    monetization: str = "",
    age_years: Optional[float] = None,
    country: str = "",
    description: str = "",
    seller_handle: str = "",
    listing_id: str = "",
    fetched_at: Optional[str] = None,
    scrape_method: str = "",
    tags: Optional[list] = None,
) -> dict:
    arr_usd = monthly_revenue_usd * 12 if monthly_revenue_usd else None
    ebitda_usd = monthly_profit_usd * 12 if monthly_profit_usd else None
    price_inr_cr = round(asking_price_usd * USD_TO_INR / 10_000_000, 3) if asking_price_usd else None
    revenue_cr   = round(arr_usd      * USD_TO_INR / 10_000_000, 4) if arr_usd else None
    ebitda_l     = round(ebitda_usd   * USD_TO_INR / 100_000, 1)    if ebitda_usd else None

    return {
        "source":               source,
        "title":                title[:120],
        "url":                  url,
        "asking_price_usd":     asking_price_usd,
        "asking_price_inr_cr":  price_inr_cr,
        "monthly_profit_usd":   monthly_profit_usd,
        "monthly_revenue_usd":  monthly_revenue_usd,
        "multiple":             multiple,
        "arr_usd":              arr_usd,
        "ebitda_usd":           ebitda_usd,
        "revenue_cr":           revenue_cr or 0.0,
        "ebitda_l":             ebitda_l or 0.0,
        "niche":                niche,
        "monetization":         monetization,
        "age_years":            age_years,
        "country":              country,
        "description":          description[:2000],
        "seller_handle":        seller_handle,
        "listing_id":           listing_id,
        "scrape_method":        scrape_method,
        "tags":                 tags or [],
        "fetched_at":           fetched_at or datetime.utcnow().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Firecrawl factory
# ─────────────────────────────────────────────────────────────────────────────

def _get_fc():
    key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not key:
        return None
    try:
        from firecrawl import FirecrawlApp
        return FirecrawlApp(api_key=key)
    except Exception as exc:
        logger.warning("Firecrawl init: %s", exc)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Currency parsers
# ─────────────────────────────────────────────────────────────────────────────

def _usd(text: str) -> Optional[float]:
    if not text:
        return None
    text = str(text).replace(",", "")
    # Require suffix to NOT be followed by another letter (avoids $5400MRR → $5.4B)
    m = re.search(r"\$?([\d]+(?:\.[\d]+)?)\s*([kKmM](?![a-zA-Z]))?", text)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    s = (m.group(2) or "").lower()
    if s == "k": val *= 1_000
    elif s == "m": val *= 1_000_000
    return val


def _multiple(text: str) -> Optional[float]:
    if not text:
        return None
    m = re.search(r"([\d]+(?:\.[\d]+)?)\s*[xX]", str(text))
    return float(m.group(1)) if m else None


# ─────────────────────────────────────────────────────────────────────────────
# Source 1 — Empire Flippers (Firecrawl LLM extract)
# ─────────────────────────────────────────────────────────────────────────────

EF_SCHEMA = {
    "type": "object",
    "properties": {
        "listings": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "listing_id":      {"type": "string"},
                    "url":             {"type": "string"},
                    "niche":           {"type": "string"},
                    "asking_price":    {"type": "string"},
                    "monthly_profit":  {"type": "string"},
                    "monthly_revenue": {"type": "string"},
                    "multiple":        {"type": "string"},
                    "monetization":    {"type": "string"},
                    "age":             {"type": "string"},
                    "description":     {"type": "string"},
                },
            },
        }
    },
}

EF_PAGES = [
    "https://empireflippers.com/marketplace/",
    "https://empireflippers.com/marketplace/?niche=saas",
]


def fetch_empire_flippers(max_deals: int = 20) -> list[dict]:
    """Scrape Empire Flippers via Firecrawl LLM extract."""
    fc = _get_fc()
    if not fc:
        logger.warning("EF: no Firecrawl key")
        return []

    deals = []
    for page_url in EF_PAGES:
        try:
            logger.info("EF extract: %s", page_url)
            result = fc.extract(
                urls=[page_url],
                prompt=(
                    "Extract ALL business listings visible on this Empire Flippers marketplace page. "
                    "For each listing capture: listing ID (e.g. #91401), URL, niche/category, "
                    "asking price, monthly net profit, monthly revenue, multiple (e.g. 37x), "
                    "monetization type, age, and description snippet."
                ),
                schema=EF_SCHEMA,
            )
            raw = getattr(result, "data", {}) or {}
            items = raw.get("listings", []) if isinstance(raw, dict) else []
            for item in items:
                lid  = str(item.get("listing_id", "")).strip("#")
                url  = item.get("url", "") or f"https://empireflippers.com/listing/{lid}"
                if not url.startswith("http"):
                    url = f"https://empireflippers.com/listing/{lid}"
                deals.append(_make_deal(
                    source="EmpireFlippers",
                    title=item.get("niche", f"EF Listing #{lid}"),
                    url=url,
                    asking_price_usd=_usd(item.get("asking_price", "")),
                    monthly_profit_usd=_usd(item.get("monthly_profit", "")),
                    monthly_revenue_usd=_usd(item.get("monthly_revenue", "")),
                    multiple=_multiple(item.get("multiple", "")),
                    niche=item.get("niche", ""),
                    monetization=item.get("monetization", ""),
                    description=item.get("description", ""),
                    listing_id=lid,
                    scrape_method="firecrawl-extract",
                ))
        except Exception as exc:
            logger.warning("EF page %s failed: %s", page_url, exc)

    logger.info("Empire Flippers: %d deals", len(deals))
    return deals[:max_deals]


# ─────────────────────────────────────────────────────────────────────────────
# Source 2 — Flippa (Firecrawl scrape + regex parse)
# ─────────────────────────────────────────────────────────────────────────────

FLIPPA_PAGES = [
    "https://flippa.com/buy/sitetype/saas",
    "https://flippa.com/buy/sitetype/app",
    "https://flippa.com/buy/sitetype/services",
    "https://flippa.com/buy/plugin-and-extensions",
    "https://flippa.com/buy/media-communities/newsletters",
    "https://flippa.com/buy/sitetype/content",
    # India-specific
    "https://flippa.com/online-businesses-india",
    # Price-filtered bargains
    "https://flippa.com/search?filter%5Bproperty_type%5D%5B%5D=website&filter%5Bproperty_type%5D%5B%5D=app&filter%5Bprice%5D%5Bmax%5D=200000&filter%5Bprice%5D%5Bmin%5D=5000&filter%5Bprofit_per_month%5D%5Bmin%5D=500&sort_by=most_active",
]

FLIPPA_ID_RE  = re.compile(r"flippa\.com/(\d{7,})")
FLIPPA_PRICE_RE = re.compile(r"\$([0-9,]+(?:\.[0-9]+)?)\s*(?:USD)?")
FLIPPA_PROFIT_RE = re.compile(r"(?:profit|net|revenue)[^\n$]*\$([0-9,]+)", re.IGNORECASE)


def _parse_flippa_markdown(markdown: str, page_url: str) -> list[dict]:
    """
    Parse Flippa listing page markdown into deal dicts.

    Structure of each card in Flippa markdown:
      [...card content...](https://flippa.com/XXXXXXX)
      * * *
      Asking Price
      ##### USD $X,XXX
      Multiple: X.Xx Profit
      [Watch|View ...](...)
    """
    deals = []
    seen_ids: set[str] = set()

    # Find all card-closing patterns: `](https://flippa.com/XXXXXXX)\n\n* * *\n\nAsking Price`
    # Each card starts from the previous such boundary
    card_pattern = re.compile(
        r"\]\(https://flippa\.com/(\d{7,})\)\s*\n\n\*\s*\*\s*\*\s*\n\nAsking Price\s*\n\n"
        r"(?:#+\s+)?(?:~~USD \$[\d,]+~~\s*\n+)?"  # optional strikethrough (reduced price)
        r"USD\s+\$?([\d,]+)"                        # actual asking price
    )

    # Also find full card bodies by splitting on "* * *\n\nAsking Price"
    card_splits = re.split(r"\n\n\*\s*\*\s*\*\s*\n\nAsking Price", markdown)

    for i, seg in enumerate(card_splits[:-1]):   # last segment has no price after it
        # Find the LAST listing URL in this segment — that's this card's ID
        id_matches = list(FLIPPA_ID_RE.finditer(seg))
        if not id_matches:
            continue
        lid = id_matches[-1].group(1)
        if lid in seen_ids:
            continue
        seen_ids.add(lid)

        # Asking price is at the START of the next segment: "\n##### USD $300,000"
        # (may have strikethrough first: ~~USD $X~~\n then USD $Y)
        next_seg = card_splits[i + 1][:300]
        # Skip any strikethrough (reduced price shows both)
        ask_m = re.search(
            r"#+\s+(?:~~USD\s+\$[\d,]+~~\s*\n+)?USD\s+\$([\d,]+)",
            next_seg, re.IGNORECASE
        )
        if not ask_m:
            ask_m = re.search(r"USD\s+\$([\d,]+)", next_seg, re.IGNORECASE)
        ask = _usd("$" + ask_m.group(1)) if ask_m else None

        listing_url = f"https://flippa.com/{lid}"

        # The listing card content is at the TAIL of seg (last ~1200 chars)
        card_body = seg[-1500:]
        # Normalize Flippa's \\ continuation markers for easier parsing
        card_clean = re.sub(r"\\\\\s*\\\\\s*", "\n", card_body)
        card_clean = re.sub(r"\\\\", "", card_clean)

        # Extract description line — sentence after the country name
        title = ""
        desc_m = re.search(
            r"\b(?:India|United States|USA|UK|Australia|Canada|Singapore|Israel|Hong Kong|CA,|AU,)[^\n]*\n+([^\n\[\]]{30,200})\n",
            card_clean
        )
        if desc_m:
            title = re.sub(r"[\[\]\*#!]", "", desc_m.group(1)).strip()

        if not title or "Verified Listing" in title:
            fallback_m = re.search(
                r"((?:Established|A \d+|An? \w+)[^\n\[\]]{20,150}(?:revenue|profit|ARR|MRR|platform|solution|app|business)[^\n\[\]]*)",
                card_clean, re.IGNORECASE
            )
            title = re.sub(r"[\[\]\*#!]", "", fallback_m.group(1)).strip() if fallback_m else f"Flippa #{lid}"

        # Country (search full card)
        country = ""
        country_m = re.search(
            r"\b(India|United States|USA|UK|Australia|Canada|Singapore|Israel|Hong Kong)\b",
            card_clean
        )
        if country_m:
            country = country_m.group(1)

        # ARR/MRR mention in description → monthly revenue
        rev = None
        arr_m = re.search(r"\$([\d,]+)K?\s*(?:ARR|MRR)", card_clean, re.IGNORECASE)
        if arr_m:
            val = _usd(arr_m.group(0))
            if val:
                rev = val / 12 if "ARR" in arr_m.group(0).upper() else val

        # Net Profit p/mo (from Flippa card row)
        profit_m = re.search(r"Net Profit\s+(?:-?USD\s+)?\$?([\d,]+)\s*p/mo", card_clean, re.IGNORECASE)
        profit = _usd("$" + profit_m.group(1)) if profit_m else None
        if not profit:
            margin_m = re.search(r"([\d]+)%\s*profit\s*margin", card_clean, re.IGNORECASE)
            if margin_m and rev:
                profit = rev * int(margin_m.group(1)) / 100

        # Multiple (from next_seg after Asking Price)
        multiple = None
        mult_m = re.search(r"Multiple:\s*([\d.]+)x\s*(?:Profit|Revenue)", next_seg, re.IGNORECASE)
        if mult_m:
            try:
                multiple = float(mult_m.group(1))
            except ValueError:
                pass

        # Site age
        age_m = re.search(r"Site Age\s+([\d]+)\s*year", card_clean, re.IGNORECASE)
        age_years = float(age_m.group(1)) if age_m else None

        # Monetization
        mon_m = re.search(r"Monetization\s+([^\n\[]{5,40})", card_clean)
        monetization = re.sub(r"[\\\[\]\*#]", "", mon_m.group(1)).strip() if mon_m else ""

        deals.append(_make_deal(
            source="Flippa",
            title=title[:120] or f"Flippa #{lid}",
            url=listing_url,
            asking_price_usd=ask,
            monthly_revenue_usd=rev,
            monthly_profit_usd=profit,
            multiple=multiple,
            age_years=age_years,
            niche=page_url.split("/")[-1],
            monetization=monetization,
            country=country,
            description=seg[:700],
            listing_id=lid,
            scrape_method="firecrawl-scrape+regex",
            tags=["Flippa"],
        ))

    return deals


def fetch_flippa(max_deals: int = 20) -> list[dict]:
    """Scrape Flippa listing pages."""
    fc = _get_fc()
    if not fc:
        logger.warning("Flippa: no Firecrawl key")
        return []

    deals = []
    for page_url in FLIPPA_PAGES:
        try:
            logger.info("Flippa scrape: %s", page_url)
            doc = fc.scrape(page_url, formats=["markdown"])
            md  = getattr(doc, "markdown", "") or ""
            if md:
                page_deals = _parse_flippa_markdown(md, page_url)
                deals.extend(page_deals)
                logger.info("Flippa %s: %d deals", page_url, len(page_deals))
        except Exception as exc:
            logger.warning("Flippa %s failed: %s", page_url, exc)

    seen, unique = set(), []
    for d in deals:
        if d["listing_id"] not in seen:
            seen.add(d["listing_id"])
            unique.append(d)

    logger.info("Flippa total: %d unique deals", len(unique))
    return unique[:max_deals]


# ─────────────────────────────────────────────────────────────────────────────
# Source 3 — Reddit (PRAW OAuth)
# ─────────────────────────────────────────────────────────────────────────────

REDDIT_SUBS = [
    "SaaS", "Entrepreneur", "startups", "IndieHackers",
    "smallbusiness", "microsaas", "EntrepreneurRideAlong",
    "sidehustle", "passive_income", "Flipping",
    "india", "indianstartups",
]
# These queries are highly specific to actual sale posts
REDDIT_QUERIES = [
    '"for sale" "MRR" exit acquisition',
    '"selling my" SaaS business profitable exit',
    '"want to sell" OR "looking to sell" bootstrapped MRR',
    '"asset purchase" OR "quick close" SaaS newsletter',
    'acquired selling business "no earnout"',
    '"asking price" "monthly revenue" sale bootstrapped',
    'selling profitable website blog newsletter "asking"',
    '"India" "for sale" SaaS software business "MRR"',
]
# Post must contain at least 2 of these for-sale indicators
STRONG_EXIT_KW = [
    "for sale", "selling my", "want to sell", "looking to sell",
    "asset purchase", "quick close", "no earnout", "fast close",
    "acquisition", "acqui-hire", "want out", "ready to exit",
    "handover", "transition", "due diligence", "asking price",
]
# Post must have at least one of these
WEAK_EXIT_KW = [
    "selling", "exit", "mrr", "asking price", "valuation", "multiple",
    "buyer", "acquire", "purchase",
]


def _praw_client():
    cid  = os.environ.get("REDDIT_CLIENT_ID", "")
    csec = os.environ.get("REDDIT_CLIENT_SECRET", "")
    ua   = os.environ.get("REDDIT_USER_AGENT", "python:pocketfund:v1.0")
    if not cid or not csec:
        return None
    try:
        import praw
        r = praw.Reddit(client_id=cid, client_secret=csec, user_agent=ua)
        _ = r.subreddit("SaaS").hot(limit=1)
        return r
    except Exception as exc:
        logger.warning("PRAW init: %s", exc)
        return None


def _submission_to_deal(sub) -> Optional[dict]:
    title = sub.title or ""
    body  = sub.selftext or ""
    combined = f"{title} {body}".lower()

    # Require at least 1 strong signal OR (2+ weak signals + a price mention)
    strong_hits = sum(1 for kw in STRONG_EXIT_KW if kw in combined)
    weak_hits   = sum(1 for kw in WEAK_EXIT_KW   if kw in combined)
    has_price   = bool(re.search(r"\$([\d,]{3,})", combined))  # min $100

    if strong_hits == 0 and not (weak_hits >= 2 and has_price):
        return None
    if len(body.strip()) < 120:   # require more substance
        return None
    # Exclude "how to sell", "should I sell", "want to buy" posts
    if re.search(r"how (do|to|can)\b.*\b(sell|exit)|want to buy|looking to buy", combined):
        return None

    # Asking price: look for explicit "asking" or "priced at" mentions
    ask_m = re.search(r"asking\s+\$?([\d,]+(?:k|m)?)\b", combined, re.IGNORECASE)
    if not ask_m:
        ask_m = re.search(r"priced at\s+\$?([\d,]+(?:k|m)?)\b", combined, re.IGNORECASE)
    if not ask_m:
        ask_m = re.search(r"sale price\s+\$?([\d,]+(?:k|m)?)\b", combined, re.IGNORECASE)
    if not ask_m:
        # Fallback: any dollar amount ≥ $1,000 that isn't MRR/monthly
        ask_m_gen = re.search(r"\$([\d,]{4,}(?:k|m)?)\b", combined, re.IGNORECASE)
        ask_usd = _usd(ask_m_gen.group(0)) if ask_m_gen else None
    else:
        ask_usd = _usd("$" + ask_m.group(1))

    price_m = ask_m  # keep for compatibility
    mrr_m   = re.search(r"\$([\d,]+)\s*(?:MRR|mrr|monthly)", body, re.IGNORECASE)
    mrr_usd = _usd(mrr_m.group(0)) if mrr_m else None

    age_days = max(0, int((time.time() - sub.created_utc) / 86400))
    multiple = (ask_usd / (mrr_usd * 12)) if ask_usd and mrr_usd else None

    handle = f"u/{sub.author.name}" if sub.author else "u/deleted"
    country = "India" if re.search(r"\bindia\b|\bindian\b|₹|inr|crore|lakh|rupee", combined) else ""

    return _make_deal(
        source="Reddit",
        title=title[:120],
        url=f"https://reddit.com{sub.permalink}",
        asking_price_usd=ask_usd,
        monthly_revenue_usd=mrr_usd,
        monthly_profit_usd=mrr_usd * 0.7 if mrr_usd else None,
        multiple=multiple,
        country=country,
        description=body[:1500],
        seller_handle=handle,
        listing_id=sub.id,
        scrape_method="praw-oauth",
        tags=["Reddit", str(sub.subreddit)],
        niche="SaaS" if "saas" in combined else ("newsletter" if "newsletter" in combined else "unknown"),
    )


def fetch_reddit(max_deals: int = 30) -> list[dict]:
    """Fetch Reddit exit-signal posts via PRAW OAuth."""
    reddit = _praw_client()
    if not reddit:
        logger.warning("Reddit: no PRAW creds — trying JSON API")
        return _fetch_reddit_json(max_deals)

    deals = []
    seen: set[str] = set()

    for query in REDDIT_QUERIES:
        for sub_name in REDDIT_SUBS:
            try:
                sub = reddit.subreddit(sub_name)
                for post in sub.search(query, sort="new", limit=20, time_filter="year"):
                    if post.id in seen:
                        continue
                    seen.add(post.id)
                    deal = _submission_to_deal(post)
                    if deal:
                        deals.append(deal)
            except Exception as exc:
                logger.debug("PRAW %s/%s: %s", sub_name, query, exc)

    # Search all in combined "SaaS+Entrepreneur+microsaas" for broader coverage
    combined_sub = "SaaS+Entrepreneur+microsaas+IndieHackers+startups"
    try:
        for post in reddit.subreddit(combined_sub).search(
            '"for sale" OR "selling my" MRR exit acquisition',
            sort="new", limit=30, time_filter="year"
        ):
            if post.id in seen:
                continue
            seen.add(post.id)
            deal = _submission_to_deal(post)
            if deal:
                deals.append(deal)
    except Exception as exc:
        logger.debug("PRAW combined search: %s", exc)

    logger.info("Reddit PRAW: %d deals", len(deals))
    return deals[:max_deals]


def _fetch_reddit_json(max_deals: int = 20) -> list[dict]:
    """Unauthenticated Reddit JSON API fallback."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; PocketFundBot/1.0)"}
    urls = [
        "https://www.reddit.com/r/SaaS/search.json?q=selling+exit+for+sale&sort=new&restrict_sr=1&limit=20",
        "https://www.reddit.com/r/Entrepreneur/search.json?q=selling+my+business+exit&sort=new&restrict_sr=1&limit=20",
        "https://www.reddit.com/r/microsaas/new.json?limit=20",
        "https://www.reddit.com/r/IndieHackers/search.json?q=selling+exit+MRR&sort=new&restrict_sr=1&limit=15",
    ]
    deals = []
    seen: set[str] = set()

    for url in urls:
        try:
            resp = requests.get(url, headers=headers, timeout=10)
            resp.raise_for_status()
            posts = resp.json().get("data", {}).get("children", [])
            for post in posts:
                d = post["data"]
                pid = d.get("id", "")
                if pid in seen:
                    continue
                seen.add(pid)

                title = d.get("title", "")
                body  = d.get("selftext", "")
                combined = f"{title} {body}".lower()

                if not any(kw in combined for kw in EXIT_KW):
                    continue
                if len(body.strip()) < 40:
                    continue

                price_m = re.search(r"\$([\d,]+(?:k|m)?)", body + " " + title, re.IGNORECASE)
                ask_usd = _usd(price_m.group(0)) if price_m else None
                mrr_m   = re.search(r"\$([\d,]+)\s*(?:MRR|mrr|monthly)", body, re.IGNORECASE)
                mrr_usd = _usd(mrr_m.group(0)) if mrr_m else None
                age_days = int((time.time() - d.get("created_utc", time.time())) / 86400)
                country = "India" if re.search(r"\bindia\b|₹|crore|lakh", combined) else ""

                deals.append(_make_deal(
                    source="Reddit",
                    title=title[:120],
                    url=f"https://reddit.com{d.get('permalink','')}",
                    asking_price_usd=ask_usd,
                    monthly_revenue_usd=mrr_usd,
                    monthly_profit_usd=mrr_usd * 0.7 if mrr_usd else None,
                    country=country,
                    description=body[:1500],
                    seller_handle=f"u/{d.get('author','?')}",
                    listing_id=pid,
                    scrape_method="reddit-json",
                    tags=["Reddit", d.get("subreddit", "")],
                    niche="SaaS" if "saas" in combined else "unknown",
                ))
        except Exception as exc:
            logger.debug("Reddit JSON %s: %s", url, exc)

    logger.info("Reddit JSON: %d deals", len(deals))
    return deals[:max_deals]


# ─────────────────────────────────────────────────────────────────────────────
# Source 4 — Acquire.com RSS (public, no auth)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_acquire_rss(max_deals: int = 15) -> list[dict]:
    """Pull listings from Acquire.com public RSS feed."""
    try:
        import feedparser
        feed = feedparser.parse("https://acquire.com/rss/listings")
        if not feed.entries:
            logger.warning("Acquire RSS: empty feed")
            return []

        deals = []
        for i, entry in enumerate(feed.entries[:max_deals]):
            title   = entry.get("title", "")
            summary = entry.get("summary", entry.get("description", ""))
            body    = BeautifulSoup(summary, "html.parser").get_text(separator=" ")
            url     = entry.get("link", "")

            pub = entry.get("published_parsed")
            age = 0
            if pub:
                age = max(0, (datetime.utcnow() - datetime(*pub[:6])).days)

            price_m = re.search(r"\$([\d,]+(?:k|m)?)\b", body + " " + title, re.IGNORECASE)
            ask_usd = _usd(price_m.group(0)) if price_m else None
            mrr_m   = re.search(r"\$([\d,]+)\s*(?:MRR|mrr)", body, re.IGNORECASE)
            mrr_usd = _usd(mrr_m.group(0)) if mrr_m else None
            country = "India" if re.search(r"\bindia\b|₹|crore|lakh", (body+title).lower()) else ""

            deals.append(_make_deal(
                source="Acquire.com",
                title=title,
                url=url,
                asking_price_usd=ask_usd,
                monthly_revenue_usd=mrr_usd,
                age_years=round(age / 365, 1) if age else None,
                country=country,
                description=body[:1000],
                listing_id=f"rss-{i}",
                scrape_method="acquire-rss",
                tags=["Acquire"],
            ))

        logger.info("Acquire RSS: %d deals", len(deals))
        return deals
    except Exception as exc:
        logger.warning("Acquire RSS: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Source 5 — SideProjectors.com  (public RSS + listing pages)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_sideprojectors_rss_markdown(md: str) -> list[dict]:
    """
    Parse RSS feed returned as escaped markdown by Firecrawl.

    Each entry is on a single line like:
      PROJECTID DATE<![CDATA[ TITLE ]]>https://sideprojectors.com/project/ID/slug
    followed by description lines.

    In Firecrawl markdown, brackets are escaped: <!\[CDATA\[ ... \]\]>
    """
    deals = []
    lines = md.split("\n")

    # Collect (title, listing_id, url, description_lines) per entry
    entries = []
    current: dict | None = None

    for line in lines:
        # Check for entry header: ID DATE <!\[CDATA\[ TITLE \]\]> [URL]
        # Pattern matches e.g. "52942 2026-02-24 ...<!\[CDATA\[ For Sale - ... \]\]>https://..."
        m = re.search(
            r"(\d{4,7})\s*\d{4}-\d{2}-\d{2}[^\[]*"
            r"<!\\\[CDATA\\\[\s*(.*?)\s*\\\]\\\]>"
            r"(https?://\S*)?",
            line
        )
        if m:
            if current:
                entries.append(current)
            current = {
                "listing_id": m.group(1),
                "raw_title": m.group(2).strip(),
                "url": (m.group(3) or "").strip(),
                "desc_lines": [],
            }
        elif current and line.strip() and not line.strip().startswith(("**[", "![", "##", "- **")):
            # Collect description lines (skip markdown formatting noise)
            clean = re.sub(r"\*+|\\\*|\[.*?\]\(.*?\)", "", line).strip()
            if 15 < len(clean) < 300:
                current["desc_lines"].append(clean)

    if current:
        entries.append(current)

    for entry in entries:
        raw_title  = entry["raw_title"]
        listing_id = entry["listing_id"]
        url        = entry["url"]

        # Only interested in "For Sale" entries
        if not re.search(r"for sale|for sell", raw_title, re.IGNORECASE):
            continue

        # Price from title: "For Sale - $49,000 USD : TITLE" or "For Sale - 299 USD : TITLE"
        price_m = re.search(r"[-–]\s*\$?([\d,]+(?:\.\d+)?)\s*(?:USD)?\s*:", raw_title, re.IGNORECASE)
        ask_usd = None
        if price_m:
            try:
                val = float(price_m.group(1).replace(",", ""))
                if val >= 5:
                    ask_usd = val
            except ValueError:
                pass

        # Title: text after "For Sale - $PRICE USD : "
        title_m = re.search(r":\s*(.+)$", raw_title, re.IGNORECASE)
        title = title_m.group(1).strip() if title_m else raw_title
        if not title or len(title) < 5:
            continue

        description = " ".join(entry["desc_lines"][:4])[:600]
        combined    = f"{raw_title} {description}".lower()
        country     = "India" if re.search(r"\bindia\b|\bindian\b|₹|lakh|crore", combined) else ""
        niche       = "SaaS"
        for cat in ["newsletter", "ecommerce", "app", "plugin", "tool", "content", "saas", "shopify"]:
            if cat in combined:
                niche = cat.title()
                break

        sp_url = url if url else f"https://www.sideprojectors.com/project/{listing_id}/"

        deals.append(_make_deal(
            source="SideProjectors",
            title=title[:120],
            url=sp_url,
            asking_price_usd=ask_usd,
            monthly_revenue_usd=None,
            monthly_profit_usd=None,
            multiple=None,
            country=country,
            description=description,
            seller_handle="",
            listing_id=listing_id,
            scrape_method="rss-firecrawl",
            tags=["SideProjectors", niche],
            niche=niche,
        ))

    return deals


def _parse_sideprojectors_page(markdown: str) -> list[dict]:
    """
    Parse Firecrawl markdown from SideProjectors /project/index?wants_to_sell=1.
    Each listing has:
      'For sale ⋅ > $PRICE'  line
      then title on the next non-image/non-url line
      then 'Posted on DATE'
      then description text
    """
    deals = []
    seen_ids: set[str] = set()

    # Split on "For sale" markers
    blocks = re.split(r"(?=For sale\s*[⋅·•])", markdown)

    for block in blocks:
        if not re.search(r"For sale\s*[⋅·•]", block, re.IGNORECASE):
            continue

        # Price: "For sale ⋅ > $780"
        price_m = re.search(r">\s*\$\s*([\d,]+(?:\.\d+)?)\s*([kKmM](?![a-zA-Z]))?", block)
        ask_usd = None
        if price_m:
            try:
                val = float(price_m.group(1).replace(",", ""))
                suffix = (price_m.group(2) or "").lower()
                if suffix == "k":   val *= 1_000
                elif suffix == "m": val *= 1_000_000
                if val >= 5:
                    ask_usd = val
            except ValueError:
                pass

        # SideProjectors project URL (internal)
        sp_url_m = re.search(r"https://www\.sideprojectors\.com/project/(\d+)/[\w-]+", block)
        listing_id = sp_url_m.group(1) if sp_url_m else ""
        url = sp_url_m.group(0) if sp_url_m else ""

        if not url or listing_id in seen_ids:
            continue
        if listing_id:
            seen_ids.add(listing_id)

        # Title: line that looks like a product name (not a URL, image, date, tag)
        lines = [l.strip() for l in block.split("\n") if l.strip()]
        title = ""
        for line in lines:
            # Skip URL-only lines like [https://foo.com](https://foo.com)
            if re.match(r"^\[?https?://", line):
                continue
            if (len(line) > 6 and len(line) < 80
                    and not line.startswith(("#", "[!", "(", "For sale",
                                            "Posted", "Project", "Joined", "From",
                                            "Publishing", "Finance", "SaaS", "Web"))
                    and "sideprojectors.com" not in line.lower()
                    and "![" not in line
                    and not re.search(r"^\*{2,}\[", line)):
                title = line.strip("*").strip()
                break

        if not title:
            continue

        # Description: "Project description" section
        desc_m = re.search(r"Project description\s*\n(.{30,500}?)(?:\nHow this|$)", block, re.DOTALL)
        description = desc_m.group(1).strip()[:500] if desc_m else ""
        if not description:
            # Fallback: first decent paragraph
            desc_lines = [l.strip() for l in block.split("\n")
                          if 25 < len(l.strip()) < 200
                          and not l.strip().startswith(("[", "!", "#", "http", "www"))]
            description = " ".join(desc_lines[:3])[:400]

        combined = (title + " " + description).lower()
        country  = "India" if re.search(r"\bindia\b|\bindian\b|₹|lakh|crore", combined) else ""
        niche    = "SaaS"
        for cat in ["newsletter", "ecommerce", "app", "plugin", "tool", "content", "saas"]:
            if cat in combined:
                niche = cat.title()
                break

        deals.append(_make_deal(
            source="SideProjectors",
            title=title[:120],
            url=url,
            asking_price_usd=ask_usd,
            monthly_revenue_usd=None,
            monthly_profit_usd=None,
            multiple=None,
            country=country,
            description=description,
            seller_handle="",
            listing_id=listing_id,
            scrape_method="firecrawl-page",
            tags=["SideProjectors", niche],
            niche=niche,
        ))
    return deals


def fetch_side_projectors(max_deals: int = 40) -> list[dict]:
    """
    Fetch listings from SideProjectors.com via:
      1. RSS feed scraped through Firecrawl (bypasses direct 403)
      2. Listing index page scrape for richer data
    """
    fc = _get_fc()
    deals: list[dict] = []
    seen_urls: set[str] = set()

    # RSS via Firecrawl (bypasses direct 403 on RSS endpoint)
    if fc:
        try:
            rss_doc = fc.scrape("https://www.sideprojectors.com/rss", formats=["markdown"])
            rss_md  = rss_doc.markdown or ""
            rss_deals = _parse_sideprojectors_rss_markdown(rss_md)
            for d in rss_deals:
                key = d["url"].rstrip("/")
                if key not in seen_urls:
                    seen_urls.add(key)
                    deals.append(d)
            logger.info("SideProjectors RSS: %d deals", len(rss_deals))
        except Exception as exc:
            logger.warning("SideProjectors RSS (FC): %s", exc)

    # Listing index page for structured data
    if fc:
        try:
            for page_url in [
                "https://www.sideprojectors.com/project/index?wants_to_sell=1",
            ]:
                doc = fc.scrape(page_url, formats=["markdown"])
                md  = doc.markdown or ""
                page_deals = _parse_sideprojectors_page(md)
                for d in page_deals:
                    key = d["url"].rstrip("/")
                    if key not in seen_urls:
                        seen_urls.add(key)
                        deals.append(d)
                logger.info("SideProjectors page: %d additional deals", len(page_deals))
        except Exception as exc:
            logger.warning("SideProjectors page scrape: %s", exc)

    return deals[:max_deals]


# ─────────────────────────────────────────────────────────────────────────────
# Source 6 — Firecrawl web search for live deal signals
# ─────────────────────────────────────────────────────────────────────────────

HUNT_QUERIES = [
    "SaaS profitable exit for sale MRR bootstrapped 2026",
    "selling micro SaaS business MRR profitable asset purchase",
    "India SaaS business for sale acquisition MRR profitable",
    "newsletter for sale email list MRR exit 2026",
    "profitable web app tool for sale bootstrap exit founder",
    "sideprojectors flippa selling website blog $5k $50k 2026",
]


def fetch_web_search_deals(max_deals: int = 10) -> list[dict]:
    """Use Firecrawl web search to surface live deal signals."""
    fc = _get_fc()
    if not fc:
        return []

    deals = []
    seen_urls: set[str] = set()

    for query in HUNT_QUERIES[:3]:
        try:
            result = fc.search(query, limit=5)
            items  = getattr(result, "web", None) or []
            for item in items:
                url   = getattr(item, "url", "") or ""
                title = getattr(item, "title", "") or ""
                desc  = getattr(item, "description", "") or ""
                if not url or url in seen_urls:
                    continue
                # Only include results that look like actual deal listings
                combined = (title + " " + desc).lower()
                if not any(kw in combined for kw in ["for sale", "exit", "mrr", "acquisition", "selling"]):
                    continue
                seen_urls.add(url)
                price_m = re.search(r"\$([\d,]+(?:k|m)?)\b", combined, re.IGNORECASE)
                ask     = _usd(price_m.group(0)) if price_m else None
                mrr_m   = re.search(r"\$([\d,]+)\s*(?:mrr|monthly)", combined, re.IGNORECASE)
                mrr     = _usd(mrr_m.group(0)) if mrr_m else None
                country = "India" if re.search(r"\bindia\b|₹", combined) else ""

                deals.append(_make_deal(
                    source="WebSearch",
                    title=title[:120],
                    url=url,
                    asking_price_usd=ask,
                    monthly_revenue_usd=mrr,
                    country=country,
                    description=desc[:600],
                    scrape_method="firecrawl-search",
                    tags=["WebSearch"],
                ))
        except Exception as exc:
            logger.warning("Web search '%s': %s", query[:40], exc)

    logger.info("Web search deals: %d", len(deals))
    return deals[:max_deals]


# ─────────────────────────────────────────────────────────────────────────────
# Scorer — quick deal signal score for ranking
# ─────────────────────────────────────────────────────────────────────────────

def score_deal(deal: dict) -> float:
    """
    Score 0-100. Higher = more attractive for micro-PE.
    Weights: price range (sweet spot <$300k) + profitable + India signal + exit urgency.
    """
    score = 0.0
    desc = (deal.get("description", "") + " " + deal.get("title", "") + " " + deal.get("niche", "")).lower()

    # Price range score (sweet spot: $15k-$300k / ₹1-25Cr)
    ask = deal.get("asking_price_usd") or 0
    if 15_000 < ask <= 150_000:
        score += 30       # ideal micro-PE range
    elif 150_000 < ask <= 350_000:
        score += 20       # stretch but viable
    elif ask > 0:
        score += 8        # too cheap or too expensive

    # Has verified profit/revenue data (structured source premium)
    if deal.get("monthly_profit_usd"):
        score += 18
    if deal.get("monthly_revenue_usd"):
        score += 8

    # Multiple attractiveness
    mult = deal.get("multiple") or 0.0
    if 0 < mult <= 18:
        score += 20       # excellent deal
    elif 18 < mult <= 30:
        score += 12       # fair
    elif 30 < mult <= 40:
        score += 5        # market rate
    # > 40x: no bonus

    # India signal (+premium for India-focused fund)
    if deal.get("country") == "India" or re.search(r"\bindia\b|₹|lakh|crore|indian\b", desc):
        score += 18

    # Exit urgency signals
    urgency_kw = [
        "burnout", "tired", "quick close", "no earnout", "fast close",
        "moving on", "new opportunity", "full-time job", "vc-backed",
        "solo founder", "maintenance mode", "day job", "join a",
        "feature complete", "runs itself", "autopilot",
    ]
    score += min(sum(2 for kw in urgency_kw if kw in desc), 12)

    # SaaS / recurring revenue premium
    if re.search(r"\bsaas\b|\bsubscription\b|\brecurring\b|\bmrr\b|\barr\b", desc):
        score += 6

    # Source quality bonus
    if deal.get("source") in ("Acquire.com",):
        score += 8   # verified, structured, NDA gated
    elif deal.get("source") == "EmpireFlippers":
        score += 5   # vetted listings
    elif deal.get("source") == "Flippa":
        score += 3
    elif deal.get("source") == "SideProjectors":
        score += 2

    # Description length / depth bonus
    if len(deal.get("description","")) > 300:
        score += 3
    if deal.get("highlights"):
        score += 2

    return min(score, 100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Master orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def hunt_live_deals(
    use_acquire_auth: bool = True,
    use_ef: bool = True,
    use_flippa: bool = True,
    use_reddit: bool = True,
    use_acquire_rss: bool = True,
    use_web_search: bool = True,
    use_side_projectors: bool = True,
    max_total: int = 150,
) -> list[dict]:
    """
    Fetch from all enabled sources, deduplicate, score, and sort by attractiveness.
    Returns list of deal dicts sorted by score descending.
    """
    all_deals: list[dict] = []

    # Acquire.com authenticated (best quality — verified, structured data)
    if use_acquire_auth and os.environ.get("ACQUIRE_EMAIL"):
        try:
            from engine.acquire_auth import fetch_acquire_authenticated, score_acquire_deal
            acq_deals = fetch_acquire_authenticated(max_deals=60)
            for d in acq_deals:
                d["score"] = score_acquire_deal(d)
            all_deals.extend(acq_deals)
            logger.info("Acquire auth: %d deals", len(acq_deals))
        except Exception as exc:
            logger.warning("Acquire auth fetch error: %s", exc)

    if use_acquire_rss:
        all_deals.extend(fetch_acquire_rss(20))
    if use_side_projectors:
        try:
            sp_deals = fetch_side_projectors(30)
            all_deals.extend(sp_deals)
            logger.info("SideProjectors: %d deals", len(sp_deals))
        except Exception as exc:
            logger.warning("SideProjectors error: %s", exc)
    if use_reddit:
        all_deals.extend(fetch_reddit(40))
    if use_ef:
        all_deals.extend(fetch_empire_flippers(20))
    if use_flippa:
        all_deals.extend(fetch_flippa(30))
    if use_web_search:
        all_deals.extend(fetch_web_search_deals(15))

    # Deduplicate by URL
    seen, unique = set(), []
    for d in all_deals:
        key = d["url"].rstrip("/")
        if key not in seen:
            seen.add(key)
            d["score"] = score_deal(d)
            unique.append(d)

    # Secondary dedup: collapse listings with very similar titles (same deal, different source)
    title_seen: set[str] = set()
    final: list[dict] = []
    for d in sorted(unique, key=lambda x: x["score"], reverse=True):
        title_key = re.sub(r"[^a-z0-9]", "", d["title"].lower())[:40]
        if title_key not in title_seen:
            title_seen.add(title_key)
            final.append(d)

    logger.info(
        "Live deal hunt: %d raw → %d URL-unique → %d title-unique",
        len(all_deals), len(unique), len(final)
    )
    return final[:max_total]
