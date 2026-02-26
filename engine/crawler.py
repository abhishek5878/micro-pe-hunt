"""
Firecrawl-powered web scraper for the Acquisition Intelligence Layer.
Uses Firecrawl v2 SDK (firecrawl-py >= 1.0).

Four modes:
1. scrape_listing_url()   → scrape any listing page → structured deal dict
2. crawl_acquire_listings()→ extract deal cards from Acquire.com
3. scrape_founder_profile()→ extract founder signals from social/profile URL
4. research_url()          → return clean markdown from any URL for agent context

Falls back gracefully at every level — Firecrawl → BS4 → empty dict.
"""

import json
import logging
import os
import re
from typing import Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

USD_TO_INR = 83.5


# ─────────────────────────────────────────────────────────────────────────────
# SDK factory
# ─────────────────────────────────────────────────────────────────────────────

def _get_fc():
    """Return an authenticated FirecrawlApp or None."""
    api_key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not api_key:
        return None
    try:
        from firecrawl import FirecrawlApp
        return FirecrawlApp(api_key=api_key)
    except Exception as exc:
        logger.warning("Firecrawl init failed: %s", exc)
        return None


def _md(doc) -> str:
    """Extract markdown string from a Firecrawl Document object."""
    return getattr(doc, "markdown", "") or ""


def _meta(doc) -> dict:
    """Extract metadata dict from a Firecrawl Document."""
    m = getattr(doc, "metadata", None)
    if m is None:
        return {}
    return dict(m) if not isinstance(m, dict) else m


# ─────────────────────────────────────────────────────────────────────────────
# Currency parsers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_usd(text: str) -> Optional[float]:
    if not text:
        return None
    text = str(text).replace(",", "").replace(" ", "")
    m = re.search(r"\$?([\d]+(?:\.[\d]+)?)\s*([kKmM])?", text)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    suffix = (m.group(2) or "").lower()
    if suffix == "k":   val *= 1_000
    elif suffix == "m": val *= 1_000_000
    return val


def _parse_inr_cr(text: str) -> Optional[float]:
    """Parse INR text → Crore float."""
    if not text:
        return None
    text = str(text).lower().replace(",", "")
    m = re.search(r"([\d.]+)\s*(?:cr|crore)", text)
    if m:
        return float(m.group(1))
    m = re.search(r"([\d.]+)\s*(?:l\b|lakh|lakhs)", text)
    if m:
        return round(float(m.group(1)) / 100, 4)
    return None


def _detect_source(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if "acquire"  in domain: return "Acquire.com"
    if "flippa"   in domain: return "Flippa"
    if "reddit"   in domain: return "Reddit"
    if "twitter"  in domain or "x.com" in domain: return "X/Twitter"
    if "linkedin" in domain: return "LinkedIn"
    return "Web"


# ─────────────────────────────────────────────────────────────────────────────
# Parse helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_deal_from_markdown(markdown: str, url: str, method: str = "firecrawl-markdown") -> dict:
    """Build a deal dict from raw scraped markdown."""
    title = ""
    m = re.search(r"^#{1,2}\s+(.+)$", markdown, re.MULTILINE)
    if m:
        title = m.group(1).strip()

    # Extract price/mrr hints
    price_str, mrr_str = "", ""
    for line in markdown.splitlines():
        line_l = line.lower()
        if re.search(r"asking|price|valuation|listed.?at", line_l):
            price_str = line
        if re.search(r"mrr|monthly recurring|monthly revenue", line_l):
            mrr_str = line

    price_usd = _parse_usd(price_str)
    mrr_usd   = _parse_usd(mrr_str)
    price_inr = _parse_inr_cr(price_str)

    if not price_inr and price_usd:
        price_inr = round(price_usd * USD_TO_INR / 10_000_000, 4)

    revenue_cr = 0.0
    if mrr_usd:
        revenue_cr = round(mrr_usd * 12 * USD_TO_INR / 10_000_000, 4)

    return {
        "title":               title or urlparse(url).netloc,
        "description":         markdown[:3000],
        "url":                 url,
        "source":              _detect_source(url),
        "scrape_source":       method,
        "asking_price_usd":    price_usd,
        "asking_price_inr_cr": price_inr,
        "mrr_usd":             mrr_usd,
        "revenue_cr":          revenue_cr,
        "ebitda_l":            0.0,
        "reason_for_sale":     "",
        "sector_hint":         "",
        "location":            "",
        "highlights":          [],
        "risks":               [],
        "seller_handle":       "",
        "india_gst_registered": None,
        "india_family_run":    None,
        "india_state":         "",
        "raw_markdown":        markdown[:5000],
    }


def _build_deal_from_extract(extracted: dict, url: str) -> dict:
    """Normalise Firecrawl extract() output → deal dict."""
    price_str  = str(extracted.get("asking_price", ""))
    mrr_str    = str(extracted.get("mrr", ""))
    rev_str    = str(extracted.get("revenue_annual", ""))
    ebitda_str = str(extracted.get("ebitda", ""))

    price_inr = _parse_inr_cr(price_str)
    price_usd = _parse_usd(price_str) if not price_inr else None
    mrr_usd   = _parse_usd(mrr_str)
    ebitda_usd = _parse_usd(ebitda_str)
    rev_usd    = _parse_usd(rev_str)

    if not price_inr and price_usd:
        price_inr = round(price_usd * USD_TO_INR / 10_000_000, 4)

    revenue_cr = 0.0
    if rev_usd:
        revenue_cr = round(rev_usd * USD_TO_INR / 10_000_000, 4)
    elif mrr_usd:
        revenue_cr = round(mrr_usd * 12 * USD_TO_INR / 10_000_000, 4)

    ebitda_l = 0.0
    if ebitda_usd:
        ebitda_l = round(ebitda_usd * USD_TO_INR / 100_000, 1)

    india = extracted.get("india_signals") or {}
    if isinstance(india, str):
        india = {}

    return {
        "title":               extracted.get("business_name") or urlparse(url).netloc,
        "description":         extracted.get("full_description", ""),
        "url":                 url,
        "source":              _detect_source(url),
        "scrape_source":       "firecrawl-extract",
        "asking_price_usd":    price_usd,
        "asking_price_inr_cr": price_inr,
        "mrr_usd":             mrr_usd,
        "revenue_cr":          revenue_cr,
        "ebitda_l":            ebitda_l,
        "reason_for_sale":     extracted.get("reason_for_sale", ""),
        "sector_hint":         extracted.get("sector", ""),
        "location":            extracted.get("location", ""),
        "customers":           extracted.get("customers", ""),
        "employees":           extracted.get("employees", ""),
        "highlights":          extracted.get("highlights") or [],
        "risks":               extracted.get("risks") or [],
        "seller_handle":       extracted.get("seller_handle", ""),
        "india_gst_registered": india.get("gst_registered"),
        "india_family_run":    india.get("family_business"),
        "india_state":         india.get("state", ""),
        "raw_markdown":        "",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Mode 1 — Scrape a single listing URL
# ─────────────────────────────────────────────────────────────────────────────

DEAL_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "business_name":    {"type": "string"},
        "asking_price":     {"type": "string"},
        "mrr":              {"type": "string"},
        "arr":              {"type": "string"},
        "revenue_annual":   {"type": "string"},
        "ebitda":           {"type": "string"},
        "sector":           {"type": "string"},
        "location":         {"type": "string"},
        "reason_for_sale":  {"type": "string"},
        "founded_year":     {"type": "string"},
        "employees":        {"type": "string"},
        "customers":        {"type": "string"},
        "seller_handle":    {"type": "string"},
        "full_description": {"type": "string"},
        "highlights":       {"type": "array", "items": {"type": "string"}},
        "risks":            {"type": "array", "items": {"type": "string"}},
        "india_signals":    {
            "type": "object",
            "properties": {
                "gst_registered":  {"type": "boolean"},
                "family_business": {"type": "boolean"},
                "state":           {"type": "string"},
            },
        },
    },
    "required": ["business_name", "full_description"],
}


def scrape_listing_url(url: str) -> dict:
    """
    Scrape a listing URL and return a structured deal dict.
    Priority: Firecrawl extract() → Firecrawl scrape() → BS4
    """
    fc = _get_fc()

    # ── 1. Firecrawl extract (LLM-powered structured output) ─────────────────
    if fc:
        try:
            logger.info("Firecrawl extract: %s", url)
            result = fc.extract(
                urls=[url],
                prompt=(
                    "Extract all available information about this business listing for sale. "
                    "Focus on financials (MRR, ARR, revenue, EBITDA), asking price, "
                    "reason for sale, and any India-specific signals (GST, location, family business). "
                    "Return exact numbers as stated."
                ),
                schema=DEAL_EXTRACT_SCHEMA,
            )
            # ExtractResponse.data is a dict keyed by schema field (not a list)
            raw = getattr(result, "data", None) or {}
            extracted = raw if isinstance(raw, dict) else {}
            if extracted.get("business_name") or extracted.get("full_description"):
                logger.info("Firecrawl extract succeeded for %s", url)
                return _build_deal_from_extract(extracted, url)
        except Exception as exc:
            logger.warning("Firecrawl extract failed (%s) — trying scrape", exc)

        # ── 2. Firecrawl raw markdown scrape ─────────────────────────────────
        try:
            logger.info("Firecrawl scrape: %s", url)
            doc = fc.scrape(url, formats=["markdown"])
            markdown = _md(doc)
            if markdown and len(markdown) > 200:
                logger.info("Firecrawl scrape OK: %d chars", len(markdown))
                return _build_deal_from_markdown(markdown, url, "firecrawl-scrape")
        except Exception as exc:
            logger.warning("Firecrawl scrape failed (%s) — using BS4", exc)

    # ── 3. BS4 fallback ───────────────────────────────────────────────────────
    logger.info("BS4 scrape: %s", url)
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0 (compatible; PocketFundBot/1.0)"}, timeout=12)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "lxml")
        h1   = soup.find("h1") or soup.find("h2") or soup.find("title")
        text = soup.get_text(separator=" ", strip=True)[:3000]
        return {
            **_build_deal_from_markdown(text, url, "bs4-fallback"),
            "title": h1.get_text(strip=True) if h1 else urlparse(url).netloc,
        }
    except Exception as exc:
        logger.error("BS4 failed for %s: %s", url, exc)
        return {"title": url, "description": "", "url": url, "source": _detect_source(url),
                "scrape_source": "failed", "revenue_cr": 0.0, "ebitda_l": 0.0}


# ─────────────────────────────────────────────────────────────────────────────
# Mode 2 — Search + crawl Acquire.com / web
# ─────────────────────────────────────────────────────────────────────────────

def crawl_acquire_listings(max_listings: int = 8) -> list[dict]:
    """
    Use Firecrawl search to find fresh Acquire.com listings, then scrape each.
    Falls back to RSS if no Firecrawl key.
    """
    fc = _get_fc()
    results = []

    if fc:
        try:
            logger.info("Firecrawl search: Acquire.com listings")
            search_result = fc.search(
                "site:acquire.com selling SaaS MRR exit",
                limit=max_listings,
            )
            items = getattr(search_result, "web", None) or getattr(search_result, "data", []) or []
            for item in items[:max_listings]:
                item_url   = getattr(item, "url", "")         or ""
                item_title = getattr(item, "title", "")       or ""
                item_desc  = getattr(item, "description", "") or getattr(item, "markdown", "") or ""
                if not item_url:
                    continue
                d = _build_deal_from_markdown(item_desc or item_title, item_url, "firecrawl-search")
                d["title"] = item_title or d["title"]
                results.append(d)

            if results:
                logger.info("Acquire search: %d results", len(results))
                return results
        except Exception as exc:
            logger.warning("Firecrawl Acquire search failed: %s — RSS fallback", exc)

    return _acquire_rss_fallback(max_listings)


def _acquire_rss_fallback(max_listings: int) -> list[dict]:
    """RSS feed fallback for Acquire.com."""
    try:
        import feedparser
        feed = feedparser.parse("https://acquire.com/rss/listings")
        results = []
        for entry in feed.entries[:max_listings]:
            soup = BeautifulSoup(entry.get("summary", ""), "html.parser")
            body = soup.get_text(separator=" ")
            d = _build_deal_from_markdown(body, entry.get("link", ""), "rss-fallback")
            d["title"] = entry.get("title", d["title"])
            results.append(d)
        return results
    except Exception as exc:
        logger.error("RSS fallback failed: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Mode 3 — Founder / company profile scraping
# ─────────────────────────────────────────────────────────────────────────────

FOUNDER_EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "name":             {"type": "string"},
        "bio":              {"type": "string"},
        "current_role":     {"type": "string"},
        "location":         {"type": "string"},
        "recent_activity":  {"type": "string"},
        "exit_signals":     {"type": "string"},
        "burnout_signals":  {"type": "string"},
        "momentum_signals": {"type": "string"},
    },
    "required": ["bio"],
}


def scrape_founder_profile(url: str) -> dict:
    """
    Scrape a founder's Twitter/X, LinkedIn, or personal site.
    Returns signals for the Psychologist agent.
    """
    fc = _get_fc()
    if not fc:
        return {"bio": "", "recent_activity": "", "exit_signals": ""}

    # Try extract first for structured founder data
    try:
        result = fc.extract(
            urls=[url],
            prompt=(
                "Extract this person's bio, current role, location, recent public posts/activity, "
                "and any signals they want to sell or exit a business, are burned out, "
                "or are working on something new."
            ),
            schema=FOUNDER_EXTRACT_SCHEMA,
        )
        raw = getattr(result, "data", None) or {}
        if isinstance(raw, dict) and raw.get("bio"):
            return raw
    except Exception as exc:
        logger.warning("Founder profile extract failed: %s", exc)

    # Fallback: raw scrape
    try:
        doc = fc.scrape(url, formats=["markdown"])
        markdown = _md(doc)
        return {"bio": markdown[:800], "recent_activity": "", "exit_signals": ""}
    except Exception as exc:
        logger.warning("Founder profile scrape failed: %s", exc)
        return {"bio": "", "recent_activity": "", "exit_signals": ""}


# ─────────────────────────────────────────────────────────────────────────────
# Mode 4 — Research URL → clean markdown
# ─────────────────────────────────────────────────────────────────────────────

def research_url(url: str) -> str:
    """Scrape any URL → clean markdown for agent context enrichment."""
    fc = _get_fc()
    if fc:
        try:
            doc = fc.scrape(url, formats=["markdown"])
            md  = _md(doc)
            if md:
                return md[:6000]
        except Exception as exc:
            logger.warning("Research scrape failed: %s", exc)

    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        soup = BeautifulSoup(resp.text, "lxml")
        return soup.get_text(separator="\n", strip=True)[:4000]
    except Exception as exc:
        logger.error("Research BS4 fallback: %s", exc)
        return ""


# ─────────────────────────────────────────────────────────────────────────────
# Mode 5 — Web search for deal/sector research
# ─────────────────────────────────────────────────────────────────────────────

def web_search_deals(query: str, limit: int = 5) -> list[dict]:
    """
    Use Firecrawl search to find relevant web results for a research query.
    Returns list of {title, url, snippet} dicts.

    SearchData.web is a list of SearchResultWeb(url, title, description, ...)
    """
    fc = _get_fc()
    if not fc:
        return []
    try:
        result = fc.search(query, limit=limit)
        # Use .web for organic results; fall back to checking all known fields
        items = getattr(result, "web", None) or []
        if not items:
            items = getattr(result, "data", None) or []
        return [
            {
                "title":   getattr(i, "title", "") or "",
                "url":     getattr(i, "url", "")   or "",
                "snippet": (getattr(i, "description", "") or getattr(i, "markdown", "") or "")[:300],
            }
            for i in items
        ]
    except Exception as exc:
        logger.warning("Firecrawl search failed: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Status check
# ─────────────────────────────────────────────────────────────────────────────

def check_firecrawl_status() -> dict:
    """Verify Firecrawl key is valid with a lightweight test scrape."""
    api_key = os.environ.get("FIRECRAWL_API_KEY", "")
    if not api_key:
        return {"connected": False, "reason": "No API key"}
    try:
        from firecrawl import FirecrawlApp
        fc  = FirecrawlApp(api_key=api_key)
        doc = fc.scrape("https://example.com", formats=["markdown"])
        md  = _md(doc)
        return {
            "connected":  bool(md),
            "key_last4":  api_key[-4:],
            "test_chars": len(md),
        }
    except Exception as exc:
        return {"connected": False, "reason": str(exc)}
