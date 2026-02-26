"""
Acquire.com Authenticated Scraper — Playwright + individual listing detail pages.

Approach:
  1. Login with stored credentials
  2. Collect listing URLs from browse page (+ filter tabs for variety)
  3. Scrape EACH individual listing page for full detail:
     - Description, highlights, key metrics, financial history
     - Asking price reasoning, seller notes, India signals
  4. Return normalised LiveDeal dicts with rich context for the agent pipeline
"""

import asyncio
import logging
import os
import re
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

USD_TO_INR = 83.5

# ─────────────────────────────────────────────────────────────────────────────
# Deal normalisation
# ─────────────────────────────────────────────────────────────────────────────

def _usd(text: str) -> Optional[float]:
    if not text:
        return None
    text = str(text).replace(",", "").strip()
    m = re.search(r"\$?([\d]+(?:\.[\d]+)?)\s*([kKmM](?![a-zA-Z]))?", text)
    if not m:
        return None
    try:
        val = float(m.group(1))
    except ValueError:
        return None
    s = (m.group(2) or "").lower()
    if s == "k":   val *= 1_000
    elif s == "m": val *= 1_000_000
    return val


def _make_deal(
    title, url, ttm_rev, ttm_profit, ask, last_month_rev=None,
    last_month_profit=None, sector="", description="",
    listing_id="", country="", highlights=None, tags=None,
    asking_price_reasoning="", seller_notes="",
) -> dict:
    mrr_usd   = last_month_rev or (ttm_rev / 12 if ttm_rev else None)
    profit_mo = last_month_profit or (ttm_profit / 12 if ttm_profit else None)
    price_inr = round(ask * USD_TO_INR / 10_000_000, 3) if ask else None
    rev_cr    = round(ttm_rev * USD_TO_INR / 10_000_000, 4) if ttm_rev else 0.0
    mult      = round(ask / ttm_profit, 1) if (ask and ttm_profit and ttm_profit > 0) else None

    desc_parts = []
    if description:      desc_parts.append(description[:800])
    if highlights:       desc_parts.append("Highlights: " + "; ".join(highlights[:4]))
    if asking_price_reasoning: desc_parts.append("Pricing: " + asking_price_reasoning[:200])
    full_desc = "\n\n".join(desc_parts)

    return {
        "source":               "Acquire.com",
        "title":                title[:120],
        "url":                  url,
        "asking_price_usd":     ask,
        "asking_price_inr_cr":  price_inr,
        "monthly_profit_usd":   profit_mo,
        "monthly_revenue_usd":  mrr_usd,
        "multiple":             mult,
        "arr_usd":              ttm_rev,
        "ebitda_usd":           ttm_profit,
        "revenue_cr":           rev_cr,
        "ebitda_l":             round(ttm_profit * USD_TO_INR / 100_000, 1) if ttm_profit else 0.0,
        "niche":                sector,
        "monetization":         sector,
        "age_years":            None,
        "country":              country,
        "description":          full_desc[:1500],
        "highlights":           highlights or [],
        "seller_handle":        "",
        "listing_id":           listing_id,
        "scrape_method":        "playwright-auth-detail",
        "tags":                 tags or ["Acquire.com", sector],
        "fetched_at":           datetime.utcnow().isoformat(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Individual listing page parser
# ─────────────────────────────────────────────────────────────────────────────

def _parse_listing_page(text: str, url: str) -> Optional[dict]:
    """Parse the inner_text() of an individual Acquire.com /startup/ page."""
    if len(text) < 200:
        return None

    # Title: first substantial description line
    title = ""
    # Typical structure: "Startup type\nCountry\nUpgrade to unlock...\nDESCRIPTION_TITLE"
    desc_section = text[text.find("ASKING PRICE") - 2000 if "ASKING PRICE" in text else 0:]
    title_m = re.search(
        r"(?:Upgrade to unlock startup name\n|SaaS Startup\n|AI Startup\n|"
        r"Mobile Startup\n|Ecommerce Startup\n|Content Startup\n|Newsletter Startup\n)"
        r"([^\n]{20,200})\n",
        text, re.IGNORECASE
    )
    if title_m:
        title = title_m.group(1).strip()
    if not title:
        # Try the first long line after "India" or after the startup type
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        for i, line in enumerate(lines[:30]):
            if len(line) > 30 and not any(x in line for x in [
                "PRICE", "PROFIT", "REVENUE", "MULTIPLES", "GROWTH", "Upgrade", "Add to", "Listings",
                "My deals", "Inbox", "Conduct", "community", "Browse", "Need help"
            ]):
                title = line
                break

    if not title:
        return None

    # Country
    country = ""
    if re.search(r"\bIndia\b", text[:1000]):
        country = "India"
    elif re.search(r"\bUnited States\b|\bUSA\b|\bUS\b", text[:1000]):
        country = "United States"
    elif re.search(r"\bUK\b|\bUnited Kingdom\b", text[:1000]):
        country = "UK"

    # Sector
    sector = ""
    for s in ["SaaS", "AI", "Ecommerce", "Mobile", "Newsletter", "Content", "Marketplace",
              "Shopify App", "Chrome Extension", "Digital"]:
        if s.lower() in text.lower()[:500]:
            sector = s
            break

    # Financial metrics
    ask_m    = re.search(r"ASKING PRICE\s*\n\s*\$?([\d.,]+[kKmM]?)", text)
    ttm_rev_m   = re.search(r"TTM REVENUE\s*\n\s*\$?([\d.,]+[kKmM]?)", text)
    ttm_prof_m  = re.search(r"(?:TTM )?PROFIT\s*\n\s*\$?([\d.,]+[kKmM]?)", text)
    lm_rev_m    = re.search(r"LAST (?:MONTH'?S? )?REVENUE\s*\n\s*\$?([\d.,]+[kKmM]?)", text)
    lm_prof_m   = re.search(r"LAST (?:MONTH'?S? )?PROFIT\s*\n\s*\$?([\d.,]+[kKmM]?)", text)

    ask         = _usd("$" + ask_m.group(1))    if ask_m    else None
    ttm_rev     = _usd("$" + ttm_rev_m.group(1)) if ttm_rev_m else None
    ttm_profit  = _usd("$" + ttm_prof_m.group(1)) if ttm_prof_m else None
    lm_rev      = _usd("$" + lm_rev_m.group(1))  if lm_rev_m  else None
    lm_profit   = _usd("$" + lm_prof_m.group(1)) if lm_prof_m else None

    # Description: find the paragraph section
    description = ""
    # Look for text after highlights/key metrics section
    desc_m = re.search(
        r"(?:This|The|An? |[A-Z][a-z])[^\n]{50,}(?:\n[^\n]{20,}){0,8}",
        text[500:3000]
    )
    if desc_m:
        description = desc_m.group(0)[:600]

    # Key Highlights: bullet-pointed lines starting with ✅ or •
    highlights = re.findall(r"[✅•\-]\s*(.{20,100})", text)[:6]

    # Asking price reasoning
    pr_m = re.search(r"ASKING PRICE REASONING\s*\n([^\n]{20,}(?:\n[^\n]{10,}){0,3})", text)
    pricing_text = pr_m.group(1)[:250] if pr_m else ""

    # Annual growth
    growth_m = re.search(r"ANNUAL GROWTH RATE\s*\n\s*([\d]+)%", text)
    growth = int(growth_m.group(1)) if growth_m else None

    if not title:
        return None

    listing_id = url.split("/")[-1] if "/" in url else f"acq-{abs(hash(title)) % 10_000_000}"

    deal = _make_deal(
        title=title,
        url=url,
        ttm_rev=ttm_rev,
        ttm_profit=ttm_profit,
        ask=ask,
        last_month_rev=lm_rev,
        last_month_profit=lm_profit,
        sector=sector,
        description=description,
        listing_id=listing_id,
        country=country,
        highlights=highlights,
        asking_price_reasoning=pricing_text,
    )

    if growth:
        deal["annual_growth_pct"] = growth

    return deal


# ─────────────────────────────────────────────────────────────────────────────
# Playwright scraper
# ─────────────────────────────────────────────────────────────────────────────

# Browse filter tabs to click for variety
BROWSE_FILTERS = [
    "https://app.acquire.com/browse",                        # personalized
    "https://app.acquire.com/browse?sort=newest",            # newest
    "https://app.acquire.com/browse?tag=profitable-saas",    # profitable SaaS
    "https://app.acquire.com/browse?sort=trending",          # trending
    "https://app.acquire.com/browse?tag=established",        # established
]


async def _async_fetch_acquire(email: str, password: str) -> list[dict]:
    """Login, collect listing URLs from multiple browse views, scrape each detail page."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(user_agent=(
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
        ))
        page = await context.new_page()

        # ── Login ────────────────────────────────────────────────────────────
        logger.info("Acquire: logging in as %s", email)
        await page.goto("https://app.acquire.com/signin", wait_until="domcontentloaded")

        email_el, pwd_el = None, None
        for attempt in range(5):
            await asyncio.sleep(2)
            for sel in ["input.special-input[type='text']", "input[type='text']",
                        "input[placeholder*='richard']"]:
                email_el = await page.query_selector(sel)
                if email_el: break
            pwd_el = await page.query_selector("input[type='password']")
            if email_el and pwd_el:
                break
            logger.debug("Acquire login inputs not ready, attempt %d", attempt + 1)

        if not email_el or not pwd_el:
            logger.error("Acquire: login form not found")
            await browser.close()
            return []

        await email_el.click()
        await email_el.fill(email)
        await asyncio.sleep(0.3)
        await pwd_el.click()
        await pwd_el.fill(password)
        await asyncio.sleep(0.3)
        await page.keyboard.press("Enter")
        await asyncio.sleep(7)

        if "signin" in page.url:
            logger.error("Acquire: login failed")
            await browser.close()
            return []
        logger.info("Acquire: logged in → %s", page.url)

        # ── Collect listing URLs from multiple browse views ───────────────────
        all_listing_urls: set[str] = set()

        for browse_url in BROWSE_FILTERS:
            try:
                await page.goto(browse_url, wait_until="domcontentloaded")
                await asyncio.sleep(3)
                # Scroll to load lazy-loaded cards
                for _ in range(6):
                    await page.keyboard.press("End")
                    await asyncio.sleep(1.2)

                links = await page.eval_on_selector_all(
                    "a[href*='/startup/']",
                    "els => [...new Set(els.map(e => e.href.split('?')[0]))]"
                )
                new = set(links) - all_listing_urls
                all_listing_urls.update(links)
                logger.info("Browse %s: +%d new URLs (total %d)", browse_url, len(new), len(all_listing_urls))
            except Exception as exc:
                logger.warning("Browse %s failed: %s", browse_url, exc)

        logger.info("Acquire: collected %d unique listing URLs", len(all_listing_urls))

        # ── Scrape each individual listing page ───────────────────────────────
        deals: list[dict] = []

        for listing_url in list(all_listing_urls)[:60]:  # cap at 60
            try:
                await page.goto(listing_url, wait_until="domcontentloaded")
                await asyncio.sleep(2)
                text = await page.inner_text("body")
                deal = _parse_listing_page(text, listing_url)
                if deal:
                    deals.append(deal)
                    logger.debug("Parsed listing %s: %s", listing_url.split("/")[-1], deal["title"][:50])
            except Exception as exc:
                logger.debug("Failed to scrape %s: %s", listing_url, exc)

        logger.info("Acquire: parsed %d deals from %d listing pages", len(deals), len(all_listing_urls))
        await browser.close()
        return deals


def fetch_acquire_authenticated(max_deals: int = 60) -> list[dict]:
    """Synchronous wrapper for Acquire.com authenticated scraping."""
    email    = os.environ.get("ACQUIRE_EMAIL", "")
    password = os.environ.get("ACQUIRE_PASSWORD", "")

    if not email or not password:
        logger.warning("Acquire auth: no credentials")
        return []

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, _async_fetch_acquire(email, password))
                return future.result(timeout=180)[:max_deals]
        else:
            return loop.run_until_complete(_async_fetch_acquire(email, password))[:max_deals]
    except Exception as exc:
        logger.error("Acquire fetch failed: %s", exc)
        return []


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────

def score_acquire_deal(deal: dict) -> float:
    score = 0.0
    desc  = (deal.get("description","") + " " + deal.get("title","")).lower()
    ask   = deal.get("asking_price_usd") or 0
    profit = deal.get("monthly_profit_usd") or 0
    mult  = deal.get("multiple") or 0

    if 10_000 < ask <= 150_000:   score += 30
    elif 150_000 < ask <= 400_000: score += 20
    elif ask > 0:                  score += 8

    if profit > 0:                 score += 20
    if deal.get("arr_usd"):        score += 8

    if 0 < mult <= 18:    score += 20
    elif 18 < mult <= 30: score += 12
    elif 30 < mult <= 40: score += 5

    if deal.get("country") == "India" or re.search(r"\bindia\b|₹|lakh|crore", desc):
        score += 20

    urgency_kw = ["burnout","quick close","no earnout","fast close","moving on",
                  "solo founder","maintenance mode","full-time","autopilot",
                  "feature complete","new chapter","life change"]
    score += min(sum(2 for kw in urgency_kw if kw in desc), 12)

    if re.search(r"\bsaas\b|\bsubscription\b|\brecurring\b|\bmrr\b|\barr\b", desc):
        score += 6

    growth = deal.get("annual_growth_pct") or 0
    if growth > 50:  score += 5
    if growth > 100: score += 5

    score += 5  # Acquire quality bonus
    return min(score, 100.0)
