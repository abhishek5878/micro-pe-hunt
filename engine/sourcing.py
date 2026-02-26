"""
Sourcing Engine — Acquire.com RSS, Reddit (PRAW OAuth), and Nitter/X scraping.
All sources fall back gracefully to curated mock data when live APIs are unavailable.
"""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional
import random

import feedparser
import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

logger = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    )
}

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class RawListing:
    id: str
    source: str          # "Acquire", "Reddit", "X"
    title: str
    body: str
    url: str
    asking_price: Optional[float] = None
    mrr: Optional[float] = None
    arr: Optional[float] = None
    age_days: Optional[int] = None
    seller_handle: Optional[str] = None
    fetched_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tags: list = field(default_factory=list)

# ---------------------------------------------------------------------------
# Mock / seed data — instantly demo-ready without live API keys
# ---------------------------------------------------------------------------

MOCK_LISTINGS: list[RawListing] = [
    RawListing(
        id="mock-001",
        source="Acquire",
        title="Profitable SaaS Invoice Automation Tool — Burnt Out, Ready to Exit",
        body=(
            "Built this 3 years ago as a side project. It automates invoice reconciliation "
            "for freelancers. $4,200 MRR, ~87% margins, zero churn last 6 months. "
            "I'm the only developer and honestly I'm just tired. Running this while holding "
            "a full-time job has drained me. Asking 2.5x ARR. I want a fast, clean close — "
            "no earnouts, no drama. I am the brand/support person but I'll do a 30-day handover."
        ),
        url="https://acquire.com/listings/mock-001",
        asking_price=126000,
        mrr=4200,
        arr=50400,
        age_days=12,
        seller_handle="@devfounder_x",
        tags=["SaaS", "invoicing", "solo-founder", "burnout"],
    ),
    RawListing(
        id="mock-002",
        source="Reddit",
        title="[For Sale] Newsletter — 18k subscribers, $2.1k MRR, moving on to a new chapter",
        body=(
            "r/Entrepreneur — Hey all. I've been running a B2B fintech newsletter for 2 years. "
            "18,000 confirmed subs, 42% open rate, $2,100 MRR from sponsorships. "
            "I got a job offer I can't refuse and don't have time for this anymore. "
            "I'd rather sell to someone who'll grow it than let it die. "
            "Asking $42k — basically 20x monthly. DM me if serious."
        ),
        url="https://reddit.com/r/Entrepreneur/mock-002",
        asking_price=42000,
        mrr=2100,
        arr=25200,
        age_days=5,
        seller_handle="u/fintech_newsletter_guy",
        tags=["newsletter", "fintech", "B2B", "job-offer-exit"],
    ),
    RawListing(
        id="mock-003",
        source="X",
        title="Selling my micro-SaaS — $800 MRR Notion template marketplace, going all-in on consulting",
        body=(
            "Selling my Notion template store. $800/mo passive, 2,200 customers, "
            "zero support overhead. I'm pivoting to high-ticket consulting and just "
            "don't want the mental overhead of a separate product. "
            "Asking 24x = $19,200. DM if interested. Quick close preferred. "
            "No earnout. Simple asset purchase."
        ),
        url="https://x.com/mock-003",
        asking_price=19200,
        mrr=800,
        arr=9600,
        age_days=3,
        seller_handle="@notion_stack_guy",
        tags=["Notion", "templates", "marketplace", "pivot"],
    ),
    RawListing(
        id="mock-004",
        source="Acquire",
        title="B2B Chrome Extension — $3.8k MRR, LinkedIn automation, founder exiting to VC-backed startup",
        body=(
            "LinkedIn outreach automation Chrome extension. $3,800 MRR, 480 paying customers, "
            "churn < 3%. I co-founded this but got a spot at a VC-backed AI startup and "
            "need to fully commit. My co-founder left 6 months ago. I am basically running "
            "this alone. Solid product, runs itself 90% of the time. Looking for 2x ARR = $91k. "
            "Will help with transition. Fast close only."
        ),
        url="https://acquire.com/listings/mock-004",
        asking_price=91200,
        mrr=3800,
        arr=45600,
        age_days=8,
        seller_handle="@chromeext_pete",
        tags=["Chrome extension", "LinkedIn", "B2B", "VC-exit"],
    ),
    RawListing(
        id="mock-005",
        source="Reddit",
        title="Selling bootstrapped SaaS — small HR onboarding tool, $1.5k MRR — bored and moving on",
        body=(
            "r/SaaS — This project made me proud 2 years ago, now it just feels like a "
            "maintenance task. $1,500 MRR, 60 SMB customers, NPS is decent, "
            "product is feature-complete. I have a new idea I'm obsessed with and this "
            "is just mentally blocking me. Asking $36k (24x MRR). "
            "Prefer asset purchase, no earnout. Happy to do a 2-week handover call."
        ),
        url="https://reddit.com/r/SaaS/mock-005",
        asking_price=36000,
        mrr=1500,
        arr=18000,
        age_days=18,
        seller_handle="u/hr_saas_bob",
        tags=["SaaS", "HR", "SMB", "boredom-exit"],
    ),
]

# ---------------------------------------------------------------------------
# Acquire.com RSS scraper
# ---------------------------------------------------------------------------

ACQUIRE_RSS_URL = "https://acquire.com/rss/listings"

def _parse_price(text: str) -> Optional[float]:
    """Extract the first dollar amount from a string."""
    text = text.replace(",", "")
    match = re.search(r"\$([0-9]+(?:\.[0-9]+)?)[kKmM]?", text)
    if not match:
        return None
    val = float(match.group(1))
    suffix = text[match.end():match.end() + 1].lower()
    if suffix == "k":
        val *= 1_000
    elif suffix == "m":
        val *= 1_000_000
    return val

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=4))
def fetch_acquire_rss(max_items: int = 20) -> list[RawListing]:
    """Fetch listings from Acquire.com RSS feed."""
    try:
        feed = feedparser.parse(ACQUIRE_RSS_URL)
        if feed.bozo and not feed.entries:
            raise ValueError("Feed parse error")

        results = []
        for i, entry in enumerate(feed.entries[:max_items]):
            title = entry.get("title", "Untitled Listing")
            summary = entry.get("summary", entry.get("description", ""))
            soup = BeautifulSoup(summary, "html.parser")
            body = soup.get_text(separator=" ").strip()
            url = entry.get("link", "")
            published = entry.get("published_parsed")
            age = 0
            if published:
                pub_dt = datetime(*published[:6])
                age = (datetime.utcnow() - pub_dt).days

            price = _parse_price(body) or _parse_price(title)
            mrr_match = re.search(r"\$([0-9,]+)\s*(?:MRR|mrr|monthly)", body)
            mrr = float(mrr_match.group(1).replace(",", "")) if mrr_match else None

            results.append(RawListing(
                id=f"acquire-{i}-{int(time.time())}",
                source="Acquire",
                title=title,
                body=body,
                url=url,
                asking_price=price,
                mrr=mrr,
                arr=mrr * 12 if mrr else None,
                age_days=age,
                tags=["Acquire"],
            ))
        logger.info("Acquire RSS: fetched %d listings", len(results))
        return results

    except Exception as exc:
        logger.warning("Acquire RSS fetch failed: %s — using mocks", exc)
        return [m for m in MOCK_LISTINGS if m.source == "Acquire"]

# ---------------------------------------------------------------------------
# Reddit scraper — PRAW (OAuth) with JSON API fallback
# ---------------------------------------------------------------------------

REDDIT_SUBREDDITS = ["SaaS", "Entrepreneur", "startups", "IndieHackers", "smallbusiness"]
REDDIT_QUERIES    = [
    "selling my SaaS exit MRR",
    "selling my business exit acquisition",
    "for sale bootstrapped profitable",
    "want to sell newsletter",
]
REDDIT_EXIT_KEYWORDS = [
    "selling", "for sale", "exit", "MRR", "acqui", "burnout",
    "moving on", "pivot", "quick close", "asset purchase", "acquisition",
]


def _get_praw_reddit():
    """Return an authenticated praw.Reddit instance using env/secrets credentials."""
    client_id     = os.environ.get("REDDIT_CLIENT_ID", "")
    client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
    user_agent    = os.environ.get("REDDIT_USER_AGENT", "python:pocketfund:v1.0 (by u/pocketfund)")
    if not client_id or not client_secret:
        return None
    try:
        import praw
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent,
        )
        # Verify read-only access
        _ = reddit.subreddit("SaaS").hot(limit=1)
        return reddit
    except Exception as exc:
        logger.warning("PRAW init failed: %s", exc)
        return None


def _praw_to_listing(submission, subreddit_name: str) -> Optional[RawListing]:
    """Convert a PRAW Submission to a RawListing."""
    title    = submission.title or ""
    selftext = submission.selftext or ""
    combined = f"{title} {selftext}".lower()

    if not any(kw in combined for kw in REDDIT_EXIT_KEYWORDS):
        return None
    if len(selftext.strip()) < 50:
        return None  # skip title-only posts

    price = _parse_price(selftext) or _parse_price(title)
    mrr_match = re.search(r"\$([0-9,]+)\s*(?:MRR|mrr|monthly)", selftext, re.IGNORECASE)
    mrr   = float(mrr_match.group(1).replace(",", "")) if mrr_match else None
    age   = max(0, int((time.time() - submission.created_utc) / 86400))

    return RawListing(
        id=f"reddit-{submission.id}",
        source="Reddit",
        title=title[:200],
        body=selftext[:2500],
        url=f"https://reddit.com{submission.permalink}",
        asking_price=price,
        mrr=mrr,
        arr=mrr * 12 if mrr else None,
        age_days=age,
        seller_handle=f"u/{submission.author.name if submission.author else 'deleted'}",
        tags=["Reddit", subreddit_name],
    )


def _fetch_via_praw(reddit, max_per_query: int = 8) -> list[RawListing]:
    """Use authenticated PRAW to search multiple subreddits."""
    results: list[RawListing] = []
    seen: set[str] = set()

    for query in REDDIT_QUERIES[:2]:  # limit to 2 queries to avoid rate limits
        for sub in REDDIT_SUBREDDITS[:3]:
            try:
                subreddit = reddit.subreddit(sub)
                for submission in subreddit.search(query, sort="new", limit=max_per_query, time_filter="month"):
                    if submission.id in seen:
                        continue
                    seen.add(submission.id)
                    listing = _praw_to_listing(submission, sub)
                    if listing:
                        results.append(listing)
            except Exception as exc:
                logger.debug("PRAW search %s/%s failed: %s", sub, query, exc)

    return results


def _fetch_via_json_api(max_per_sub: int = 10) -> list[RawListing]:
    """Unauthenticated Reddit JSON API fallback."""
    urls = [
        "https://www.reddit.com/r/SaaS/search.json?q=selling+exit+MRR&sort=new&restrict_sr=1&limit=15",
        "https://www.reddit.com/r/Entrepreneur/search.json?q=selling+my+business&sort=new&restrict_sr=1&limit=15",
    ]
    results: list[RawListing] = []
    for url in urls:
        try:
            resp = requests.get(url, headers=HEADERS, timeout=10)
            resp.raise_for_status()
            posts = resp.json().get("data", {}).get("children", [])
            for post in posts[:max_per_sub]:
                d = post["data"]
                title    = d.get("title", "")
                selftext = d.get("selftext", "")
                combined = f"{title} {selftext}".lower()
                if not any(kw in combined for kw in REDDIT_EXIT_KEYWORDS):
                    continue
                price = _parse_price(selftext) or _parse_price(title)
                mrr_match = re.search(r"\$([0-9,]+)\s*(?:MRR|mrr|monthly)", selftext, re.IGNORECASE)
                mrr = float(mrr_match.group(1).replace(",", "")) if mrr_match else None
                age = int((time.time() - d.get("created_utc", time.time())) / 86400)
                results.append(RawListing(
                    id=f"reddit-{d.get('id', str(int(time.time())))}",
                    source="Reddit",
                    title=title[:200],
                    body=selftext[:2500],
                    url=f"https://reddit.com{d.get('permalink', '')}",
                    asking_price=price,
                    mrr=mrr,
                    arr=mrr * 12 if mrr else None,
                    age_days=age,
                    seller_handle=f"u/{d.get('author', 'unknown')}",
                    tags=["Reddit", d.get("subreddit", "")],
                ))
        except Exception as exc:
            logger.debug("JSON API %s failed: %s", url, exc)
    return results


def fetch_reddit_listings(max_items: int = 20) -> list[RawListing]:
    """
    Fetch Reddit listings.
    Priority: PRAW (OAuth, richer results) → JSON API → mock data.
    """
    # 1. Try PRAW with credentials from env/secrets
    reddit = _get_praw_reddit()
    if reddit:
        try:
            results = _fetch_via_praw(reddit, max_per_query=8)
            if results:
                logger.info("Reddit PRAW: fetched %d listings", len(results))
                return results[:max_items]
        except Exception as exc:
            logger.warning("PRAW fetch failed: %s — trying JSON API", exc)

    # 2. JSON API fallback (no creds needed)
    try:
        results = _fetch_via_json_api()
        if results:
            logger.info("Reddit JSON API: fetched %d listings", len(results))
            return results[:max_items]
    except Exception as exc:
        logger.warning("Reddit JSON API failed: %s — using mocks", exc)

    # 3. Mock fallback
    logger.warning("Reddit: all methods failed — using mock data")
    return [m for m in MOCK_LISTINGS if m.source == "Reddit"]

# ---------------------------------------------------------------------------
# X / Twitter simulated scraper (Nitter fallback + mock)
# ---------------------------------------------------------------------------

NITTER_INSTANCES = [
    "https://nitter.privacydev.net",
    "https://nitter.poast.org",
    "https://nitter.1d4.us",
]

X_SEARCH_QUERY = "selling my SaaS OR \"selling my newsletter\" OR \"for sale\" MRR exit"

def _scrape_nitter(base_url: str, query: str, max_tweets: int = 10) -> list[RawListing]:
    """Scrape a Nitter instance for exit-signal tweets."""
    results = []
    encoded = requests.utils.quote(query)
    url = f"{base_url}/search?f=tweets&q={encoded}"
    resp = requests.get(url, headers=HEADERS, timeout=8)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "lxml")
    tweet_cards = soup.select(".timeline-item")[:max_tweets]

    for i, card in enumerate(tweet_cards):
        content_el = card.select_one(".tweet-content")
        handle_el = card.select_one(".username")
        link_el = card.select_one("a.tweet-link")

        if not content_el:
            continue

        text = content_el.get_text(separator=" ").strip()
        handle = handle_el.get_text().strip() if handle_el else "@unknown"
        tweet_url = (base_url + link_el["href"]) if link_el else base_url

        price = _parse_price(text)
        mrr_match = re.search(r"\$([0-9,]+)\s*(?:MRR|mrr|monthly)", text, re.IGNORECASE)
        mrr = float(mrr_match.group(1).replace(",", "")) if mrr_match else None

        results.append(RawListing(
            id=f"x-{i}-{int(time.time())}",
            source="X",
            title=text[:100] + ("…" if len(text) > 100 else ""),
            body=text,
            url=tweet_url,
            asking_price=price,
            mrr=mrr,
            arr=mrr * 12 if mrr else None,
            age_days=random.randint(0, 7),
            seller_handle=handle,
            tags=["X", "Twitter"],
        ))
    return results

def fetch_x_listings(max_items: int = 10) -> list[RawListing]:
    """Attempt Nitter instances in sequence; fall back to mock data."""
    for instance in NITTER_INSTANCES:
        try:
            results = _scrape_nitter(instance, X_SEARCH_QUERY, max_items)
            if results:
                logger.info("X/Nitter: fetched %d tweets from %s", len(results), instance)
                return results
        except Exception as exc:
            logger.debug("Nitter %s failed: %s", instance, exc)
            continue

    logger.warning("All Nitter instances failed — using X mock data")
    return [m for m in MOCK_LISTINGS if m.source == "X"]

# ---------------------------------------------------------------------------
# Async orchestrator
# ---------------------------------------------------------------------------

async def _run_in_executor(fn, *args):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, fn, *args)

async def fetch_all_sources(
    use_acquire: bool = True,
    use_reddit: bool = True,
    use_x: bool = True,
    force_mock: bool = False,
) -> list[RawListing]:
    """
    Concurrently fetch from all enabled sources.
    Returns deduplicated, combined listing list.
    """
    if force_mock:
        return MOCK_LISTINGS.copy()

    tasks = []
    if use_acquire:
        tasks.append(_run_in_executor(fetch_acquire_rss))
    if use_reddit:
        tasks.append(_run_in_executor(fetch_reddit_listings))
    if use_x:
        tasks.append(_run_in_executor(fetch_x_listings))

    results_nested = await asyncio.gather(*tasks, return_exceptions=True)

    combined: list[RawListing] = []
    seen_ids: set[str] = set()
    for batch in results_nested:
        if isinstance(batch, Exception):
            logger.error("Source fetch exception: %s", batch)
            continue
        for listing in batch:
            if listing.id not in seen_ids:
                seen_ids.add(listing.id)
                combined.append(listing)

    if not combined:
        logger.warning("All sources returned empty — falling back to full mock set")
        return MOCK_LISTINGS.copy()

    return combined


def fetch_all_sources_sync(
    use_acquire: bool = True,
    use_reddit: bool = True,
    use_x: bool = True,
    force_mock: bool = False,
) -> list[RawListing]:
    """Synchronous wrapper for Streamlit (runs the async loop internally)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(
                    asyncio.run,
                    fetch_all_sources(use_acquire, use_reddit, use_x, force_mock),
                )
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(
                fetch_all_sources(use_acquire, use_reddit, use_x, force_mock)
            )
    except Exception as exc:
        logger.error("fetch_all_sources_sync failed: %s", exc)
        return MOCK_LISTINGS.copy()
