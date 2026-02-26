"""
Curated demo deals — India-focused micro-PE targets.
Rich founder context baked in so agents can reason deeply even in demo mode.
"""

from datetime import datetime, timedelta
import random

# ─────────────────────────────────────────────────────────────────────────────
# Mock Deals — 5 hand-crafted "Gold Standard" examples
# Mix of India-based and global; varying motivations for training the AI's palate
# ─────────────────────────────────────────────────────────────────────────────

MOCK_DEALS = [
    {
        "id": "demo-001",
        "source": "Acquire",
        "title": "BillZap — GST Invoice SaaS for Indian Freelancers & SMBs",
        "url": "https://acquire.com/listings/demo-001",
        "asking_price_usd": 87000,
        "asking_price_inr": 7250000,
        "mrr_usd": 3200,
        "mrr_inr": 266000,
        "arr_usd": 38400,
        "arr_inr": 3192000,
        "multiple": "2.3x ARR",
        "listing_age_days": 9,
        "seller_handle": "@aakash_builds",
        "seller_name": "Aakash Mehta",
        "location": "Pune, Maharashtra, India",
        "background": "IIT Roorkee grad, 2019 batch. Built BillZap as a weekend project during COVID; it grew via word-of-mouth in CA WhatsApp groups.",
        "listing_text": (
            "Hey, I built BillZap 3 years ago to solve my own invoicing headaches as a freelancer. "
            "It's grown to 340 paying SMB customers, all organic, zero ads. ₹2.66L MRR, "
            "churn is basically zero — most customers have been with me since day 1. "
            "I got a really exciting offer to join a Series A fintech as Head of Product and "
            "I genuinely cannot do both. The product is feature-complete, runs on autopilot mostly, "
            "but I'm the only dev and it would need someone who cares. "
            "I want a fair price but more than that I want it to go to someone who'll take care of the customers. "
            "Asking ₹72.5L (~$87k). Prefer quick close. No earnout. "
            "Will do 45-day handover. GST reg and all compliances up to date."
        ),
        "founder_signals": {
            "twitter_bio": "Building things @BillZapApp | Ex-Infosys | IIT Roorkee '19",
            "last_tweet": "Super excited about the next chapter 🚀 More soon. (2 days ago)",
            "linkedin_activity": "Updated job title to 'Head of Product @ [stealth fintech]' 3 weeks ago",
            "reddit_post": "posted in r/IndieHackers: 'Thinking of selling my SaaS — anyone been through this?'",
        },
        "tags": ["India", "SaaS", "GST", "B2B", "fintech-adjacent", "solo-founder", "IIT"],
        "india_flags": {
            "gst_registered": True,
            "business_type": "Sole Proprietorship",
            "payment_gateway": "Razorpay",
            "family_involved": False,
            "customer_geography": "70% metros, 30% tier-2",
        },
    },
    {
        "id": "demo-002",
        "source": "Reddit",
        "title": "StartupPulse — India Startup Weekly Newsletter (12k subscribers, ₹90k/mo sponsors)",
        "url": "https://reddit.com/r/Entrepreneur/demo-002",
        "asking_price_usd": 32000,
        "asking_price_inr": 2666000,
        "mrr_usd": 1090,
        "mrr_inr": 90000,
        "arr_usd": 13080,
        "arr_inr": 1080000,
        "multiple": "2.5x ARR",
        "listing_age_days": 21,
        "seller_handle": "u/priya_newsletter",
        "seller_name": "Priya Nair",
        "location": "Bengaluru, Karnataka, India",
        "background": "Ex-journalist, covered Indian startup ecosystem for 4 years. Started the newsletter during her notice period at a media house.",
        "listing_text": (
            "Selling my newsletter StartupPulse. Been running it 2 years. "
            "12,000 subscribers, 48% open rate (verified via Mailchimp). "
            "Revenue: ₹90,000/month from 3 recurring sponsors (all renewed at least once). "
            "I'm being honest — I'm exhausted and I've been offered a staff writer role "
            "at a prominent VC firm covering their portfolio. It's my dream job and it conflicts "
            "with running an independent newsletter. I'd rather sell than let it die. "
            "Asking ₹26.66L. The brand is stronger than me — I've kept it brand-first. "
            "Subscriber list, Mailchimp account, website, all transferable. "
            "Sponsors have been informed and are open to working with new owner."
        ),
        "founder_signals": {
            "twitter_bio": "Newsletter nerd | Covering Indian startups | StartupPulse.in",
            "last_tweet": "Grateful for the journey. Big news coming soon 🙏 (4 days ago)",
            "linkedin_activity": "Connected with 8 VCs in the last 30 days",
            "reddit_post": "This is the original post",
        },
        "tags": ["India", "newsletter", "B2B", "media", "startup-ecosystem", "Mailchimp"],
        "india_flags": {
            "gst_registered": False,
            "business_type": "Individual / Freelancer",
            "payment_gateway": "Manual bank transfer from sponsors",
            "family_involved": False,
            "customer_geography": "National + some diaspora",
        },
    },
    {
        "id": "demo-003",
        "source": "X",
        "title": "WhatsBot Pro — WhatsApp Automation SaaS for Indian D2C & SMBs",
        "url": "https://x.com/demo-003",
        "asking_price_usd": 96000,
        "asking_price_inr": 7990000,
        "mrr_usd": 4150,
        "mrr_inr": 345000,
        "arr_usd": 49800,
        "arr_inr": 4140000,
        "multiple": "1.93x ARR",
        "listing_age_days": 5,
        "seller_handle": "@rk_founder",
        "seller_name": "Rohit Kumar",
        "location": "Mumbai, Maharashtra, India",
        "background": "Self-taught dev, built WhatsBot Pro after noticing D2C brands struggling with WhatsApp support at scale. Business boomed post-Jio.",
        "listing_text": (
            "Built WhatsBot Pro 2 years ago. ₹3.45L MRR, 180 customers, all D2C brands and SMBs. "
            "Product: WhatsApp Business API automation — order tracking, abandoned cart, "
            "support ticket routing. Customers love it, churn is ~4%. "
            "I'm selling because I want to go all-in on an enterprise play (full API platform) "
            "and WhatsBot Pro is B2SMB which is a completely different sales motion. "
            "Asking ₹79.9L. This needs a buyer who understands Indian D2C space. "
            "Warning: WhatsApp API access is in my entity's name — new buyer needs to apply "
            "fresh or transfer (I'll help with this). Razorpay integration also needs new PG setup."
        ),
        "founder_signals": {
            "twitter_bio": "Building WhatsBot Pro | DMs open | Mumbai 🇮🇳",
            "last_tweet": "Working on something new in the B2B API space. Keep watching this space (1 week ago)",
            "linkedin_activity": "Posted 'The future of enterprise automation' article 2 weeks ago",
            "reddit_post": None,
        },
        "tags": ["India", "WhatsApp", "SaaS", "D2C", "automation", "B2B", "Mumbai"],
        "india_flags": {
            "gst_registered": True,
            "business_type": "Private Limited",
            "payment_gateway": "Razorpay",
            "family_involved": False,
            "customer_geography": "All India, metro-heavy",
            "special_risk": "WhatsApp API access not directly transferable — requires new meta approval",
        },
    },
    {
        "id": "demo-004",
        "source": "Acquire",
        "title": "HRFlow — Employee Onboarding & HRMS Lite for Indian Startups (< 200 employees)",
        "url": "https://acquire.com/listings/demo-004",
        "asking_price_usd": 52000,
        "asking_price_inr": 4330000,
        "mrr_usd": 1800,
        "mrr_inr": 150000,
        "arr_usd": 21600,
        "arr_inr": 1800000,
        "multiple": "2.4x ARR",
        "listing_age_days": 34,
        "seller_handle": "@sameer_hrflow",
        "seller_name": "Sameer Joshi",
        "location": "Hyderabad, Telangana, India",
        "background": "Ex-Deloitte HR consultant. Built HRFlow with a co-founder who left 8 months ago. Now running solo, bored of the day-to-day.",
        "listing_text": (
            "Selling HRFlow — been running it for 3 years now. ₹1.5L MRR, 65 startup customers. "
            "Product is honestly feature-complete; nothing new to build. "
            "My co-founder left to join a startup in Singapore last year and since then "
            "I've been doing everything myself. I'm still profitable but I've mentally moved on. "
            "I have a new idea I'm excited about in the edtech space. "
            "HRFlow basically runs itself — most support is tier-1 stuff. "
            "Asking ₹43.3L (2.4x ARR). Prefer an operator who'll keep it running, "
            "not flip it. Will do full 60-day handover. GST compliant. PVT LTD entity."
        ),
        "founder_signals": {
            "twitter_bio": "Building things in HR tech | Hyderabad | Open to acquisition convos",
            "last_tweet": "The unsexy truth about running a B2B SaaS: 80% is customer success (3 weeks ago)",
            "linkedin_activity": "Viewed 14 EdTech startup profiles in last 30 days",
            "reddit_post": "Posted in r/SaaS: 'How do you know when it's time to sell?'",
        },
        "tags": ["India", "SaaS", "HR", "B2B", "Hyderabad", "solo-founder", "boredom-exit"],
        "india_flags": {
            "gst_registered": True,
            "business_type": "Private Limited",
            "payment_gateway": "Razorpay",
            "family_involved": False,
            "customer_geography": "Mostly Hyderabad, Pune, Bengaluru startups",
        },
    },
    {
        "id": "demo-005",
        "source": "Reddit",
        "title": "Notion India Templates — Premium Notion templates for Indian CAs, Founders, and Students (₹45k/mo)",
        "url": "https://reddit.com/r/SaaS/demo-005",
        "asking_price_usd": 13000,
        "asking_price_inr": 1082000,
        "mrr_usd": 540,
        "mrr_inr": 45000,
        "arr_usd": 6480,
        "arr_inr": 540000,
        "multiple": "2x ARR",
        "listing_age_days": 14,
        "seller_handle": "u/notion_india_girl",
        "seller_name": "Ananya Sood",
        "location": "Delhi, India",
        "background": "CA Final student who built a Notion template store targeting Indian CAs, startup founders, and students. Pure side project that took on a life of its own.",
        "listing_text": (
            "So I built this Notion template store 18 months ago as a side project "
            "while studying for my CA Finals. It's generating ₹45k/month completely passively. "
            "I have 2,200 customers, 18 templates (CA compliance, startup runbooks, student trackers). "
            "I'm now done with my CA exams and joining a Big 4 firm next month. "
            "This store is a distraction. I'd rather sell it than let it sit. "
            "Asking ₹10.82L. Gumroad store + website + customer email list. "
            "Zero overhead. Basically pure profit. Will handover in a weekend call."
        ),
        "founder_signals": {
            "twitter_bio": "CA Final | Notion enthusiast | Side project 🛠️",
            "last_tweet": "Officially done with CA exams. Now what? (10 days ago)",
            "linkedin_activity": "Added 'Chartered Accountant (pending)' to profile",
            "reddit_post": "This is the original post",
        },
        "tags": ["India", "Notion", "templates", "Gumroad", "CA", "student", "passive-income", "side-project"],
        "india_flags": {
            "gst_registered": False,
            "business_type": "Individual / Gumroad account",
            "payment_gateway": "Gumroad (global)",
            "family_involved": False,
            "customer_geography": "All India",
        },
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Helper: Format deals for agent consumption
# ─────────────────────────────────────────────────────────────────────────────

def format_deals_for_agent(deals: list[dict]) -> str:
    """
    Convert deal dicts into a rich text block agents can reason over.
    This is the "context" that replaces complex retrieval pipelines.
    """
    lines = []
    for i, deal in enumerate(deals, 1):
        inr = deal.get("asking_price_inr", 0)
        usd = deal.get("asking_price_usd", 0)
        mrr_inr = deal.get("mrr_inr", 0)
        age = deal.get("listing_age_days", "?")
        handle = deal.get("seller_handle", "N/A")
        location = deal.get("location", "Unknown")
        tags = ", ".join(deal.get("tags", []))

        india_flags = deal.get("india_flags", {})
        gst = "✅ GST Reg" if india_flags.get("gst_registered") else "⚠️ No GST"
        biz_type = india_flags.get("business_type", "Unknown")
        pg = india_flags.get("payment_gateway", "Unknown")
        family = "⚠️ Family involved" if india_flags.get("family_involved") else "✅ No family involvement"
        special = india_flags.get("special_risk", None)

        founder_signals = deal.get("founder_signals", {})
        signals_text = "\n".join([
            f"  • Twitter: {founder_signals.get('last_tweet', 'N/A')}",
            f"  • LinkedIn: {founder_signals.get('linkedin_activity', 'N/A')}",
            f"  • Reddit: {founder_signals.get('reddit_post', 'N/A')}",
        ])

        lines.append(f"""
{'='*60}
DEAL #{i}: {deal['title']}
ID: {deal['id']} | Source: {deal['source']} | Age: {age} days | Location: {location}
URL: {deal['url']}
Seller: {deal.get('seller_name', 'Unknown')} ({handle})
Background: {deal.get('background', 'N/A')}

FINANCIALS:
  Ask: ₹{inr:,.0f} (~${usd:,.0f}) | MRR: ₹{mrr_inr:,.0f}/mo | Multiple: {deal.get('multiple', 'N/A')}

INDIA COMPLIANCE FLAGS:
  {gst} | Entity: {biz_type} | PG: {pg} | {family}
  {f'⚠️ Special Risk: {special}' if special else ''}

FOUNDER SIGNALS:
{signals_text}
  • Bio: {founder_signals.get('twitter_bio', 'N/A')}

LISTING TEXT (Verbatim):
{deal.get('listing_text', '')}

TAGS: {tags}
{'='*60}
""")
    return "\n".join(lines)


def get_mock_deals_formatted() -> str:
    """Returns all mock deals as a formatted string for agent consumption."""
    return format_deals_for_agent(MOCK_DEALS)


def get_mock_deals_summary() -> list[dict]:
    """Returns lightweight summary dicts for UI rendering."""
    summaries = []
    for deal in MOCK_DEALS:
        summaries.append({
            "id": deal["id"],
            "title": deal["title"],
            "source": deal["source"],
            "location": deal.get("location", "Unknown"),
            "asking_price_inr": deal.get("asking_price_inr", 0),
            "asking_price_usd": deal.get("asking_price_usd", 0),
            "mrr_inr": deal.get("mrr_inr", 0),
            "multiple": deal.get("multiple", "N/A"),
            "listing_age_days": deal.get("listing_age_days", 0),
            "seller_name": deal.get("seller_name", ""),
            "seller_handle": deal.get("seller_handle", ""),
            "tags": deal.get("tags", []),
            "url": deal.get("url", ""),
        })
    return summaries
