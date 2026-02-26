"""
Strategy Agent — Outreach Generator & LOI Drafter.
Produces personalized outreach messages (WhatsApp/DM/Email) and LOI outlines.
Adapts tone and structure to founder psychology and India cultural context.
Works standalone — LLM adds personalization depth when available.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

KB_PATH = Path(__file__).parent.parent / "data" / "sme_knowledge.json"


def _load_kb() -> dict:
    with open(KB_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Outreach Templates (India-tuned, operator voice)
# ─────────────────────────────────────────────────────────────────────────────

OUTREACH_TEMPLATES = {
    "Burnout Exit": """\
Hey {seller_name},

Came across your listing — sounds like you've been running hard for a long time and are ready for a clean break. I respect that.

Quick background: I'm Dev Shah, I run Pocket Fund — we acquire and operate small profitable businesses in India. No brokers, no drawn-out process.

A few quick questions before we go further:
1. Is the revenue holding steady or has it changed in the last 3 months?
2. How tied in are you to day-to-day — could someone step in within 30 days?
3. Any flexibility on timeline if we move fast?

If the basics check out, I can get an LOI to you this week. Clean close, no earnout drama.

Talk soon,
Dev Shah
Pocket Fund
""",

    "Opportunity Exit": """\
Hey {seller_name},

Saw the listing — congrats on the new opportunity. Smart move to sell now while the business is in good shape.

I'm Dev Shah from Pocket Fund. We focus on acquiring profitable digital/operational businesses in India. Cash buyer, clean process, and I can move on your timeline.

Two things before we talk:
1. What's the revenue run-rate look like right now?
2. Any specific transition support you'd want in the deal?

If you're targeting a close before your new role starts, that's actually doable. Let's jump on a 15-min call and I'll tell you straight if we're a fit.

Dev Shah
Pocket Fund | {buyer_email}
""",

    "Family Exit": """\
Hi {seller_name},

I came across your business and it caught my attention. Looks like something that was built with real care over many years.

I'm Dev Shah — I run Pocket Fund, and we look for businesses like this: ones with a solid foundation, good customer relationships, and a founder (or family) who wants to ensure it goes to someone who will continue the work.

I understand family businesses often involve multiple stakeholders in a decision like this. I'm happy to have a conversation at whatever pace makes sense — no pressure.

Would a short call work this week? I'd love to understand the business and share how we think about acquisitions.

Regards,
Dev Shah
Pocket Fund
""",

    "Boredom / Drift": """\
Hey {seller_name},

Saw your listing — honestly sounds like the business is in good shape and running on autopilot, and you've just mentally moved on. I get it.

I'm Dev Shah, I run Pocket Fund. We buy and operate businesses like this — the "boring profitable" ones that don't need a VC, just a focused operator.

No rush from my side, but if you're open to a quick chat: I'd love to understand what it actually takes to run it week-to-week, and we can go from there.

Clean deal, simple structure, no earnout.

Dev
""",

    "Financial Distress": """\
Hi {seller_name},

I saw your listing and wanted to reach out directly. I don't know your full situation, but if timing matters, I can move fast — I'm a direct cash buyer with no financing dependency.

I want to be fair — not looking to lowball anyone. But I do need the numbers to make sense on both sides.

If you're open to a quick conversation, I'll tell you within 15 minutes if I'm a real buyer or not. No wasted time.

Dev Shah
Pocket Fund
""",

    "Low Signal": """\
Hi {seller_name},

Came across your listing for {business_name} — looks interesting. I focus on acquiring profitable small businesses in India, and this fits the profile.

I'm Dev Shah from Pocket Fund. Quick question: what's driving the decision to sell, and what does the ideal buyer look like from your end?

Happy to connect and explore whether there's a fit.

Dev
""",
}

# ─────────────────────────────────────────────────────────────────────────────
# LOI Template
# ─────────────────────────────────────────────────────────────────────────────

LOI_TEMPLATE = """\
LETTER OF INTENT — ASSET ACQUISITION
[NON-BINDING EXCEPT CLAUSES 4 & 7]

Date: {date}
Reference: PF-LOI-{ref}

FROM: {buyer_name}, {buyer_entity} ("Buyer")
TO: {seller_name} ("Seller")
RE: Acquisition of {business_name} ("the Business")

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. PROPOSED PURCHASE PRICE

   Total Consideration: ₹{purchase_price_inr} (approx. ${purchase_price_usd})
   
   Payment Structure:
   ✦ ₹{upfront_inr} ({upfront_pct}%) — at closing, wire transfer
   ✦ ₹{holdback_inr} ({holdback_pct}%) — held in escrow for {holdback_days} days post-close;
     released upon successful transition milestone completion

2. TRANSACTION STRUCTURE

   Recommended: {structure_type}
   
   Assets to be transferred:
   ✦ All intellectual property (code, brand, content, trademarks if registered)
   ✦ Customer database, subscription lists, and active contracts
   ✦ Domain names, social media accounts, and digital assets
   ✦ Operational documentation, SOPs, and vendor relationships
   ✦ Business-critical software licenses and tool subscriptions
   
   Excluded (unless separately negotiated):
   ✦ Pre-closing cash balances and receivables
   ✦ Personal liabilities of Seller
   ✦ Pre-closing tax obligations

3. INDIA-SPECIFIC CONDITIONS (Precedent to Closing)

   ✦ GST compliance certificate for last 12 months (CA-signed)
   ✦ Udyam registration transfer documentation (or buyer fresh registration)
   ✦ Clear title to all transferred assets; no encumbrances
   ✦ {entity_specific_condition}
   ✦ All family/co-owner sign-offs on sale (if applicable)
   ✦ Client consent letters for key accounts (if required by contract)
   ✦ NOC from bank (if working capital facility exists)

4. EXCLUSIVITY [BINDING]

   Upon both parties signing this LOI, Seller agrees not to solicit, entertain,
   or share information with any other buyer for {exclusivity_days} calendar days.

5. DUE DILIGENCE PERIOD

   {dd_days} days from LOI signing. Seller to provide:
   ✦ Last 24 months bank statements (all accounts)
   ✦ Last 3 years audited P&L and balance sheets (or CA-certified)
   ✦ GST returns (GSTR-1, GSTR-3B) for last 12 months
   ✦ Customer MRR/ARR breakdown and churn data
   ✦ Key employee list and their status (will they stay?)
   ✦ Any pending litigation, notices, or disputes

6. TRANSITION SUPPORT

   Seller to provide {transition_days} days of structured transition support:
   ✦ Documentation of all operational processes
   ✦ Warm introduction to top 10 customers/vendors
   ✦ Daily/weekly async support via WhatsApp/email
   ✦ Access to all tool admin panels for knowledge transfer

7. CONFIDENTIALITY [BINDING]

   Both parties to treat all shared information as strictly confidential.
   This clause survives termination of this LOI.

8. TIMELINE

   Target signing: Within {signing_days} business days of mutual agreement
   Target close: {close_date}
   
   Buyer is a cash buyer — no financing contingency.

9. NON-BINDING NATURE

   This LOI is non-binding except Clauses 4 and 7. It represents intent only.
   A formal Purchase Agreement will be executed by both parties post-DD.

10. EXPIRY

    This LOI expires if not countersigned by {expiry_date}.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ACCEPTED:

BUYER: _______________________        SELLER: _______________________
{buyer_name}                          {seller_name}
{buyer_entity}                        
Date: _______________                 Date: _______________

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[Prototype document — Pocket Fund. Not legal advice. Consult a qualified Indian M&A advocate.]
"""


# ─────────────────────────────────────────────────────────────────────────────
# Strategy Agent
# ─────────────────────────────────────────────────────────────────────────────

class StrategyAgent:
    """
    Generates acquisition strategy documents:
    - Personalized outreach (WhatsApp/DM/Email)
    - LOI outline
    - Deal structure note
    - Negotiation playbook
    """

    def __init__(self, llm=None, buyer_name: str = "Dev Shah", buyer_entity: str = "Pocket Fund", buyer_email: str = "dev@pocketfund.in"):
        self.llm = llm
        self.buyer_name = buyer_name
        self.buyer_entity = buyer_entity
        self.buyer_email = buyer_email
        self.kb = _load_kb()

    def generate_outreach(
        self,
        motivation_type: str,
        seller_name: str = "Founder",
        business_name: str = "the business",
        psych_data: dict = None,
        reg_data: dict = None,
        memory_context: str = "",
    ) -> str:
        """Generate a personalized outreach message."""

        # Template base
        template = OUTREACH_TEMPLATES.get(motivation_type, OUTREACH_TEMPLATES["Low Signal"])
        base_message = template.format(
            seller_name=seller_name,
            business_name=business_name,
            buyer_name=self.buyer_name,
            buyer_entity=self.buyer_entity,
            buyer_email=self.buyer_email,
        )

        # Add India-cultural note if family signals detected
        family_signals = (psych_data or {}).get("family_signals", [])
        if family_signals and motivation_type not in ("Family Exit",):
            base_message += "\n\nP.S. I understand this may be a family business — I'm happy to have the conversation at whatever pace works for everyone involved."

        if self.llm:
            from langchain_core.messages import HumanMessage, SystemMessage
            system = (
                f"You are {self.buyer_name}, an India micro-PE operator. "
                "Write outreach in YOUR voice — casual, direct, no corporate language. "
                "Max 150 words. Use the template as a base but personalize with context."
            )
            prompt = f"""
Personalize this outreach message using the deal context:

BASE TEMPLATE:
{base_message}

DEAL CONTEXT:
- Business: {business_name}
- Seller motivation: {motivation_type}
- Burnout score: {(psych_data or {}).get('burnout_score', 'N/A')}/10
- Fast close signal: {(psych_data or {}).get('fast_close_signal', False)}
- India risk level: {(reg_data or {}).get('india_risk_score', 'N/A')}/10
- Family signals: {family_signals[:3]}
{('MEMORY: ' + memory_context[:300]) if memory_context else ''}

Rules:
- Keep it under 130 words
- Sound like a real person, not a template
- Don't change the core offer structure
- Add ONE specific detail from the deal context that shows you've actually read their listing
"""
            try:
                resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
                return resp.content
            except Exception as e:
                logger.warning("LLM outreach failed: %s", e)

        return base_message

    def generate_loi(
        self,
        business_name: str,
        seller_name: str = "Founder",
        purchase_price_inr_cr: float = 2.0,
        upfront_pct: int = 90,
        holdback_days: int = 30,
        dd_days: int = 14,
        exclusivity_days: int = 21,
        transition_days: int = 30,
        structure_type: str = "Asset Purchase Agreement",
        business_type: str = "Private Limited",
    ) -> str:
        """Render a complete LOI document."""
        today = datetime.today()
        price_inr = price_inr_cr = purchase_price_inr_cr
        price_inr_lakh = price_inr * 100
        usd_approx = round(price_inr * 1_000_000 / 83.5, 0)  # approx ₹/$ rate

        upfront_inr = round(price_inr * upfront_pct / 100, 2)
        holdback_inr = round(price_inr * (100 - upfront_pct) / 100, 2)

        if "Proprietorship" in business_type or "Individual" in business_type:
            entity_condition = "Signed Business Transfer Agreement (BTA) from proprietor"
        else:
            entity_condition = "Board resolution authorizing share/asset sale + signed SH-4 (if share transfer)"

        close_date = (today + timedelta(days=dd_days + 14)).strftime("%B %d, %Y")
        expiry_date = (today + timedelta(days=7)).strftime("%B %d, %Y")

        return LOI_TEMPLATE.format(
            date=today.strftime("%B %d, %Y"),
            ref=today.strftime("%Y%m") + business_name[:4].upper().replace(" ", ""),
            buyer_name=self.buyer_name,
            buyer_entity=self.buyer_entity,
            seller_name=seller_name,
            business_name=business_name,
            purchase_price_inr=f"₹{price_inr:.2f} Crore",
            purchase_price_usd=f"${usd_approx:,.0f}",
            upfront_inr=f"₹{upfront_inr:.2f} Crore",
            upfront_pct=upfront_pct,
            holdback_inr=f"₹{holdback_inr:.2f} Crore",
            holdback_pct=100 - upfront_pct,
            holdback_days=holdback_days,
            structure_type=structure_type,
            entity_specific_condition=entity_condition,
            exclusivity_days=exclusivity_days,
            dd_days=dd_days,
            transition_days=transition_days,
            signing_days=3,
            close_date=close_date,
            expiry_date=expiry_date,
        )

    def generate_negotiation_playbook(
        self,
        motivation_type: str,
        valuation_data: dict,
        psych_data: dict,
        reg_data: dict,
        similar_deals: list[dict],
    ) -> dict:
        """Generate a deal-specific negotiation playbook."""
        low = valuation_data.get("blended_valuation_low_cr", 0)
        mid = valuation_data.get("blended_valuation_mid_cr", 0)
        high = valuation_data.get("blended_valuation_high_cr", 0)
        burnout = psych_data.get("burnout_score", 5)
        fast_close = psych_data.get("fast_close_signal", False)
        india_risk = reg_data.get("india_risk_score", 5)
        deal_killers = reg_data.get("deal_killers", [])

        # Opening anchor
        if burnout >= 7 or motivation_type in ("Burnout Exit", "Financial Distress"):
            opening_offer_cr = round(low * 0.9, 2)  # 10% below low
            walk_away_cr = round(high * 1.05, 2)
            strategy_tone = "Aggressive but respectful. Emphasize certainty and speed over price."
        elif motivation_type == "Family Exit":
            opening_offer_cr = round(mid * 0.95, 2)  # Close to mid (respect the legacy)
            walk_away_cr = round(high * 1.1, 2)
            strategy_tone = "Relationship-first. Don't open with price. Let them set the anchor."
        else:
            opening_offer_cr = round((low + mid) / 2, 2)
            walk_away_cr = round(high, 2)
            strategy_tone = "Standard. Lead with fit and certainty, then price."

        # Concession strategy
        concessions = []
        if fast_close:
            concessions.append("Offer to close in 21 days instead of 30 — high value signal for them")
        if burnout >= 6:
            concessions.append("Offer 95% upfront (instead of 90%) — reduces mental friction of earnout")
        if india_risk >= 6:
            concessions.append("Offer to absorb GST transition costs — removes a key risk they'd worry about")
        if deal_killers:
            concessions.append(f"Address '{deal_killers[0]['category']}' proactively in your first call — shows diligence")

        # Walk-away signals
        walk_away_triggers = [
            "Seller asks for earnout > 15% of total price",
            "Patriarch/family refuses to sign or engage",
            "Revenue verification shows >15% decline from stated numbers",
            "More than 2 pending GST notices or MCA defaults",
        ]
        if india_risk >= 8:
            walk_away_triggers.insert(0, "⚠️ CRITICAL: India risk score is very high — add extra walk-away conditions")

        # Memory-informed insight
        memory_insight = ""
        if similar_deals:
            best_match = similar_deals[0]
            memory_insight = best_match.get("memory_note", "")

        return {
            "opening_offer_cr": opening_offer_cr,
            "target_mid_cr": mid,
            "walk_away_cr": walk_away_cr,
            "strategy_tone": strategy_tone,
            "first_call_agenda": [
                "Warm up: understand the story behind the business",
                "Validate stated revenue (ask for bank statement month/current MRR)",
                "Identify all stakeholders who need to sign off",
                "Understand timeline pressure and flexibility",
                "End with: 'If the numbers check out, I can have something in writing this week'",
            ],
            "concessions_to_offer": concessions,
            "walk_away_triggers": walk_away_triggers,
            "memory_insight": memory_insight,
        }

    def generate_strategy_summary(
        self,
        deal_context: str,
        motivation_type: str,
        valuation_data: dict,
        psych_data: dict,
        reg_data: dict,
        memory_context: str = "",
    ) -> str:
        """Generate full strategy narrative. LLM if available, else template."""

        low = valuation_data.get("blended_valuation_low_cr", 0)
        mid = valuation_data.get("blended_valuation_mid_cr", 0)
        high = valuation_data.get("blended_valuation_high_cr", 0)
        motivation_profile = psych_data.get("motivation_profile", {})

        if self.llm:
            from langchain_core.messages import HumanMessage, SystemMessage
            system = (
                "You are Dev Shah, micro-PE operator at Pocket Fund. "
                "Write a crisp, actionable acquisition strategy note. IC memo style. Max 250 words."
            )
            prompt = f"""
Write an acquisition strategy note for this deal:

CONTEXT: {deal_context[:500]}
MOTIVATION: {motivation_type}
VALUATION RANGE: ₹{low:.2f}Cr – ₹{high:.2f}Cr (mid: ₹{mid:.2f}Cr)
APPROACH: {motivation_profile.get('negotiation_tip', '')}
INDIA RISK: {reg_data.get('india_risk_score', '?')}/10
MEMORY CONTEXT: {memory_context[:300] if memory_context else 'No comparable deals'}

Write:
1. Opening move (how to initiate)
2. Price negotiation strategy (opening anchor, concessions, walk-away)
3. India-specific deal angle (one insight the seller doesn't expect)
4. Recommended close timeline
"""
            try:
                resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
                return resp.content
            except Exception as e:
                logger.warning("LLM strategy failed: %s", e)

        # Template
        tip = motivation_profile.get("negotiation_tip", "Approach with direct intent.")
        urgency = motivation_profile.get("urgency", "unknown")
        return f"""
**Opening Move:**
{tip}

**Price Strategy:**
Open at ₹{round((low + mid)/2, 2):.2f}Cr. Target ₹{mid:.2f}Cr. Walk if above ₹{high:.2f}Cr without significant risk mitigation.

**India Angle:**
For a {motivation_type.lower()}, the emotional driver matters as much as the price. 
Lead with certainty ("Cash buyer, 21-day close") — not just numbers.

**Timeline:** {urgency.title()}
{('Memory: ' + memory_context[:150]) if memory_context else ''}
""".strip()
