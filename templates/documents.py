"""
Document Templates — LOI, NDA, and outreach messages for micro-PE deal flow.
All templates are parameterized and operator-voice-consistent.
"""

from datetime import datetime, timedelta
from typing import Optional


# ---------------------------------------------------------------------------
# LOI Template
# ---------------------------------------------------------------------------

LOI_TEMPLATE = """\
LETTER OF INTENT — ASSET PURCHASE

Date: {date}
Deal Reference: {deal_ref}

FROM:
{buyer_name}
{buyer_entity}
{buyer_email}

TO:
{seller_name} ("Seller")
{seller_handle}

RE: Non-Binding Letter of Intent — Acquisition of {asset_name}

---

Dear {seller_name},

This Letter of Intent ("LOI") sets out the non-binding terms under which {buyer_entity} 
("Buyer") proposes to acquire the digital business known as {asset_name} ("the Asset") 
from the Seller.

1. PURCHASE PRICE

   Buyer proposes a total purchase price of {purchase_price} (the "Purchase Price"), 
   payable as follows:
   - {upfront_pct}% ({upfront_amount}) in cash at closing
   - {holdback_pct}% ({holdback_amount}) held in escrow for {holdback_period} days 
     post-closing, released upon successful transition

2. TRANSACTION STRUCTURE

   This transaction is structured as a clean Asset Purchase (not a share purchase). 
   Assets to be transferred include:
   - All code, intellectual property, and proprietary software
   - Customer database and subscription list
   - Domain names, social accounts, and brand assets
   - All existing customer contracts and terms of service
   - Operational documentation and SOPs (to be delivered at close)

3. EXCLUSIONS

   The following are explicitly excluded:
   - Any personal liabilities of Seller
   - Any pre-closing tax obligations
   - Cash or receivables prior to closing date

4. DUE DILIGENCE PERIOD

   Buyer requests a {dd_period}-day exclusive due diligence period commencing upon 
   mutual signing of this LOI. During this period:
   - Seller will provide Stripe/revenue dashboards, churn data, codebase access, 
     and customer concentration breakdown
   - Seller will not solicit or accept competing offers

5. CLOSING TIMELINE

   Target closing: {closing_date}. Buyer is a cash buyer and can move quickly.
   No financing contingency.

6. TRANSITION SUPPORT

   Seller agrees to provide {transition_days} days of post-close transition support 
   (email/async), covering product documentation, customer introductions, and 
   operational handoff. Additional paid consulting available by mutual agreement.

7. CONFIDENTIALITY

   Both parties agree to treat all shared information as strictly confidential. 
   A formal NDA will be executed prior to data room access.

8. NON-BINDING NATURE

   This LOI is non-binding in its entirety (except for Clauses 4 and 7, which are 
   binding). A binding Purchase Agreement will be prepared by Buyer's counsel 
   upon completion of due diligence.

9. EXPIRATION

   This LOI expires if not countersigned by {loi_expiry}.

---

Accepted and agreed:

BUYER:

_______________________________
{buyer_name}
{buyer_entity}
Date: _______________

SELLER:

_______________________________
{seller_name}
Date: _______________

---
[Prepared using Pocket Fund Deal Sourcing Agent — For Illustrative Purposes]
"""


def generate_loi(
    asset_name: str,
    seller_name: str = "Founder",
    seller_handle: str = "",
    purchase_price: float = 0,
    upfront_pct: int = 90,
    holdback_period: int = 30,
    dd_period: int = 14,
    transition_days: int = 30,
    buyer_name: str = "Deal Principal",
    buyer_entity: str = "MicroPE Ventures LLC",
    buyer_email: str = "deals@micropefund.com",
) -> str:
    """Render a complete LOI document from template."""
    today = datetime.today()
    upfront_amount = purchase_price * upfront_pct / 100
    holdback_amount = purchase_price * (100 - upfront_pct) / 100
    closing_date = (today + timedelta(days=dd_period + 7)).strftime("%B %d, %Y")
    loi_expiry = (today + timedelta(days=5)).strftime("%B %d, %Y")
    deal_ref = f"PF-{today.strftime('%Y%m%d')}-{asset_name[:6].upper().replace(' ', '')}"

    return LOI_TEMPLATE.format(
        date=today.strftime("%B %d, %Y"),
        deal_ref=deal_ref,
        buyer_name=buyer_name,
        buyer_entity=buyer_entity,
        buyer_email=buyer_email,
        seller_name=seller_name,
        seller_handle=seller_handle or "Not provided",
        asset_name=asset_name,
        purchase_price=f"${purchase_price:,.0f}",
        upfront_pct=upfront_pct,
        upfront_amount=f"${upfront_amount:,.0f}",
        holdback_pct=100 - upfront_pct,
        holdback_amount=f"${holdback_amount:,.0f}",
        holdback_period=holdback_period,
        dd_period=dd_period,
        closing_date=closing_date,
        transition_days=transition_days,
        loi_expiry=loi_expiry,
    )


# ---------------------------------------------------------------------------
# Outreach Templates (Operator Voice)
# ---------------------------------------------------------------------------

OUTREACH_TEMPLATES = {
    "Burnout": """\
Hey {seller_name},

Saw your listing — sounds like you're ready to move on and I respect that.

Quick background: I'm a cash buyer, buy small digital businesses, close fast with no drama. \
This looks like a clean fit for what I'm looking for.

A few quick questions before I send over an LOI:
1. Is the {mrr_str} MRR holding steady or trending?
2. How tied in are you to daily operations?
3. Any preference on close timeline?

If the numbers hold up in a quick call, I can have a draft LOI to you within 48 hours.

No brokers, no BS, no drawn-out process. Just a clean offer.

Talk soon,
{buyer_name}
{buyer_entity}
{buyer_email}
""",

    "Pivot": """\
Hey {seller_name},

Saw you're selling — sounds like you've got something new you're focused on. \
Respect the move.

I'm a direct cash buyer, no financing, no middlemen. I look for exactly this kind of \
transition — founder moving on to something bigger, solid asset left behind.

If the underlying numbers are solid, I can move fast. Here's my usual process:
→ 15-min call to validate basics
→ LOI within 48 hours
→ Close in 2–3 weeks max

Happy to keep this async too if you prefer. What's the best way to connect?

{buyer_name}
{buyer_entity}
""",

    "Boredom": """\
Hey {seller_name},

Noticed the listing — it honestly sounds like this thing has been on autopilot \
and you've mentally moved on. Been there.

I'm a micro-PE operator. I buy small, boring, cash-flowing digital businesses \
and operate them long-term. This looks right in my wheelhouse.

Not going to waste your time with 20 questions — if you want, let's jump on a \
quick call and I'll tell you within 5 minutes if I'm serious.

Fast close, asset purchase, no earnout. My kind of deal.

{buyer_name}
{buyer_email}
""",

    "Financial": """\
Hey {seller_name},

Saw the listing. If you need to move quickly I can work on that timeline — \
I'm a direct cash buyer and I don't need financing approval.

I'm not here to lowball you. I want a fair price for a clean asset. \
If the fundamentals check out, I'll have an LOI in your inbox fast.

Let me know the best way to connect. Happy to do a quick async Q&A if that's easier.

{buyer_name}
{buyer_entity}
{buyer_email}
""",

    "Unknown": """\
Hey {seller_name},

Came across your listing — looks interesting. I'm a cash buyer focused on \
small digital businesses under $100k. Clean asset purchases, fast close, no drama.

If you're serious about selling, let's talk. I don't need weeks to make a decision.

{buyer_name}
{buyer_entity}
""",
}


def generate_outreach(
    intent_label: str,
    seller_name: str = "Founder",
    mrr: Optional[float] = None,
    buyer_name: str = "Deal Principal",
    buyer_entity: str = "MicroPE Ventures LLC",
    buyer_email: str = "deals@micropefund.com",
) -> str:
    """Generate a personalized outreach message based on seller intent."""
    template = OUTREACH_TEMPLATES.get(intent_label, OUTREACH_TEMPLATES["Unknown"])
    mrr_str = f"${mrr:,.0f}" if mrr else "stated"

    return template.format(
        seller_name=seller_name,
        mrr_str=mrr_str,
        buyer_name=buyer_name,
        buyer_entity=buyer_entity,
        buyer_email=buyer_email,
    )


# ---------------------------------------------------------------------------
# NDA Template (Lightweight)
# ---------------------------------------------------------------------------

NDA_TEMPLATE = """\
MUTUAL NON-DISCLOSURE AGREEMENT

Date: {date}
Reference: {deal_ref}

PARTIES:

Disclosing Party: {seller_name} ("Seller"), owner of {asset_name}
Receiving Party: {buyer_name}, {buyer_entity} ("Buyer")

1. PURPOSE

The parties wish to explore a potential acquisition transaction involving {asset_name} 
(the "Transaction"). In connection with this exploration, each party may disclose 
confidential information to the other.

2. CONFIDENTIAL INFORMATION

"Confidential Information" means any non-public business, financial, technical, or 
operational information disclosed by either party, including but not limited to: 
revenue data, customer lists, source code, and financial statements.

3. OBLIGATIONS

Each receiving party agrees to:
(a) Hold all Confidential Information in strict confidence
(b) Not use it for any purpose other than evaluating the Transaction
(c) Not disclose it to any third party without prior written consent

4. TERM

This Agreement shall remain in effect for 24 months from the date of signing.

5. EXCEPTIONS

These obligations do not apply to information that is: (a) publicly known, 
(b) independently developed, or (c) required to be disclosed by law.

6. GOVERNING LAW

This Agreement shall be governed by the laws of the State of Delaware.

---

BUYER:                                    SELLER:

_________________________                 _________________________
{buyer_name}                              {seller_name}
{buyer_entity}                            
Date: ____________                        Date: ____________

---
[Prepared using Pocket Fund Deal Sourcing Agent — For Illustrative Purposes]
"""


def generate_nda(
    asset_name: str,
    seller_name: str = "Founder",
    buyer_name: str = "Deal Principal",
    buyer_entity: str = "MicroPE Ventures LLC",
) -> str:
    """Render a lightweight NDA document."""
    today = datetime.today()
    deal_ref = f"NDA-{today.strftime('%Y%m%d')}-{asset_name[:6].upper().replace(' ', '')}"

    return NDA_TEMPLATE.format(
        date=today.strftime("%B %d, %Y"),
        deal_ref=deal_ref,
        seller_name=seller_name,
        asset_name=asset_name,
        buyer_name=buyer_name,
        buyer_entity=buyer_entity,
    )
