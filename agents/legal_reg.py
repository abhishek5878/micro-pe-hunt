"""
Legal & Regulatory Intelligence Agent.
Flags India-specific compliance risks for micro-PE acquisitions.
Simulates MCA filing checks, GST status, and state-level regulatory concerns.
All signals are mocked/estimated — real diligence requires actual API access.
"""

import json
import logging
import random
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

KB_PATH = Path(__file__).parent.parent / "data" / "sme_knowledge.json"


def _load_kb() -> dict:
    with open(KB_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Risk Badge Definitions
# ─────────────────────────────────────────────────────────────────────────────

RISK_LEVELS = {
    "GREEN": {"color": "#10b981", "bg": "#022c22", "label": "LOW RISK", "icon": "✅"},
    "YELLOW": {"color": "#f59e0b", "bg": "#2d1f00", "label": "CAUTION", "icon": "⚠️"},
    "RED": {"color": "#ef4444", "bg": "#2d0000", "label": "HIGH RISK", "icon": "🔴"},
    "BLOCKER": {"color": "#dc2626", "bg": "#450a0a", "label": "DEAL-KILLER", "icon": "🚫"},
}

# ─────────────────────────────────────────────────────────────────────────────
# Simulated MCA / GST Signal Engine
# ─────────────────────────────────────────────────────────────────────────────

def simulate_mca_signals(business_type: str, revenue_cr: float, years_old: int = 5) -> dict:
    """
    Simulate MCA filing signals for a business.
    In production: integrate MCA21 API or Zauba/Tracxn data.
    """
    is_pvt_ltd = "private limited" in business_type.lower() or "pvt" in business_type.lower()
    is_proprietorship = "proprietorship" in business_type.lower() or "individual" in business_type.lower()

    if is_pvt_ltd:
        # PVT LTD: needs ROC filings, annual returns
        filings_current = random.random() > 0.2  # 80% chance current
        has_dir_kyc = random.random() > 0.1
        has_aoc4 = random.random() > 0.15  # Form AOC-4 (annual accounts)
        has_mgt7 = random.random() > 0.15  # Form MGT-7 (annual return)
        return {
            "entity_type": "Private Limited Company",
            "roc_filings_current": filings_current,
            "director_kyc_current": has_dir_kyc,
            "form_aoc4_filed": has_aoc4,
            "form_mgt7_filed": has_mgt7,
            "share_transfer_feasible": True,
            "stamp_duty_on_transfer": "0.25% on share value (varies by state)",
            "transfer_complexity": "Medium — requires board resolution, SH-4, ROC intimation",
            "notes": "Share purchase possible but asset purchase recommended for cleaner transfer.",
        }
    elif is_proprietorship:
        return {
            "entity_type": "Sole Proprietorship",
            "roc_filings_current": True,  # No ROC for proprietorship
            "director_kyc_current": True,
            "form_aoc4_filed": False,
            "form_mgt7_filed": False,
            "share_transfer_feasible": False,
            "stamp_duty_on_transfer": "Asset purchase only — stamp duty on specific assets (state-varies)",
            "transfer_complexity": "Low — asset purchase agreement, business transfer agreement",
            "notes": "No share structure. Must be asset purchase. GST implications on asset transfer.",
        }
    else:
        return {
            "entity_type": business_type or "Unknown",
            "roc_filings_current": None,
            "director_kyc_current": None,
            "form_aoc4_filed": None,
            "form_mgt7_filed": None,
            "share_transfer_feasible": None,
            "stamp_duty_on_transfer": "Verify entity type first",
            "transfer_complexity": "Unknown — verify entity type",
            "notes": "Entity type not clear. Request incorporation documents.",
        }


def simulate_gst_signals(gst_registered: bool, revenue_cr: float, sector_key: str) -> dict:
    """Simulate GST compliance signals."""
    if not gst_registered:
        return {
            "gst_registered": False,
            "gst_returns_current": False,
            "gst_liability_on_transfer": "Business transfer without GST may be treated as supply — buyer takes on liability",
            "gst_risk_level": "RED",
            "gst_note": "CRITICAL: Unregistered business must verify if GST registration was mandatory (>₹40L revenue typically). Buyer risks retroactive GST demand.",
            "tcs_applicable": False,
        }

    # Simulated compliance signals for registered business
    returns_current = random.random() > 0.15  # 85% chance current
    has_gstr1 = random.random() > 0.1
    has_gstr3b = random.random() > 0.1
    has_gstr9 = random.random() > 0.2  # Annual return

    # TCS (Tax Collected at Source) for digital businesses
    tcs_applicable = sector_key in ("saas", "automation_ai_tools", "d2c_ecommerce") and revenue_cr > 0.5

    return {
        "gst_registered": True,
        "gst_returns_current": returns_current,
        "gstr1_filed": has_gstr1,
        "gstr3b_filed": has_gstr3b,
        "gstr9_filed": has_gstr9,
        "gst_risk_level": "GREEN" if (returns_current and has_gstr1 and has_gstr3b) else "YELLOW",
        "gst_note": (
            "Clean GST compliance reduces buyer risk. Transfer via business transfer agreement (BTA) "
            "or slump sale; verify GST treatment on transfer of going concern."
            if returns_current else
            "⚠️ GST returns appear delayed. Request last 12 months GSTR-3B + GSTR-1 before close."
        ),
        "tcs_applicable": tcs_applicable,
        "tcs_note": "TCS on digital payments applicable — verify compliance" if tcs_applicable else None,
        "itc_credit_status": "Verify Input Tax Credit (ITC) availability — check GSTR-2B reconciliation",
    }


def simulate_udyam_signals(udyam_registered: bool, revenue_cr: float) -> dict:
    """Simulate Udyam registration signals."""
    if not udyam_registered:
        return {
            "udyam_registered": False,
            "udyam_benefits_transferable": False,
            "note": "Not Udyam registered — buyer misses out on MSME priority sector benefits, TReDS access, and PSU vendor preferences.",
            "recommendation": "Register under Udyam post-acquisition (process is online, usually 48-72 hours).",
        }

    # Classify tier
    from agents.analyst import classify_msme
    msme_class = classify_msme(revenue_cr)

    return {
        "udyam_registered": True,
        "classification": msme_class["label"],
        "udyam_benefits_transferable": True,
        "benefits": [
            "Priority sector lending benefits",
            "MSME subsidy schemes access",
            "TReDS invoice discounting access",
            "Government tender preference (MSME clause)",
            "Protection under MSMED Act payment terms (45-day rule)",
        ],
        "transfer_note": "Udyam registration is entity-linked. New buyer needs fresh Udyam if entity changes. Add to LOI as condition.",
        "note": "Udyam certified — strategic value for PSU/govt vendor opportunities.",
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main Legal/Reg Agent
# ─────────────────────────────────────────────────────────────────────────────

class LegalRegAgent:
    """
    India-specific legal and regulatory risk flagging agent.
    Produces: risk flags (colored), state-specific risks, deal-structure recommendation.
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.kb = _load_kb()

    def _compute_compliance_score(self, gst_data: dict, mca_data: dict, udyam_data: dict) -> tuple[int, str]:
        """Compute an overall compliance score 0–100."""
        weights = self.kb["compliance_score_weights"]
        score = 0

        if gst_data.get("gst_registered"):
            score += weights["gst_registered"]
        if gst_data.get("gst_returns_current"):
            score += weights["gst_returns_current"]
        if udyam_data.get("udyam_registered"):
            score += weights["udyam_registered"]
        if mca_data.get("roc_filings_current"):
            score += weights["mca_filings_current"]
        if mca_data.get("director_kyc_current"):
            score += 5
        if udyam_data.get("udyam_registered") or gst_data.get("gst_registered"):
            score += weights["clean_bank_statements"] // 2  # assumed if registered

        return min(score, 100), (
            "Strong" if score >= 75 else "Moderate" if score >= 50 else "Weak"
        )

    def analyze(
        self,
        sector_key: str,
        state: str,
        business_type: str = "Private Limited",
        gst_registered: bool = True,
        udyam_registered: bool = False,
        is_family_run: bool = False,
        revenue_cr: float = 1.0,
        years_old: int = 5,
        special_risks: list[str] = None,
    ) -> dict:
        """
        Full legal/regulatory analysis.
        Returns structured risk dict with colored flags.
        """
        # Simulated signals
        mca_data = simulate_mca_signals(business_type, revenue_cr, years_old)
        gst_data = simulate_gst_signals(gst_registered, revenue_cr, sector_key)
        udyam_data = simulate_udyam_signals(udyam_registered, revenue_cr)

        # State profile
        state_data = self.kb["india_state_profiles"].get(state, {})

        compliance_score, compliance_label = self._compute_compliance_score(gst_data, mca_data, udyam_data)

        # ── Build Risk Flags ──────────────────────────────────────────────────
        flags = []

        # GST
        gst_level = gst_data.get("gst_risk_level", "YELLOW")
        flags.append({
            "category": "GST Compliance",
            "level": gst_level,
            "badge": RISK_LEVELS[gst_level],
            "detail": gst_data.get("gst_note", ""),
            "action": "Request GST compliance certificate from CA" if gst_level != "GREEN" else "Verify with CA — appears clean",
        })

        # MCA / Entity Structure
        transfer_complexity = mca_data.get("transfer_complexity", "Unknown")
        entity_level = "GREEN" if "Low" in transfer_complexity else "YELLOW" if "Medium" in transfer_complexity else "RED"
        flags.append({
            "category": "Entity Structure & Transfer",
            "level": entity_level,
            "badge": RISK_LEVELS[entity_level],
            "detail": f"{mca_data['entity_type']} — {mca_data['transfer_complexity']}",
            "action": mca_data.get("notes", ""),
        })

        # Family run
        if is_family_run:
            flags.append({
                "category": "Family Business Risk",
                "level": "RED",
                "badge": RISK_LEVELS["RED"],
                "detail": "Family-run businesses in India carry 20–30% handover risk discount. Patriarch alignment and succession clarity are critical.",
                "action": "Meet all key stakeholders (founder + patriarch) before issuing LOI. Add family sign-off as condition.",
            })

        # Udyam
        udyam_level = "GREEN" if udyam_data.get("udyam_registered") else "YELLOW"
        flags.append({
            "category": "Udyam / MSME Registration",
            "level": udyam_level,
            "badge": RISK_LEVELS[udyam_level],
            "detail": udyam_data.get("note", ""),
            "action": udyam_data.get("recommendation", "Verify current Udyam certificate"),
        })

        # Sector-specific risks
        sector_data = self.kb["sectors"].get(sector_key, {})
        for risk in sector_data.get("key_risks", [])[:3]:
            flags.append({
                "category": f"Sector Risk: {risk.replace('_', ' ').title()}",
                "level": "YELLOW",
                "badge": RISK_LEVELS["YELLOW"],
                "detail": f"Known risk for {sector_data.get('label', sector_key)} deals",
                "action": f"Add specific diligence questions around {risk.replace('_', ' ')}",
            })

        # State-specific risks
        for state_risk in state_data.get("state_specific_risks", [])[:2]:
            flags.append({
                "category": f"{state} State Risk",
                "level": "YELLOW",
                "badge": RISK_LEVELS["YELLOW"],
                "detail": state_risk,
                "action": f"Consult local {state} CA/advocate for current status",
            })

        # Additional special risks passed in
        if special_risks:
            for risk in special_risks:
                flags.append({
                    "category": "Special Risk",
                    "level": "RED",
                    "badge": RISK_LEVELS["RED"],
                    "detail": risk,
                    "action": "Flag for legal review before LOI",
                })

        # TCS check for digital
        if gst_data.get("tcs_applicable"):
            flags.append({
                "category": "TCS Digital Compliance",
                "level": "YELLOW",
                "badge": RISK_LEVELS["YELLOW"],
                "detail": gst_data.get("tcs_note", ""),
                "action": "Verify TCS collection and remittance with CA",
            })

        # Deal structure recommendation
        deal_structure = self._recommend_structure(mca_data, gst_data, is_family_run, sector_key)

        # Overall India risk score (0–10)
        blocker_count = sum(1 for f in flags if f["level"] == "BLOCKER")
        red_count = sum(1 for f in flags if f["level"] == "RED")
        yellow_count = sum(1 for f in flags if f["level"] == "YELLOW")
        india_risk_score = min(blocker_count * 4 + red_count * 2 + yellow_count * 0.5, 10)

        return {
            "compliance_score": compliance_score,
            "compliance_label": compliance_label,
            "india_risk_score": round(india_risk_score, 1),
            "flags": flags,
            "mca_signals": mca_data,
            "gst_signals": gst_data,
            "udyam_signals": udyam_data,
            "state_profile": state_data,
            "deal_structure_recommendation": deal_structure,
            "deal_killers": [f for f in flags if f["level"] in ("RED", "BLOCKER")],
        }

    def _recommend_structure(self, mca_data: dict, gst_data: dict, is_family_run: bool, sector_key: str) -> dict:
        """Recommend the deal structure based on legal findings."""
        entity = mca_data.get("entity_type", "Unknown")

        if "Proprietorship" in entity or "Individual" in entity:
            structure = "Asset Purchase Agreement (APA)"
            rationale = "No share structure exists. Transfer IP, customer contracts, domain, accounts, and specific assets via APA."
            complexity = "Low"
        elif is_family_run:
            structure = "Asset Purchase (preferred) or Share Purchase with extensive reps & warranties"
            rationale = "Family businesses carry unknown liabilities. Asset purchase ring-fences buyer from legacy issues."
            complexity = "High"
        else:
            structure = "Asset Purchase Agreement or Share Purchase"
            rationale = "Clean PVT LTD with current filings — share purchase feasible with standard reps & warranties."
            complexity = "Medium"

        return {
            "recommended_structure": structure,
            "rationale": rationale,
            "complexity": complexity,
            "key_documents": [
                "Letter of Intent (LOI) with exclusivity",
                "Non-Disclosure Agreement (NDA)",
                "Business Transfer Agreement / Share Purchase Agreement",
                "GST-adjusted Slump Sale deed (if applicable)",
                "IP Assignment Agreement",
                "Franchisor/Customer Consent letters (if required)",
                "Transition Services Agreement (TSA)",
            ],
            "india_specific_docs": [
                "Udyam certificate transfer request letter",
                "Shops & Establishments Act change of ownership",
                "GST migration notification to GST department",
                "PF/ESI transfer for employees (if applicable)",
            ],
        }

    def generate_reg_narrative(self, reg_data: dict, deal_context: str) -> str:
        """Generate regulatory narrative. LLM if available, else template."""

        if self.llm:
            from langchain_core.messages import HumanMessage, SystemMessage
            system = (
                "You are an India M&A legal advisor specializing in MSME acquisitions. "
                "Write in bullet-point IC memo style. Max 200 words. Be practical and specific."
            )
            flags_summary = "\n".join(
                f"- [{f['level']}] {f['category']}: {f['detail'][:80]}"
                for f in reg_data["flags"][:6]
            )
            prompt = f"""
Write a regulatory risk brief for this India MSME acquisition:

DEAL CONTEXT: {deal_context[:400]}

COMPLIANCE SCORE: {reg_data['compliance_score']}/100 ({reg_data['compliance_label']})
INDIA RISK SCORE: {reg_data['india_risk_score']}/10

KEY FLAGS:
{flags_summary}

DEAL STRUCTURE RECOMMENDATION: {reg_data['deal_structure_recommendation']['recommended_structure']}

Write:
1. Overall regulatory risk summary (1 sentence)
2. Top 3 items to resolve before LOI
3. Structure recommendation and why
"""
            try:
                resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
                return resp.content
            except Exception as e:
                logger.warning("LLM reg narrative failed: %s", e)

        # Template fallback
        score = reg_data["compliance_score"]
        india_risk = reg_data["india_risk_score"]
        structure = reg_data["deal_structure_recommendation"]["recommended_structure"]
        deal_killers = reg_data["deal_killers"]

        return f"""
**Compliance Overview:** {score}/100 ({reg_data['compliance_label']}) | India Risk {india_risk}/10

**Recommended Structure:** {structure}
{reg_data['deal_structure_recommendation']['rationale']}

**Pre-LOI Items ({len(deal_killers)} red flags):**
{chr(10).join('🔴 ' + f['category'] + ': ' + f['detail'][:80] for f in deal_killers[:3]) if deal_killers else '✅ No deal-killers identified.'}

**Key India Documents Required:**
{chr(10).join('• ' + doc for doc in reg_data['deal_structure_recommendation']['india_specific_docs'][:4])}
""".strip()
