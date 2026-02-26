"""
Valuation Analyst Agent.
Computes India-adjusted acquisition multiples using sector knowledge base.
Applies family-run discounts, GST premiums, state premiums, and momentum adjustments.
Works with or without an LLM — heuristic engine is standalone.
"""

import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

KB_PATH = Path(__file__).parent.parent / "data" / "sme_knowledge.json"


def _load_kb() -> dict:
    with open(KB_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# MSME Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_msme(revenue_cr: float, investment_cr: float = 0) -> dict:
    """Classify business under revised 2025 MSME Act."""
    kb = _load_kb()
    classes = kb["msme_classification_2025"]
    for tier, limits in [("micro", classes["micro"]), ("small", classes["small"]), ("medium", classes["medium"])]:
        if revenue_cr <= limits["turnover_ceiling_cr"]:
            return {
                "tier": tier,
                "label": limits["label"],
                "turnover_ceiling_cr": limits["turnover_ceiling_cr"],
                "investment_ceiling_cr": limits["investment_ceiling_cr"],
            }
    return {"tier": "large", "label": "Large Enterprise (exceeds MSME)", "turnover_ceiling_cr": None}


# ─────────────────────────────────────────────────────────────────────────────
# Core Valuation Engine
# ─────────────────────────────────────────────────────────────────────────────

class ValuationAnalyst:
    """
    India-adjusted valuation analyst.
    All math is deterministic; LLM adds narrative colour only.
    """

    def __init__(self, llm=None):
        self.llm = llm
        self.kb = _load_kb()

    def _get_sector_data(self, sector_key: str) -> dict:
        sectors = self.kb["sectors"]
        if sector_key in sectors:
            return sectors[sector_key]
        # Fuzzy match fallback
        for key, val in sectors.items():
            if key.split("_")[0] in sector_key.lower():
                return val
        return sectors["manufacturing_general"]

    def _get_state_data(self, state: str) -> dict:
        states = self.kb["india_state_profiles"]
        return states.get(state, {"avg_valuation_premium_pct": 0, "labour_law_complexity": "medium"})

    # ── Adjustment Logic ──────────────────────────────────────────────────────

    def compute_adjustments(
        self,
        sector_key: str,
        is_family_run: bool = False,
        gst_clean: bool = True,
        udyam_registered: bool = False,
        digital_ready: bool = False,
        state: str = "Maharashtra",
        conviction_score: float = 5.0,
        burnout_score: float = 5.0,
    ) -> dict:
        """
        Compute all multiplicative adjustments to the base multiple.
        Returns a breakdown dict for transparent display.
        """
        sector = self._get_sector_data(sector_key)
        state_data = self._get_state_data(state)

        adjustments = {}

        # 1. Family run discount
        if is_family_run:
            disc = -sector.get("family_run_discount", 0.25)
            adjustments["Family-Run Discount"] = {"value": disc, "reason": "Handover risk, succession friction, patriarch involvement"}
        else:
            adjustments["Family-Run Discount"] = {"value": 0.0, "reason": "No family involvement detected"}

        # 2. GST clean premium
        if gst_clean:
            prem = sector.get("gst_clean_premium", 0.10)
            adjustments["GST Clean Premium"] = {"value": prem, "reason": "Clean GST compliance reduces buyer risk"}
        else:
            adjustments["GST Clean Premium"] = {"value": -0.15, "reason": "No GST registration — significant compliance risk + buyer liability"}

        # 3. Digital/tech premium
        if digital_ready:
            prem = sector.get("digital_premium", 0.0)
            adjustments["Digital Ready Premium"] = {"value": prem, "reason": "Clean tech stack, cloud infra, or digital-native ops"}
        else:
            adjustments["Digital Ready Premium"] = {"value": 0.0, "reason": "Traditional/offline ops"}

        # 4. Udyam premium
        if udyam_registered:
            adjustments["Udyam Registered"] = {"value": 0.05, "reason": "MSME scheme benefits transfer, adds strategic value to buyer"}
        else:
            adjustments["Udyam Registered"] = {"value": 0.0, "reason": "Not Udyam registered"}

        # 5. State premium/discount
        state_adj = state_data.get("avg_valuation_premium_pct", 0) / 100
        if state_adj != 0:
            adjustments[f"{state} State Factor"] = {
                "value": state_adj,
                "reason": f"Deal density, labour law complexity, and market depth in {state}",
            }

        # 6. Seller motivation premium (boredom/burnout → accept lower multiple)
        if burnout_score >= 7:
            adjustments["Burnout Motivation Discount (Buyer Advantage)"] = {
                "value": -0.08,
                "reason": "High burnout — seller likely to accept 15–20% below market for fast close",
            }
        elif burnout_score >= 5:
            adjustments["Moderate Exit Motivation"] = {
                "value": -0.04,
                "reason": "Moderate exit pressure — mild negotiation advantage for buyer",
            }

        # 7. Founder momentum/conviction premium
        if conviction_score >= 8:
            adjustments["High Business Conviction Premium"] = {
                "value": 0.05,
                "reason": "Strong momentum signals — seller may hold for higher price",
            }

        return adjustments

    # ── Valuation Computation ─────────────────────────────────────────────────

    def compute_valuation(
        self,
        sector_key: str,
        revenue_cr: float,
        ebitda_l: float,
        is_family_run: bool = False,
        gst_clean: bool = True,
        udyam_registered: bool = False,
        digital_ready: bool = False,
        state: str = "Maharashtra",
        conviction_score: float = 5.0,
        burnout_score: float = 5.0,
    ) -> dict:
        """
        Full valuation model. Returns low/mid/high range + narrative.
        All values in ₹ Crore.
        """
        sector = self._get_sector_data(sector_key)
        ebitda_cr = ebitda_l / 100  # convert to Crore
        revenue_cr = max(revenue_cr, 0.001)
        ebitda_margin = (ebitda_cr / revenue_cr) * 100 if revenue_cr > 0 else 0

        # Base multiples
        base_ebitda_low = sector["ebitda_multiple_low"]
        base_ebitda_high = sector["ebitda_multiple_high"]
        base_rev_low = sector["revenue_multiple_low"]
        base_rev_high = sector["revenue_multiple_high"]

        # Adjustments
        adjustments = self.compute_adjustments(
            sector_key, is_family_run, gst_clean, udyam_registered,
            digital_ready, state, conviction_score, burnout_score,
        )
        total_adj = sum(a["value"] for a in adjustments.values())
        adj_multiplier = 1 + total_adj

        # Adjusted multiples
        adj_ebitda_low = base_ebitda_low * adj_multiplier
        adj_ebitda_high = base_ebitda_high * adj_multiplier
        adj_rev_low = base_rev_low * adj_multiplier
        adj_rev_high = base_rev_high * adj_multiplier

        # Valuation ranges in ₹Cr
        ebitda_val_low = round(ebitda_cr * adj_ebitda_low, 2)
        ebitda_val_high = round(ebitda_cr * adj_ebitda_high, 2)
        rev_val_low = round(revenue_cr * adj_rev_low, 2)
        rev_val_high = round(revenue_cr * adj_rev_high, 2)

        # Blended (EBITDA-weighted for manufacturing, Revenue-weighted for SaaS/digital)
        digital_sectors = {"saas", "automation_ai_tools", "newsletter_media", "d2c_ecommerce", "education_edtech"}
        if sector_key in digital_sectors or ebitda_margin > 40:
            # Revenue multiple primary for high-margin digital
            blended_low = (rev_val_low * 0.6 + ebitda_val_low * 0.4)
            blended_high = (rev_val_high * 0.6 + ebitda_val_high * 0.4)
            primary_method = "Revenue multiple (primary for digital/SaaS)"
        else:
            # EBITDA primary for traditional business
            blended_low = (ebitda_val_low * 0.7 + rev_val_low * 0.3)
            blended_high = (ebitda_val_high * 0.7 + rev_val_high * 0.3)
            primary_method = "EBITDA multiple (primary for asset/ops businesses)"

        blended_mid = round((blended_low + blended_high) / 2, 2)
        blended_low = round(blended_low, 2)
        blended_high = round(blended_high, 2)

        # MSME classification
        msme = classify_msme(revenue_cr)

        return {
            "sector_label": sector["label"],
            "primary_method": primary_method,
            "revenue_cr": revenue_cr,
            "ebitda_cr": ebitda_cr,
            "ebitda_margin_pct": round(ebitda_margin, 1),
            "msme_class": msme,

            # Base multiples
            "base_ebitda_multiple_range": f"{base_ebitda_low}x – {base_ebitda_high}x",
            "base_revenue_multiple_range": f"{base_rev_low}x – {base_rev_high}x",

            # Adjusted multiples
            "adj_ebitda_multiple_range": f"{adj_ebitda_low:.1f}x – {adj_ebitda_high:.1f}x",
            "adj_revenue_multiple_range": f"{adj_rev_low:.1f}x – {adj_rev_high}x",
            "total_adjustment_pct": round(total_adj * 100, 1),

            # Valuations
            "ebitda_valuation_low_cr": ebitda_val_low,
            "ebitda_valuation_high_cr": ebitda_val_high,
            "revenue_valuation_low_cr": rev_val_low,
            "revenue_valuation_high_cr": rev_val_high,
            "blended_valuation_low_cr": blended_low,
            "blended_valuation_mid_cr": blended_mid,
            "blended_valuation_high_cr": blended_high,

            # Adjustment breakdown
            "adjustments": adjustments,

            # Derived
            "sector_notes": sector.get("notes", ""),
            "key_risks": sector.get("key_risks", []),
        }

    # ── Comparable Comps ──────────────────────────────────────────────────────

    def generate_comps_chart_data(self, sector_key: str, blended_mid: float, similar_deals: list[dict]) -> list[dict]:
        """Generate data for comparable deals bar chart."""
        comps = [{"label": "This Deal", "value": blended_mid, "color": "#f59e0b"}]

        for deal in similar_deals[:3]:
            meta = deal.get("full_deal", deal.get("metadata", {}))
            rev = meta.get("revenue_cr", 0)
            ebitda = meta.get("ebitda_l", 0) / 100 if meta.get("ebitda_l") else 0
            verdict = meta.get("verdict", "?")
            title = meta.get("title", "Past Deal")[:40]
            color = "#ef4444" if verdict == "HOT" else "#3b82f6" if verdict == "WARM" else "#6b7280"

            # Estimate their valuation from sector mid
            if ebitda > 0:
                sector_data = self._get_sector_data(meta.get("sector", sector_key))
                mid_mult = (sector_data["ebitda_multiple_low"] + sector_data["ebitda_multiple_high"]) / 2
                val = round(ebitda * mid_mult, 2)
            elif rev > 0:
                sector_data = self._get_sector_data(meta.get("sector", sector_key))
                mid_rev = (sector_data["revenue_multiple_low"] + sector_data["revenue_multiple_high"]) / 2
                val = round(rev * mid_rev, 2)
            else:
                continue

            comps.append({"label": title, "value": val, "color": color})

        return comps

    # ── LLM Narrative ─────────────────────────────────────────────────────────

    def generate_narrative(self, valuation_data: dict, deal_context: str, memory_notes: list[str]) -> str:
        """Generate a 3-paragraph analyst narrative. Uses LLM if available, else template."""

        memory_str = "\n".join(memory_notes) if memory_notes else "No comparable deals in memory."

        if self.llm:
            from langchain.schema import HumanMessage, SystemMessage
            system = (
                "You are a senior India micro-PE acquisition analyst at Pocket Fund. "
                "Write in precise, IC-memo style. Max 250 words. No fluff. "
                "Reference comparable deals from memory when provided."
            )
            prompt = f"""
Write a 3-paragraph valuation analyst note for this deal:

DEAL CONTEXT: {deal_context}

VALUATION COMPUTED:
- EBITDA: ₹{valuation_data['ebitda_cr']:.2f}Cr ({valuation_data['ebitda_margin_pct']}% margin)
- Blended Range: ₹{valuation_data['blended_valuation_low_cr']}Cr – ₹{valuation_data['blended_valuation_high_cr']}Cr
- Key adjustments: {valuation_data['total_adjustment_pct']}% net
- Primary risks: {', '.join(valuation_data['key_risks'][:3])}

MEMORY CONTEXT (comparable past deals):
{memory_str}

Para 1: Valuation rationale and methodology
Para 2: Key adjustments and India-specific factors  
Para 3: Entry price recommendation and negotiation angle
"""
            try:
                resp = self.llm.invoke([SystemMessage(content=system), HumanMessage(content=prompt)])
                return resp.content
            except Exception as e:
                logger.warning("LLM narrative failed: %s", e)

        # Template fallback
        adj = valuation_data["total_adjustment_pct"]
        adj_dir = "premium" if adj > 0 else "discount"
        return f"""
**Valuation Methodology:** Blended {valuation_data['primary_method']} applied. Base multiples for {valuation_data['sector_label']} 
adjusted by {abs(adj):.1f}% net {adj_dir} (EBITDA {valuation_data['adj_ebitda_multiple_range']}). 
India-specific factors dominate the adjustment stack.

**India Adjustment Logic:** The net {abs(adj):.1f}% adjustment reflects India micro-PE reality: 
{', '.join([k for k, v in valuation_data['adjustments'].items() if v['value'] != 0][:3])}. 
These are structural India quirks that global comps databases systematically ignore.

**Entry Price Recommendation:** Target the midpoint (₹{valuation_data['blended_valuation_mid_cr']:.2f}Cr) 
as opening anchor. Walk if seller insists above ₹{valuation_data['blended_valuation_high_cr']:.2f}Cr without 
significant de-risking on the family/compliance flags identified.
{chr(10) + '**Memory Comparison:** ' + memory_notes[0] if memory_notes else ''}
""".strip()
