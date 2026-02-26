"""
Persistent memory store — SQLite-backed deal history and shared agent context.
"Context > Code": rich deal history makes agents smarter across every run.
"""

import json
import logging
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

DB_PATH = Path(__file__).parent.parent / "pocket_fund.db"

# ─────────────────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS deals (
    id TEXT PRIMARY KEY,
    source TEXT,
    title TEXT,
    url TEXT,
    location TEXT,
    asking_price_usd REAL,
    asking_price_inr REAL,
    mrr_usd REAL,
    mrr_inr REAL,
    multiple TEXT,
    listing_age_days INTEGER,
    seller_name TEXT,
    seller_handle TEXT,
    tags TEXT,                      -- JSON array
    india_flags TEXT,               -- JSON object
    founder_signals TEXT,           -- JSON object
    listing_text TEXT,
    -- Agent outputs (populated as pipeline runs)
    motivation_score REAL,
    handover_risk REAL,
    india_risk_score REAL,
    conviction_score REAL,
    intent_label TEXT,
    boredom_multiple_likely INTEGER,
    red_flags TEXT,                 -- JSON array
    green_flags TEXT,               -- JSON array
    acquisition_thesis TEXT,
    deal_verdict TEXT,              -- HOT / WARM / PASS
    outreach_draft TEXT,
    loi_outline TEXT,
    deal_structure_note TEXT,
    -- Lifecycle
    status TEXT DEFAULT 'new',     -- new / contacted / negotiating / closed / passed
    notes TEXT,
    first_seen TEXT,
    last_updated TEXT
);

CREATE TABLE IF NOT EXISTS agent_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE,
    timestamp TEXT,
    mode TEXT,                      -- live / demo
    sources TEXT,                   -- JSON array
    deals_sourced INTEGER,
    deals_qualified INTEGER,
    hot_deals INTEGER,
    warm_deals INTEGER,
    daily_brief TEXT,
    full_sourcer_output TEXT,
    full_qualifier_output TEXT,
    full_strategist_output TEXT,
    total_runtime_seconds REAL
);

CREATE TABLE IF NOT EXISTS context_store (
    key TEXT PRIMARY KEY,
    value TEXT,
    updated_at TEXT
);

CREATE TABLE IF NOT EXISTS outreach_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    deal_id TEXT,
    timestamp TEXT,
    channel TEXT,                   -- DM / WhatsApp / Email
    message TEXT,
    status TEXT DEFAULT 'drafted'  -- drafted / sent / replied / closed
);
"""

# ─────────────────────────────────────────────────────────────────────────────
# Main store class
# ─────────────────────────────────────────────────────────────────────────────

class DealMemoryStore:
    """
    SQLite-backed memory for the multi-agent pipeline.
    Provides historical context that agents use to reason smarter over time.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = str(db_path)
        self._init_db()

    def _conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._conn() as conn:
            conn.executescript(SCHEMA)
            conn.commit()
        logger.info("DealMemoryStore initialized at %s", self.db_path)

    # ── Deal CRUD ─────────────────────────────────────────────────────────────

    def upsert_deal(self, deal: dict) -> bool:
        """Insert or update a deal record. Returns True if new, False if updated."""
        now = datetime.utcnow().isoformat()
        deal_id = deal.get("id", "")

        # Serialize complex fields
        def _j(val): return json.dumps(val) if isinstance(val, (list, dict)) else val

        with self._conn() as conn:
            existing = conn.execute(
                "SELECT id FROM deals WHERE id = ?", (deal_id,)
            ).fetchone()

            if existing:
                conn.execute("""
                    UPDATE deals SET
                        source=?, title=?, url=?, location=?,
                        asking_price_usd=?, asking_price_inr=?,
                        mrr_usd=?, mrr_inr=?, multiple=?,
                        listing_age_days=?, seller_name=?, seller_handle=?,
                        tags=?, india_flags=?, founder_signals=?, listing_text=?,
                        last_updated=?
                    WHERE id=?
                """, (
                    deal.get("source"), deal.get("title"), deal.get("url"),
                    deal.get("location"), deal.get("asking_price_usd"),
                    deal.get("asking_price_inr"), deal.get("mrr_usd"),
                    deal.get("mrr_inr"), deal.get("multiple"),
                    deal.get("listing_age_days"), deal.get("seller_name"),
                    deal.get("seller_handle"), _j(deal.get("tags", [])),
                    _j(deal.get("india_flags", {})), _j(deal.get("founder_signals", {})),
                    deal.get("listing_text"), now, deal_id
                ))
                conn.commit()
                return False
            else:
                conn.execute("""
                    INSERT INTO deals (
                        id, source, title, url, location,
                        asking_price_usd, asking_price_inr, mrr_usd, mrr_inr,
                        multiple, listing_age_days, seller_name, seller_handle,
                        tags, india_flags, founder_signals, listing_text,
                        status, first_seen, last_updated
                    ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """, (
                    deal_id, deal.get("source"), deal.get("title"), deal.get("url"),
                    deal.get("location"), deal.get("asking_price_usd"),
                    deal.get("asking_price_inr"), deal.get("mrr_usd"),
                    deal.get("mrr_inr"), deal.get("multiple"),
                    deal.get("listing_age_days"), deal.get("seller_name"),
                    deal.get("seller_handle"), _j(deal.get("tags", [])),
                    _j(deal.get("india_flags", {})), _j(deal.get("founder_signals", {})),
                    deal.get("listing_text"), "new", now, now
                ))
                conn.commit()
                return True

    def update_deal_scores(self, deal_id: str, scores: dict):
        """Update a deal's qualification scores from the Qualifier agent."""
        now = datetime.utcnow().isoformat()
        _j = lambda v: json.dumps(v) if isinstance(v, (list, dict)) else v
        with self._conn() as conn:
            conn.execute("""
                UPDATE deals SET
                    motivation_score=?, handover_risk=?, india_risk_score=?,
                    conviction_score=?, intent_label=?, boredom_multiple_likely=?,
                    red_flags=?, green_flags=?, acquisition_thesis=?,
                    deal_verdict=?, last_updated=?
                WHERE id=?
            """, (
                scores.get("motivation_score"),
                scores.get("handover_risk"),
                scores.get("india_risk_score"),
                scores.get("conviction_score"),
                scores.get("intent_label"),
                1 if scores.get("boredom_multiple_likely") else 0,
                _j(scores.get("red_flags", [])),
                _j(scores.get("green_flags", [])),
                scores.get("acquisition_thesis"),
                scores.get("deal_verdict"),
                now, deal_id,
            ))
            conn.commit()

    def update_deal_strategy(self, deal_id: str, strategy: dict):
        """Store Strategist agent outputs for a deal."""
        now = datetime.utcnow().isoformat()
        with self._conn() as conn:
            conn.execute("""
                UPDATE deals SET
                    outreach_draft=?, loi_outline=?, deal_structure_note=?, last_updated=?
                WHERE id=?
            """, (
                strategy.get("outreach_draft"),
                strategy.get("loi_outline"),
                strategy.get("deal_structure_note"),
                now, deal_id,
            ))
            conn.commit()

    def update_deal_status(self, deal_id: str, status: str, notes: str = ""):
        """Manually update deal lifecycle status."""
        with self._conn() as conn:
            conn.execute(
                "UPDATE deals SET status=?, notes=?, last_updated=? WHERE id=?",
                (status, notes, datetime.utcnow().isoformat(), deal_id)
            )
            conn.commit()

    def get_deal(self, deal_id: str) -> Optional[dict]:
        with self._conn() as conn:
            row = conn.execute("SELECT * FROM deals WHERE id=?", (deal_id,)).fetchone()
            return dict(row) if row else None

    def get_all_deals(self, status_filter: Optional[str] = None) -> list[dict]:
        with self._conn() as conn:
            if status_filter:
                rows = conn.execute(
                    "SELECT * FROM deals WHERE status=? ORDER BY last_updated DESC", (status_filter,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM deals ORDER BY last_updated DESC"
                ).fetchall()
            return [dict(r) for r in rows]

    def get_seen_ids(self) -> list[str]:
        """Return IDs of all deals ever seen — used by Sourcer to deduplicate."""
        with self._conn() as conn:
            rows = conn.execute("SELECT id FROM deals").fetchall()
            return [r["id"] for r in rows]

    # ── Agent run logging ─────────────────────────────────────────────────────

    def log_run(self, run_data: dict):
        """Persist a full agent run summary."""
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO agent_runs (
                    run_id, timestamp, mode, sources, deals_sourced, deals_qualified,
                    hot_deals, warm_deals, daily_brief,
                    full_sourcer_output, full_qualifier_output, full_strategist_output,
                    total_runtime_seconds
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """, (
                run_data.get("run_id"),
                run_data.get("timestamp", datetime.utcnow().isoformat()),
                run_data.get("mode", "demo"),
                json.dumps(run_data.get("sources", [])),
                run_data.get("deals_sourced", 0),
                run_data.get("deals_qualified", 0),
                run_data.get("hot_deals", 0),
                run_data.get("warm_deals", 0),
                run_data.get("daily_brief", ""),
                run_data.get("full_sourcer_output", ""),
                run_data.get("full_qualifier_output", ""),
                run_data.get("full_strategist_output", ""),
                run_data.get("total_runtime_seconds", 0),
            ))
            conn.commit()

    def get_recent_runs(self, limit: int = 10) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM agent_runs ORDER BY timestamp DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]

    # ── Context store (key-value institutional memory) ────────────────────────

    def set_context(self, key: str, value: Any):
        """Store a key-value pair in the context store."""
        now = datetime.utcnow().isoformat()
        serialized = json.dumps(value) if not isinstance(value, str) else value
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO context_store (key, value, updated_at)
                VALUES (?, ?, ?)
            """, (key, serialized, now))
            conn.commit()

    def get_context(self, key: str, default: Any = None) -> Any:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT value FROM context_store WHERE key=?", (key,)
            ).fetchone()
            if not row:
                return default
            try:
                return json.loads(row["value"])
            except (json.JSONDecodeError, TypeError):
                return row["value"]

    # ── Rich context string for agents ───────────────────────────────────────

    def build_history_context(self, max_deals: int = 10) -> str:
        """
        Build a rich text summary of past deals for agent consumption.
        This is the core of "context > code" — the agent gets institutional memory.
        """
        deals = self.get_all_deals()
        runs = self.get_recent_runs(limit=5)

        if not deals and not runs:
            return "No historical deals or runs on record. This is your first cycle."

        lines = ["=== INSTITUTIONAL MEMORY ===\n"]

        # Recent runs summary
        if runs:
            lines.append("RECENT AGENT RUNS:")
            for r in runs[:3]:
                lines.append(
                    f"  [{r['timestamp'][:10]}] Mode={r['mode']} | "
                    f"Sourced={r['deals_sourced']} | Qualified={r['deals_qualified']} | "
                    f"Hot={r['hot_deals']} | Warm={r['warm_deals']}"
                )
            lines.append("")

        # Deal history grouped by verdict
        hot = [d for d in deals if d.get("deal_verdict") == "HOT"]
        warm = [d for d in deals if d.get("deal_verdict") == "WARM"]
        passed = [d for d in deals if d.get("deal_verdict") == "PASS"]
        unscored = [d for d in deals if not d.get("deal_verdict")]

        def _deal_line(d):
            inr = d.get("asking_price_inr") or 0
            mrr = d.get("mrr_inr") or 0
            conviction = d.get("conviction_score") or "-"
            status = d.get("status", "new")
            first_seen = (d.get("first_seen") or "")[:10]
            return (
                f"  • [{first_seen}] {d['title'][:60]}... "
                f"| Ask=₹{inr/100000:.1f}L | MRR=₹{mrr/1000:.0f}k "
                f"| Conv={conviction} | Status={status}"
            )

        if hot:
            lines.append(f"HOT DEALS PREVIOUSLY IDENTIFIED ({len(hot)}):")
            for d in hot[:max_deals // 2]:
                lines.append(_deal_line(d))
            lines.append("")

        if warm:
            lines.append(f"WARM DEALS ({len(warm)}):")
            for d in warm[:max_deals // 2]:
                lines.append(_deal_line(d))
            lines.append("")

        if passed:
            lines.append(f"PASSED DEALS ({len(passed)}) — avoid re-surfacing:")
            for d in passed[:3]:
                lines.append(_deal_line(d))
            lines.append("")

        # Repeat listing detector
        all_ids = [d["id"] for d in deals]
        repeat_note = self.get_context("repeat_listings_note", "")
        if repeat_note:
            lines.append(f"REPEAT LISTING PATTERNS: {repeat_note}\n")

        # Market pattern context
        market_note = self.get_context("market_patterns_note", "")
        if market_note:
            lines.append(f"MARKET PATTERNS: {market_note}\n")

        # Valuation comp context
        val_note = self.get_context("valuation_comps_note", "")
        if val_note:
            lines.append(f"VALUATION COMPS: {val_note}\n")

        lines.append(f"TOTAL DEALS IN DB: {len(deals)} (HOT={len(hot)}, WARM={len(warm)}, "
                     f"PASSED={len(passed)}, UNSCORED={len(unscored)})")
        lines.append("=== END INSTITUTIONAL MEMORY ===")

        return "\n".join(lines)

    # ── Outreach log ──────────────────────────────────────────────────────────

    def log_outreach(self, deal_id: str, channel: str, message: str):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO outreach_log (deal_id, timestamp, channel, message)
                VALUES (?, ?, ?, ?)
            """, (deal_id, datetime.utcnow().isoformat(), channel, message))
            conn.commit()

    def get_outreach_log(self, deal_id: Optional[str] = None) -> list[dict]:
        with self._conn() as conn:
            if deal_id:
                rows = conn.execute(
                    "SELECT * FROM outreach_log WHERE deal_id=? ORDER BY timestamp DESC",
                    (deal_id,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM outreach_log ORDER BY timestamp DESC LIMIT 50"
                ).fetchall()
            return [dict(r) for r in rows]

    # ── Stats ─────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with self._conn() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM deals").fetchone()["c"]
            hot = conn.execute("SELECT COUNT(*) as c FROM deals WHERE deal_verdict='HOT'").fetchone()["c"]
            warm = conn.execute("SELECT COUNT(*) as c FROM deals WHERE deal_verdict='WARM'").fetchone()["c"]
            contacted = conn.execute("SELECT COUNT(*) as c FROM deals WHERE status='contacted'").fetchone()["c"]
            runs = conn.execute("SELECT COUNT(*) as c FROM agent_runs").fetchone()["c"]
            total_mrr = conn.execute(
                "SELECT SUM(mrr_inr) as s FROM deals WHERE mrr_inr IS NOT NULL"
            ).fetchone()["s"] or 0

        return {
            "total_deals": total,
            "hot_deals": hot,
            "warm_deals": warm,
            "contacted": contacted,
            "total_runs": runs,
            "total_mrr_inr": total_mrr,
        }
