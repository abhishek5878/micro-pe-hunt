"""
Persistent Vector Memory Engine.
Uses FAISS (via LangChain) + HuggingFace sentence-transformers.
Falls back to TF-IDF cosine similarity when embeddings unavailable.
Seeded with 5 demo deals on first run so memory works immediately.
"""

import json
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

KNOWLEDGE_PATH = Path(__file__).parent.parent / "data" / "sme_knowledge.json"
FAISS_STORE_PATH = Path(__file__).parent.parent / "faiss_store"
DEAL_LOG_PATH = Path(__file__).parent.parent / "faiss_store" / "deal_log.json"
TFIDF_STORE_PATH = Path(__file__).parent.parent / "faiss_store" / "tfidf_store.pkl"


def _load_knowledge() -> dict:
    with open(KNOWLEDGE_PATH) as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF Fallback Store (works with no external APIs or downloads)
# ─────────────────────────────────────────────────────────────────────────────

class TFIDFFallbackStore:
    """
    Pure scikit-learn TF-IDF similarity store.
    Instant — no downloads, no API keys, no GPU.
    """

    def __init__(self):
        self.texts: list[str] = []
        self.metadatas: list[dict] = []
        self._vectorizer = None
        self._matrix = None

    def _rebuild(self):
        if not self.texts:
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(stop_words="english", max_features=2000)
        self._matrix = self._vectorizer.fit_transform(self.texts)

    def add(self, text: str, metadata: dict):
        self.texts.append(text)
        self.metadatas.append(metadata)
        self._rebuild()

    def search(self, query: str, k: int = 3) -> list[dict]:
        if not self.texts or self._matrix is None or self._vectorizer is None:
            return []
        from sklearn.metrics.pairwise import cosine_similarity
        q_vec = self._vectorizer.transform([query])
        scores = cosine_similarity(q_vec, self._matrix).flatten()
        top_k = np.argsort(scores)[::-1][:k]
        results = []
        for idx in top_k:
            if scores[idx] > 0.05:
                results.append({
                    "content": self.texts[idx],
                    "metadata": self.metadatas[idx],
                    "score": float(scores[idx]),
                })
        return results

    def save(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "tfidf_store.pkl", "wb") as f:
            pickle.dump({"texts": self.texts, "metadatas": self.metadatas}, f)

    @classmethod
    def load(cls, path: Path) -> "TFIDFFallbackStore":
        store = cls()
        pkl_path = path / "tfidf_store.pkl"
        if pkl_path.exists():
            with open(pkl_path, "rb") as f:
                data = pickle.load(f)
            store.texts = data["texts"]
            store.metadatas = data["metadatas"]
            store._rebuild()
        return store


# ─────────────────────────────────────────────────────────────────────────────
# FAISS Store (preferred — requires sentence-transformers)
# ─────────────────────────────────────────────────────────────────────────────

class FAISSMemoryStore:
    """FAISS-backed vector store via LangChain Community."""

    def __init__(self, index_path: Path = FAISS_STORE_PATH, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.index_path = str(index_path)
        self.embedding_model = embedding_model
        self._embedder = None
        self._store = None

    def _get_embedder(self):
        if self._embedder is None:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embedder = HuggingFaceEmbeddings(
                model_name=self.embedding_model,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        return self._embedder

    def _load_or_create(self, texts: list[str] = None, metadatas: list[dict] = None):
        from langchain_community.vectorstores import FAISS
        from langchain.docstore.document import Document

        index_file = os.path.join(self.index_path, "index.faiss")
        if os.path.exists(index_file) and not texts:
            self._store = FAISS.load_local(
                self.index_path,
                self._get_embedder(),
                allow_dangerous_deserialization=True,
            )
        elif texts:
            docs = [Document(page_content=t, metadata=m or {}) for t, m in zip(texts, metadatas or [{}] * len(texts))]
            if self._store:
                self._store.add_documents(docs)
            else:
                self._store = FAISS.from_documents(docs, self._get_embedder())
            os.makedirs(self.index_path, exist_ok=True)
            self._store.save_local(self.index_path)

    def add(self, text: str, metadata: dict):
        self._load_or_create([text], [metadata])

    def search(self, query: str, k: int = 3) -> list[dict]:
        if self._store is None:
            try:
                self._load_or_create()
            except Exception:
                return []
        if self._store is None:
            return []
        docs_scores = self._store.similarity_search_with_score(query, k=k)
        return [
            {"content": d.page_content, "metadata": d.metadata, "score": float(s)}
            for d, s in docs_scores
        ]


# ─────────────────────────────────────────────────────────────────────────────
# Unified DealMemory — picks FAISS or TF-IDF automatically
# ─────────────────────────────────────────────────────────────────────────────

class DealMemory:
    """
    Unified memory interface. Tries FAISS with HuggingFace embeddings first.
    Falls back silently to TF-IDF for offline / no-download environments.

    All deals stored in:
    - Vector store (FAISS or TF-IDF) for semantic retrieval
    - deal_log.json for full metadata + narrative storage
    """

    def __init__(self):
        FAISS_STORE_PATH.mkdir(parents=True, exist_ok=True)
        self._backend: str = "none"
        self._faiss: Optional[FAISSMemoryStore] = None
        self._tfidf: Optional[TFIDFFallbackStore] = None
        self._deals: list[dict] = self._load_deal_log()
        self._init_backend()

    def _init_backend(self):
        # Primary: TF-IDF (reliable, no downloads, no GPU, works on all Python versions)
        # FAISS/sentence-transformers is optional and only tried if explicitly configured
        try:
            self._tfidf = TFIDFFallbackStore.load(FAISS_STORE_PATH)
            self._backend = "tfidf"
            logger.info("Memory backend: TF-IDF (scikit-learn)")
            return
        except Exception as e:
            logger.warning("TF-IDF init failed: %s", e)

        # Last resort: in-memory only
        self._tfidf = TFIDFFallbackStore()
        self._backend = "tfidf"
        logger.info("Memory backend: TF-IDF (in-memory only)")

    def _load_deal_log(self) -> list[dict]:
        if DEAL_LOG_PATH.exists():
            with open(DEAL_LOG_PATH) as f:
                return json.load(f)
        return []

    def _save_deal_log(self):
        DEAL_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(DEAL_LOG_PATH, "w") as f:
            json.dump(self._deals, f, indent=2)

    def _save_tfidf(self):
        if self._tfidf:
            self._tfidf.save(FAISS_STORE_PATH)

    def is_seeded(self) -> bool:
        return len(self._deals) > 0

    # ── Core Operations ───────────────────────────────────────────────────────

    def add_deal(self, deal: dict) -> str:
        """
        Add a deal to memory. Returns deal_id.
        deal must have: title, summary, sector, state, verdict, conviction_score, etc.
        """
        deal_id = deal.get("id", f"deal-{datetime.utcnow().strftime('%Y%m%d%H%M%S')}")
        deal["id"] = deal_id
        deal["added_at"] = datetime.utcnow().isoformat()

        # Build rich text for embedding
        text = self._build_embed_text(deal)

        # Store in vector backend
        metadata = {k: v for k, v in deal.items() if k != "summary"}
        if self._backend == "faiss" and self._faiss:
            try:
                self._faiss.add(text, metadata)
            except Exception as e:
                logger.error("FAISS add failed: %s — falling back to TF-IDF", e)
                self._backend = "tfidf"
                self._tfidf = TFIDFFallbackStore()
                for d in self._deals:
                    self._tfidf.add(self._build_embed_text(d), d)
                self._tfidf.add(text, metadata)
                self._save_tfidf()

        if self._backend == "tfidf" and self._tfidf:
            self._tfidf.add(text, metadata)
            self._save_tfidf()

        # Store in deal log
        if not any(d.get("id") == deal_id for d in self._deals):
            self._deals.append(deal)
            self._save_deal_log()

        return deal_id

    def find_similar(self, query: str, k: int = 3) -> list[dict]:
        """
        Find semantically similar past deals.
        Returns list of dicts with: content, metadata, score, memory_note
        """
        raw_results = []

        if self._backend == "faiss" and self._faiss:
            try:
                raw_results = self._faiss.search(query, k)
            except Exception:
                pass

        if not raw_results and self._backend in ("tfidf", "none") and self._tfidf:
            raw_results = self._tfidf.search(query, k)

        # Enrich results with memory notes
        enriched = []
        for r in raw_results:
            meta = r.get("metadata", {})
            deal_id = meta.get("id", "?")
            full_deal = next((d for d in self._deals if d.get("id") == deal_id), meta)
            note = self._generate_memory_note(full_deal, r.get("score", 0))
            enriched.append({**r, "memory_note": note, "full_deal": full_deal})

        return enriched

    def get_all_deals(self) -> list[dict]:
        return self._deals

    def get_deal_by_id(self, deal_id: str) -> Optional[dict]:
        return next((d for d in self._deals if d.get("id") == deal_id), None)

    def get_stats(self) -> dict:
        total = len(self._deals)
        hot = sum(1 for d in self._deals if d.get("verdict") == "HOT")
        warm = sum(1 for d in self._deals if d.get("verdict") == "WARM")
        passed = sum(1 for d in self._deals if d.get("verdict") == "PASS")
        return {
            "total": total, "hot": hot, "warm": warm, "passed": passed,
            "backend": self._backend,
        }

    # ── Memory Note Generator ─────────────────────────────────────────────────

    def _generate_memory_note(self, deal: dict, similarity_score: float) -> str:
        """Generate a human-readable memory comparison note for an agent."""
        title = deal.get("title", "Unknown Deal")
        verdict = deal.get("verdict", "?")
        state = deal.get("state", "?")
        sector = deal.get("sector", "?")
        conviction = deal.get("conviction_score", "?")
        outcome = deal.get("deal_outcome", "No update")
        motivation = deal.get("motivation", "?")
        sim_pct = int(similarity_score * 100)

        return (
            f"[MEMORY ~{sim_pct}% match] Similar to: '{title}' ({state}, {sector}) | "
            f"Verdict was {verdict} | Conviction {conviction}/10 | "
            f"Motivation: {motivation} | Outcome: {outcome}"
        )

    # ── Embed Text Builder ────────────────────────────────────────────────────

    @staticmethod
    def _build_embed_text(deal: dict) -> str:
        """Build a rich text representation for embedding."""
        parts = [
            deal.get("title", ""),
            deal.get("summary", ""),
            f"Sector: {deal.get('sector', '')}",
            f"State: {deal.get('state', '')}",
            f"Motivation: {deal.get('motivation', '')}",
            f"Verdict: {deal.get('verdict', '')}",
            f"Revenue: {deal.get('revenue_cr', '')} Cr",
            f"Conviction: {deal.get('conviction_score', '')}",
            f"India Risk: {deal.get('india_risk', '')}",
        ]
        return " | ".join(p for p in parts if p and p.strip())

    # ── Context String for Agents ─────────────────────────────────────────────

    def build_agent_context(self) -> str:
        """Build a concise institutional memory summary for LLM agent prompts."""
        if not self._deals:
            return "No deals in memory yet. This is the first analysis run."

        hot = [d for d in self._deals if d.get("verdict") == "HOT"]
        warm = [d for d in self._deals if d.get("verdict") == "WARM"]
        passed = [d for d in self._deals if d.get("verdict") == "PASS"]

        lines = [
            f"=== DEAL MEMORY ({len(self._deals)} deals: {len(hot)} HOT, {len(warm)} WARM, {len(passed)} PASSED) ===",
        ]

        for d in self._deals[-8:]:  # last 8 for context window efficiency
            rev = d.get("revenue_cr", "?")
            ebitda = d.get("ebitda_l", "?")
            lines.append(
                f"• [{d.get('verdict','?')}] {d.get('title','?')} | "
                f"Rev=₹{rev}Cr | EBITDA=₹{ebitda}L | "
                f"Conviction={d.get('conviction_score','?')} | "
                f"Outcome: {d.get('deal_outcome','?')}"
            )

        lines.append("=== END MEMORY ===")
        return "\n".join(lines)

    # ── Seed Demo Data ────────────────────────────────────────────────────────

    def seed_demo_data(self, force: bool = False):
        """
        Pre-populate memory with 5 seed deals from sme_knowledge.json.
        Called on first startup — makes memory demo instantly visible.
        """
        if self.is_seeded() and not force:
            return

        kb = _load_knowledge()
        seed_deals = kb.get("demo_seed_deals", [])

        for deal in seed_deals:
            # Don't re-add if already in log
            if not any(d.get("id") == deal["id"] for d in self._deals):
                self.add_deal(deal)

        logger.info("Seeded %d demo deals into memory (backend=%s)", len(seed_deals), self._backend)
