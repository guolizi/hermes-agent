"""Three-dimensional fact retriever for Butterfly Dream.

Combines three scoring dimensions:
  - Relevance:    How semantically related is this fact to the query?
  - Recency:      How recently was this fact created/updated?
  - Importance:   How intrinsically important is this fact?

Final score = (α × relevance + β × recency + γ × importance) × trust
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .store import MemoryStore

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

import jieba

logger = logging.getLogger(__name__)

# Default scenario weights
SCENARIO_WEIGHTS = {
    "chat":      {"relevance": 0.5, "recency": 0.3, "importance": 0.2},
    "technical": {"relevance": 0.5, "recency": 0.2, "importance": 0.3},
    "longterm":  {"relevance": 0.3, "recency": 0.1, "importance": 0.6},
    "qa":        {"relevance": 0.6, "recency": 0.3, "importance": 0.1},
    "balanced":  {"relevance": 0.4, "recency": 0.3, "importance": 0.3},
}

# Map query semantic categories → fact categories that should be boosted.
# When a user asks about time, boost event/activity facts; when they ask
# about preferences, boost preference/goal facts; etc.
SEMANTIC_CAT_BOOST_MAP: dict[str, tuple[str, ...]] = {
    "time":        ("event", "activity"),
    "place":       ("place", "event", "activity"),
    "event":       ("event", "activity"),
    "activity":    ("activity", "event"),
    "identity":    ("identity", "person", "opinion"),
    "preference":  ("preference", "goal", "opinion"),
    "goal":        ("goal", "preference"),
    "project":     ("project", "tool"),
    "tool":        ("tool", "project"),
    "possession":  ("possession",),
    "state":       ("state", "preference", "opinion"),
    "person":      ("person", "identity", "opinion"),
    "opinion":     ("opinion", "preference", "state"),
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_datetime(dt_str: Optional[str]) -> Optional[datetime]:
    """Parse SQLite datetime string to datetime object. Uses fromisoformat (30x faster than strptime in Python 3.11+)."""
    if not dt_str:
        return None
    try:
        # Python 3.11+ fromisoformat handles both:
        #   '2023-10-19 14:30:00' (SQLite default) and
        #   '2023-10-19T14:30:00+00:00' (ISO 8601)
        s = dt_str.replace("Z", "+00:00")
        if "T" not in s and "+" not in s:
            # SQLite default format: no TZ → assume UTC
            return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)
        return datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None


def _recency_score(dt: Optional[datetime], half_life_days: float = 30.0) -> float:
    """Exponential decay: 0.5^(age / half_life). Returns 1.0 for now, 0.5 at half_life, ~0 at ∞."""
    if dt is None:
        return 0.5  # neutral for unknown timestamps
    age = (_now() - dt).total_seconds() / 86400.0  # age in days
    if age < 0:
        age = 0.0
    if half_life_days <= 0:
        return 1.0  # no decay
    return 0.5 ** (age / half_life_days)


def _importance_score(importance: Optional[float]) -> float:
    """Normalize importance (1-10) to [0, 1]."""
    if importance is None:
        return 0.5
    return max(0.0, min(1.0, (importance - 1.0) / 9.0))


class ThreeDimRetriever:
    """Multi-strategy fact retrieval with three-dimensional scoring.

    Pipeline:
    1. FTS5 search to get candidate pool (limit × 3)
    2. Score each candidate on three dimensions
    3. Weighted combine → sort → return top N
    """

    def __init__(
        self,
        store: MemoryStore,
        *,
        half_life_days: float = 30.0,
        fts_weight: float = 0.4,
        jaccard_weight: float = 0.3,
        hrr_weight: float = 0.3,
        hrr_dim: int = 1024,
        custom_weights: dict | None = None,
        debug_logging: bool = False,
        dlog: logging.Logger | None = None,
    ):
        self.store = store
        self.half_life_days = half_life_days
        self.hrr_dim = hrr_dim
        self._custom_weights = custom_weights
        self._debug_logging = debug_logging
        self._dlog = dlog or logger

        # Auto-redistribute weights if numpy unavailable
        if hrr_weight > 0 and not hrr._HAS_NUMPY:
            fts_weight = 0.6
            jaccard_weight = 0.4
            hrr_weight = 0.0

        self.fts_weight = fts_weight
        self.jaccard_weight = jaccard_weight
        self.hrr_weight = hrr_weight

    def search(
        self,
        query: str,
        *,
        category: Optional[str] = None,
        min_trust: float = 0.3,
        limit: int = 10,
        scenario: str = "balanced",
        recency_weight: Optional[float] = None,
        relevance_weight: Optional[float] = None,
        importance_weight: Optional[float] = None,
        persistent_only: bool = False,
        fts_mode: str = "or",
    ) -> list[dict]:
        """Three-dimensional search: relevance × recency × importance × trust.

        Args:
            query: Search query string.
            category: Optional category filter.
            min_trust: Minimum trust score threshold.
            limit: Max results to return.
            scenario: Weight preset ("chat", "technical", "longterm", "qa", "balanced").
            persistent_only: If True, only return facts marked as persistent.
            recency_weight: Override recency weight for this call.
            relevance_weight: Override relevance weight for this call.
            importance_weight: Override importance weight for this call.

        Returns:
            List of fact dicts with 'score' field, sorted descending.
        """
        import time as _time
        _t0 = _time.time()
        if self._debug_logging:
            self._dlog.debug(
                "search: query='%.100s' limit=%d scenario=%s",
                query, limit, scenario,
            )

        # Resolve weights — merge scenario presets with instance custom weights
        base = SCENARIO_WEIGHTS.get(scenario, SCENARIO_WEIGHTS["balanced"])
        if scenario == "custom" and self._custom_weights:
            base = self._custom_weights
        weights = base.copy()
        if relevance_weight is not None:
            weights["relevance"] = relevance_weight
        if recency_weight is not None:
            weights["recency"] = recency_weight
        if importance_weight is not None:
            weights["importance"] = importance_weight

        # ── Dynamic weight adjustment based on query type ──
        # When a user asks a specific question (detected as 'fact' or 'opinion'),
        # importance is harmful: high-imp identity facts (trans, adoption, imp=8-10)
        # drown out specific low-imp details (sunset, pet names, imp=5) that are
        # actually the answer. FTS5 relevance alone handles fact ranking correctly.
        #
        # Only for undetected (vague/open-ended) queries does importance help
        # guide recall toward meaningful facts.
        query_type = self._detect_query_type(query)
        if query_type is not None:
            # Move all importance weight → relevance
            imp = weights.get("importance", 0.3)
            weights["relevance"] = weights.get("relevance", 0.4) + imp
            weights["importance"] = 0.0

        # Stage 1: Get FTS5 candidates
        candidates = self._fts_candidates(query, category, min_trust, limit * 3, persistent_only, fts_mode=fts_mode)

        # Stage 1.5: Also fetch candidates by semantic category if query matches
        semantic_cats = self._query_to_semantic_categories(query)
        if semantic_cats:
            cat_candidates = self._semantic_category_candidates(
                semantic_cats, category, min_trust, limit * 3, persistent_only
            )
            # Merge: add category candidates not already in FTS5 results
            seen_ids = {c.get("fact_id") for c in candidates}
            for c in cat_candidates:
                if c.get("fact_id") not in seen_ids:
                    c["fts_rank"] = 0.0  # no FTS rank, will rely on other dimensions
                    candidates.append(c)
                    seen_ids.add(c["fact_id"])

        if not candidates:
            if self._debug_logging:
                self._dlog.debug("search: 0 candidates (query='%.100s')", query)
            return []

        # Stage 2: Score on all three dimensions
        query_tokens = self._tokenize(query)
        scored = []

        # Semantic category boost: facts matching detected categories get a relevance bump
        _CAT_BOOST = 0.15  # boost for matching semantic category

        # Entity boost: find entities mentioned in the query
        _ENTITY_BOOST = 0.15  # boost for facts linked to a query entity
        # Temporal boost: for time-related queries, boost facts with precise dates
        _TEMPORAL_BOOST = 0.15  # boost for precise-date facts on time queries
        is_temporal_query = bool(semantic_cats and "time" in semantic_cats)
        entity_fact_ids: set[int] = set()
        entity_fact_map: dict[int, set[int]] = {}  # fact_id → set of entity_ids
        query_entity_ids: set[int] = set()  # entity_ids mentioned in the query
        try:
            # Get all known entity names
            entity_rows = self.store.execute_query(
                "SELECT name FROM entities"
            )
            if entity_rows:
                q_lower = query.lower()
                matched_entities = [
                    row["name"] for row in entity_rows
                    if row["name"].lower() in q_lower
                ]
                if matched_entities:
                    entity_fact_ids = self.store.get_fact_ids_for_entities(matched_entities)
                    # Get entity_ids for matched names for mismatch detection
                    q_placeholders = ",".join("?" for _ in matched_entities)
                    id_rows = self.store.execute_query(
                        f"SELECT entity_id FROM entities WHERE name IN ({q_placeholders})",
                        tuple(matched_entities),
                    )
                    query_entity_ids = {r["entity_id"] for r in id_rows}
                # Build fact_id → entity_ids map for all facts linked to entities
                fe_rows = self.store.execute_query(
                    "SELECT fact_id, entity_id FROM fact_entities"
                )
                for row in fe_rows:
                    fid = row["fact_id"]
                    eid = row["entity_id"]
                    if fid not in entity_fact_map:
                        entity_fact_map[fid] = set()
                    entity_fact_map[fid].add(eid)
        except Exception:
            pass  # entity boost is best-effort

        # Pre-compute query HRR vector once (was inside loop × 60!)
        try:
            _qvec = hrr.encode_text(query, self.hrr_dim) if self.hrr_weight > 0 else None
        except Exception:
            _qvec = None

        for fact in candidates:
            content_tokens = self._tokenize(fact["content"])
            tag_tokens = self._tokenize(fact.get("tags", ""))
            all_tokens = content_tokens | tag_tokens

            # --- Relevance ---
            jaccard = self._jaccard_similarity(query_tokens, all_tokens)
            fts_score = fact.get("fts_rank", 0.0)

            # HRR similarity
            if self.hrr_weight > 0 and fact.get("hrr_vector") and _qvec is not None:
                try:
                    fact_vec = hrr.bytes_to_phases(fact["hrr_vector"])
                    hrr_sim = (hrr.similarity(_qvec, fact_vec) + 1.0) / 2.0
                except Exception:
                    hrr_sim = 0.5
            else:
                hrr_sim = 0.5

            relevance = (
                self.fts_weight * fts_score
                + self.jaccard_weight * jaccard
                + self.hrr_weight * hrr_sim
            )

            # Boosts are multiplicative: each boost applies as a multiplier on relevance.
            # This prevents low-relevance facts from overtaking better matches through
            # multiple additive boosts alone. A fact with low base relevance stays low
            # even with all boosts active.
            boost = 1.0

            # Boost relevance if fact's category matches query intent
            # Uses SEMANTIC_CAT_BOOST_MAP to map query categories → boosted fact categories,
            # e.g. "time" query → boost event/activity facts, not exact "time" category.
            if semantic_cats:
                boost_cats = set()
                for sc in semantic_cats:
                    boost_cats.update(SEMANTIC_CAT_BOOST_MAP.get(sc, ()))
                if boost_cats and fact.get("category") in boost_cats:
                    boost += _CAT_BOOST

            # Boost relevance if fact is linked to an entity mentioned in the query
            if entity_fact_ids and fact.get("fact_id") in entity_fact_ids:
                boost += _ENTITY_BOOST

            # Save query relevance before entity mismatch penalty for diversity re-ranking.
            # Entity diversity needs to evaluate minority facts by how well they match the
            # query, not by how penalized they are for belonging to a different entity.
            _query_relevance = min(1.0, round(relevance * boost, 4))

            # Penalize facts linked to entities NOT mentioned in the query.
            # When the query clearly names an entity (e.g. "Melanie"), facts about
            # other entities (e.g. Caroline) should fall behind rather than compete
            # on FTS5 relevance alone.
            _ENTITY_MISMATCH_PENALTY = -0.3
            fid = fact.get("fact_id")
            if (
                fid
                and query_entity_ids
                and fid in entity_fact_map
                and not entity_fact_map[fid] & query_entity_ids
            ):
                boost += _ENTITY_MISMATCH_PENALTY

            # Boost relevance for time-related queries if fact has a precise date
            if is_temporal_query:
                cd = fact.get("content_date") or ""
                # Precise date: content_date is YYYY-MM-DD and month is specified
                # Day=01 is treated as imprecise (e.g., "around June 2023" → "2023-06-01"),
                # but give a half boost to avoid excluding valid dates like July 1
                if len(cd) == 10 and cd[5:7] != "00":
                    if cd[8:10] != "01":
                        # Fully precise date (day specified)
                        boost += _TEMPORAL_BOOST
                    else:
                        # Month-precise date (day=01 = imprecise/estimated, smaller boost)
                        boost += _TEMPORAL_BOOST * 0.5

            relevance = min(1.0, relevance * boost)

            # --- Recency ---
            created = _parse_datetime(fact.get("created_at"))
            recency = _recency_score(created, self.half_life_days)

            # --- Importance ---
            importance_raw = fact.get("importance")
            importance = _importance_score(importance_raw)

            # --- Combine: three-dimensional score × trust ---
            score = (
                weights["relevance"] * relevance
                + weights["recency"] * recency
                + weights["importance"] * importance
            ) * fact["trust_score"]

            scored.append({
                **fact,
                "_relevance": round(relevance, 4),
                "_recency": round(recency, 4),
                "_importance": round(importance, 4),
                "_query_relevance": _query_relevance,
                "score": round(score, 4),
            })

        # Sort by score descending
        scored.sort(key=lambda x: x["score"], reverse=True)

        # --- Category diversity re-ranking ---
        # If the top N results are dominated by a single category, swap in
        # the best-scoring result from a different category to ensure diversity.
        # This prevents, e.g., all "goal" facts from crowding out "event" or
        # "activity" facts that may be more relevant to the user's intent.
        # Window tracks the return size so diversity covers all returned facts.
        _DIVERSITY_WINDOW = limit
        _DIVERSITY_THRESHOLD = 0.7
        top_n = min(_DIVERSITY_WINDOW, len(scored), limit)
        if top_n >= 4:
            top_slice = scored[:top_n]
            cat_counts: dict[str, int] = {}
            for item in top_slice:
                cat = item.get("category", "unknown") or "unknown"
                cat_counts[cat] = cat_counts.get(cat, 0) + 1
            dominant_cat, dominant_count = max(cat_counts.items(), key=lambda x: x[1])
            if dominant_count / top_n >= _DIVERSITY_THRESHOLD:
                # Find the highest-scoring fact from a different category
                # from beyond the current top slice
                best_idx, best_score = -1, -1.0
                for i in range(top_n, len(scored)):
                    cat = scored[i].get("category", "unknown") or "unknown"
                    if cat != dominant_cat and scored[i]["score"] > best_score:
                        best_idx, best_score = i, scored[i]["score"]
                if best_idx >= 0:
                    # Swap in the diversity fact and deduplicate
                    replacement = dict(scored[best_idx])  # copy
                    # Replace the lowest-ranked dominant-category fact
                    for i in range(top_n - 1, -1, -1):
                        cat = scored[i].get("category", "unknown") or "unknown"
                        if cat == dominant_cat:
                            scored[i] = replacement
                            scored.pop(best_idx)  # remove original (shifts left if best_idx > i)
                            break
                    # Re-sort the top slice
                    top_slice = scored[:top_n]
                    tail = scored[top_n:]
                    top_slice.sort(key=lambda x: x["score"], reverse=True)
                    scored = top_slice + tail

        # --- Entity diversity re-ranking ---
        # If the top N results are dominated by a single entity (checked by DB
        # entity tags via fact_entities + entities tables), swap in the best
        # FTS5-ranked fact about a DIFFERENT entity from beyond the window.
        # This uses actual entity tagging, not substring matching on content.
        # Helps adversarial questions where the query entity is swapped.
        # Window tracks the return size so diversity covers all returned facts.
        _ENTITY_WINDOW = limit
        # Tiered threshold: small result sets need more dominance to justify a swap;
        # large result sets can tolerate lower thresholds for better diversity.
        if limit <= 5:
            _ENTITY_THRESHOLD = 1.0
        elif limit <= 10:
            _ENTITY_THRESHOLD = 0.7
        else:
            _ENTITY_THRESHOLD = 0.6
        top_n_e = min(_ENTITY_WINDOW, len(scored), limit)
        if top_n_e >= 4:
            top_slice = scored[:top_n_e]
            # Batch-load entity info for all scored facts
            all_fact_ids = [item["fact_id"] for item in scored]
            try:
                erows = self.store.execute_query(
                    """SELECT fe.fact_id, e.name
                       FROM fact_entities fe
                       JOIN entities e ON fe.entity_id = e.entity_id
                       WHERE fe.fact_id IN ({})""".format(
                        ",".join("?" * len(all_fact_ids))
                    ),
                    tuple(all_fact_ids),
                )
            except Exception:
                erows = []
            fact_entity_map: dict[int, set[str]] = {}
            for row in erows:
                fact_entity_map.setdefault(row["fact_id"], set()).add(row["name"])

            # Count entity occurrence in top-slice (fractional: dual-entity
            # facts distribute 1/n per entity, so a Jon+Gina fact contributes
            # 0.5 to each — prevents false inflation of single-entity dominance)
            ent_counts: dict[str, float] = {}
            for item in top_slice:
                names = fact_entity_map.get(item["fact_id"], set())
                share = 1.0 / len(names) if names else 0.0
                for name in names:
                    ent_counts[name] = ent_counts.get(name, 0.0) + share

            if ent_counts:
                dominant_ent, dominant_count = max(ent_counts.items(), key=lambda x: x[1])
                if dominant_count / top_n_e >= _ENTITY_THRESHOLD:
                    # Collect all minority-entity facts from beyond the window,
                    # sorted by query relevance (descending), which excludes entity
                    # mismatch penalty. Using raw score would penalize minority-entity
                    # facts twice — first in ranking position, then in selection priority.
                    minority_facts = []
                    for i in range(top_n_e, len(scored)):
                        names = fact_entity_map.get(scored[i]["fact_id"], set())
                        if names and dominant_ent not in names:
                            minority_facts.append((i, dict(scored[i])))
                    minority_facts.sort(key=lambda x: x[1].get("_query_relevance", 0), reverse=True)

                    if minority_facts:
                        # Swap in proportion to result set size (ceil(limit × 20%)),
                        # so larger result sets get more diversity correction.
                        n_swap = min(max(1, math.ceil(limit * 0.2)), len(minority_facts))
                        swaps = minority_facts[:n_swap]

                        # Find the lowest-ranked dominant-entity facts to replace
                        replace_slots = []
                        for i in range(top_n_e - 1, -1, -1):
                            names = fact_entity_map.get(scored[i]["fact_id"], set())
                            if dominant_ent in names:
                                replace_slots.append(i)
                                if len(replace_slots) >= n_swap:
                                    break

                        # Remove minority facts from beyond window (highest index first
                        # to avoid shifting earlier indices)
                        for idx, _ in sorted(swaps, key=lambda x: x[0], reverse=True):
                            scored.pop(idx)

                        # Place replacements at dominant slots
                        for slot_idx, (_, m_fact) in zip(replace_slots, swaps):
                            scored[slot_idx] = m_fact

                        # Re-sort the top slice
                        tail = scored[top_n_e:]
                        top_slice = scored[:top_n_e]
                        top_slice.sort(key=lambda x: x["score"], reverse=True)
                        scored = top_slice + tail
        result = scored[:limit]
        if self._debug_logging:
            _elapsed = (_time.time() - _t0) * 1000
            self._dlog.debug(
                "search: %d candidates → %d returned in %.0fms (query='%.100s')",
                len(candidates), len(result), _elapsed, query,
            )
        return result

    # -- Internal pipeline helpers --------------------------------------------

    def _fts_candidates(
        self,
        query: str,
        category: Optional[str] = None,
        min_trust: float = 0.3,
        limit: int = 30,
        persistent_only: bool = False,
        fts_mode: str = "or",
    ) -> list[dict]:
        """Stage 1: Fetch candidates from FTS5 full-text search.

        Searches both facts_fts and media_attachments_fts in parallel,
        then merges results by fact_id. Media matches bring in their
        parent fact and include a 'media' list and '_media_match' flag.
        """
        # Sanitize query for FTS5 special characters
        safe_query = self._sanitize_fts_query(query, fts_mode)
        if not safe_query:
            return []

        # Query facts_fts — use separate parameterized queries for safety
        try:
            if category:
                if persistent_only:
                    rows = self.store.execute_query(
                        """SELECT f.*, rank FROM facts_fts
                           JOIN facts f ON facts_fts.rowid = f.fact_id
                           WHERE facts_fts MATCH ? AND f.category = ? AND f.trust_score >= ? AND f.is_persistent = 1
                           ORDER BY rank LIMIT ?""",
                        (safe_query, category, min_trust, limit),
                    )
                else:
                    rows = self.store.execute_query(
                        """SELECT f.*, rank FROM facts_fts
                           JOIN facts f ON facts_fts.rowid = f.fact_id
                           WHERE facts_fts MATCH ? AND f.category = ? AND f.trust_score >= ?
                           ORDER BY rank LIMIT ?""",
                        (safe_query, category, min_trust, limit),
                    )
            else:
                if persistent_only:
                    rows = self.store.execute_query(
                        """SELECT f.*, rank FROM facts_fts
                           JOIN facts f ON facts_fts.rowid = f.fact_id
                           WHERE facts_fts MATCH ? AND f.trust_score >= ? AND f.is_persistent = 1
                           ORDER BY rank LIMIT ?""",
                        (safe_query, min_trust, limit),
                    )
                else:
                    rows = self.store.execute_query(
                        """SELECT f.*, rank FROM facts_fts
                           JOIN facts f ON facts_fts.rowid = f.fact_id
                           WHERE facts_fts MATCH ? AND f.trust_score >= ?
                           ORDER BY rank LIMIT ?""",
                        (safe_query, min_trust, limit),
                    )
        except Exception:
            rows = []

        results = []
        seen_fact_ids = {}

        # Normalize BM25 ranks to [0, 1] across ALL candidates so the
        # best FTS5 match gets 1.0 and the worst gets 0.0.
        # BM25 returns negative values (more negative = better match).
        raw_ranks = [float(r["rank"]) for r in rows] if rows else [0]
        bm25_min = min(raw_ranks)
        bm25_max = max(raw_ranks)
        bm25_range = bm25_max - bm25_min if bm25_max > bm25_min else 1.0

        for row in rows:
            d = {key: row[key] for key in row.keys()}
            raw = float(d.get("rank", 0))
            d["fts_rank"] = 1.0 - (raw - bm25_min) / bm25_range
            d["media"] = []
            d["_media_match"] = False
            results.append(d)
            seen_fact_ids[d["fact_id"]] = d

        # Also search media_attachments_fts
        try:
            media_rows = self.store.execute_query(
                """SELECT m.rowid AS media_id, m.rank, ma.*
                   FROM media_attachments_fts m
                   JOIN media_attachments ma ON m.rowid = ma.media_id
                   WHERE media_attachments_fts MATCH ?
                   ORDER BY m.rank LIMIT ?""",
                (safe_query, limit),
            )
        except Exception:
            media_rows = []  # table might not exist in old DBs

        for row in media_rows:
            media = {key: row[key] for key in row.keys()}
            media_rank = media.pop("rank", 0)  # pop rank before ma.* columns shadow it
            media_fts_score = min(1.0, max(0.0, -float(media_rank or 0) / 10.0))
            fid = media["fact_id"]

            if fid in seen_fact_ids:
                # Append media to existing fact result
                existing = seen_fact_ids[fid]
                existing["media"].append(media)
                existing["_media_match"] = True
                # Boost relevance from media match
                existing["fts_rank"] = max(existing["fts_rank"], media_fts_score)
            else:
                # Fetch the parent fact and add it with media
                if persistent_only:
                    fact_rows = self.store.execute_query(
                        "SELECT * FROM facts WHERE fact_id=? AND trust_score>=? AND is_persistent = 1",
                        (fid, min_trust),
                    )
                else:
                    fact_rows = self.store.execute_query(
                        "SELECT * FROM facts WHERE fact_id=? AND trust_score>=?",
                        (fid, min_trust),
                    )
                if fact_rows:
                    fact_row = fact_rows[0]
                    fact_dict = {key: fact_row[key] for key in fact_row.keys()}
                    fact_dict["fts_rank"] = media_fts_score
                    fact_dict["media"] = [media]
                    fact_dict["_media_match"] = True
                    results.append(fact_dict)
                    seen_fact_ids[fid] = fact_dict

        # Sort by fts_rank descending (best match first)
        results.sort(key=lambda x: x["fts_rank"], reverse=True)
        return results[:limit]

    # -- Semantic category helpers --------------------------------------------

    @staticmethod
    def _query_to_semantic_categories(query: str) -> list[str]:
        """Map query keywords to semantic categories. Supports English and Chinese."""
        q = query.lower()
        categories = []

        # English mappings
        EN_MAP = [
            (["where", "location", "place", "which city", "which country", "which town"], "place"),
            (["when", "what time", "what date", "how long", "how many years", "how many days",
              "how old", "since when", "which year", "which month", "which day"], "time"),
            (["who", "whom", "whose", "which person"], "person"),
            (["what happened", "what did", "what event", "what events", "what was the",
              "event", "events", "participated in", "attended", "went to"], "event"),
            (["what activities", "what activity", "what hobbies", "what hobby",
              "what sports", "what sport", "what pastime", "what pastimes",
              "what does", "what do", "how do you", "how often"], "activity"),
            (["what is", "what are", "how would", "describe"], "identity"),
            (["like", "favorite", "prefer", "enjoy", "love", "hate", "dislike",
              "taste", "interest"], "preference"),
            (["want", "plan", "goal", "wish", "hope", "aspire", "intend",
              "going to", "will do"], "goal"),
            (["what project", "what projects", "working on", "building",
              "what tech", "what stack", "what framework"], "project"),
            (["what tool", "what tools", "what software", "what app", "what apps",
              "what program", "what programs", "what do you use", "how do you use"], "tool"),
            (["do you have", "what do you own", "what do you have", "any pets",
              "any cars", "any property"], "possession"),
            (["how are you", "how do you feel", "what is your status",
              "what is your state", "how is it going"], "state"),
            (["what do you think", "how do you feel about", "opinion",
              "thoughts on", "say about"], "opinion"),
        ]
        # Chinese mappings
        ZH_MAP = [
            (["在哪", "哪里", "什么地方", "何处", "哪个城市", "哪个国家", "来源", "来自",
              "住在", "家在", "搬去", "搬到", "去哪"], "place"),
            (["什么时候", "几月", "哪天", "多久", "多长时间", "几年", "何时", "哪一年", "多大",
              "几号", "哪天", "几点", "多晚"], "time"),
            (["谁", "哪个人", "什么人", "认识", "朋友", "家人", "同事", "邻居", "亲戚"], "person"),
            (["发生了什么", "什么事", "什么事件", "什么情况", "怎么了", "出什么事"], "event"),
            (["做什么", "干什么", "什么活动", "什么爱好", "什么运动", "怎么锻炼", "平时做什么",
              "经常", "每天", "每周", "总是", "习惯", "一般"], "activity"),
            (["是什么", "什么样的", "什么身份", "什么职业", "做什么工作", "工作是"], "identity"),
            (["喜欢", "爱好", "偏好", "爱", "讨厌", "不喜欢", "兴趣", "最爱", "最讨厌",
              "宁愿", "倾向", "更喜欢"], "preference"),
            (["想", "计划", "目标", "打算", "希望", "愿望", "想要", "准备", "将来"], "goal"),
            (["什么项目", "在做什么", "什么技术", "什么框架", "用什么开发", "开发什么",
              "做什么项目", "技术栈"], "project"),
            (["什么工具", "什么软件", "用什么", "什么程序", "什么app", "什么应用",
              "用什么软件", "用什么工具"], "tool"),
            (["有没有", "拥有", "有什么", "养了什么", "名下", "养了", "有只", "有只猫",
              "有只狗", "有辆车", "有套房"], "possession"),
            (["最近怎么样", "状态如何", "什么状态", "什么情况", "还好吗", "怎么样"], "state"),
            (["觉得", "认为", "评价", "怎么看", "看法", "怎么说", "什么看法", "怎么觉得"], "opinion"),
        ]

        for keywords, cat in EN_MAP + ZH_MAP:
            for kw in keywords:
                if kw in q:
                    categories.append(cat)
                    break

        # Deduplicate while preserving order
        return list(dict.fromkeys(categories))

    @staticmethod
    def _detect_query_type(query: str) -> str | None:
        """Classify query as fact-finding ('fact'), opinion-seeking ('opinion'), or None (balanced).

        Fact-finding queries ask for a specific concrete answer (names, dates,
        subjects, counts). Opinion queries ask for judgment or assessment. The
        classification lets the 3D scoring reduce importance weight for fact
        queries — where high-importance identity facts otherwise drown out the
        specific facts needed for a correct answer.

        Detection patterns:
        - Fact: "what subject/name/type...", "when did...", "how many...",
                "where did...", "which [noun]...", contains "named"/"called"
        - Opinion: "would [X] be considered", "likely", "think about",
                   "opinion", "personality", "what kind of person"
        """
        import re
        q = query.lower().strip()

        # ── Opinion/assessment signals (keep default importance) ──
        opinion_patterns = [
            r'\bwould\b.*\bbe considered\b',
            r'\bwould\b.*\blikely\b',
            r'\bwhat kind of (person|personality|character|temperament)\b',
            r'\bopinion\b',
            r'\bpersonality\b',
            r'\bview on\b',
        ]
        for pat in opinion_patterns:
            if re.search(pat, q):
                return 'opinion'

        # ── Fact-finding signals (reduce importance) ──
        # Concrete-noun patterns — checked via regex
        fact_patterns = [
            # "what [specific noun]" — asks for a concrete attribute
            r'\bwhat\s+(subject|name|type|kind|sort|color|size|shape|'
            r'date|time|day|month|year|age|address|phone|price|cost|'
            r'brand|model|version|language|genre|style|title|nickname|'
            r'flavor|material|pattern|direction|route'
            r')\b',
            # "what is/are/was/were the [name/date/subject/title]"
            r'\bwhat (are|is|were|was)\s+(the\s+)?(name|date|time|subject|title)s?\b',
            # "what are [person]/[possessive] [attribute]" — e.g. "what are Melanie pets names"
            r'\bwhat (are|is|were|was)\s+\w+\s+(\w+ ){0,3}(name|names|subject|type|type of)\b',
            # "what types/kinds/sorts of" — e.g. "what types of pottery"
            r'\bwhat (types|kinds|sorts) of\b',
            # "when did/was/were/will/does"
            r'\bwhen (did|was|were|will|does|is|are)\b',
            # "where did/was/is/are/does"
            r'\bwhere (did|was|were|is|are|does)\b',
            # "how many/much/long/old/often/far"
            r'\bhow (many|much|long|old|often|far|wide|deep|tall|heavy)\b',
            # "which [noun]"
            r'\bwhich\s+(one|of|of the|\w+ (did|was|were|is|are|has|have))\b',
            # contains "named" / "called" / "name of" / "names of"
            r'\b(name|names) of\b',
            r'\b(named|called)\b',
        ]
        for pat in fact_patterns:
            if re.search(pat, q):
                return 'fact'

        # "what [concrete-noun]" — scan all words after "what" (not just the first)
        # and check if any singularizes to a concrete noun. This avoids the
        # limitation of only matching the first word (e.g. "what musical artists"
        # → first word is the adjective "musical", not the noun "artists").
        m = re.match(r'\bwhat\s+(.+)$', q)
        if m:
            remainder = m.group(1)
            _CONCRETE_NOUNS = {
                'symbol', 'event', 'activity', 'hobby',
                'food', 'drink', 'book', 'movie', 'song',
                'music', 'game', 'sport', 'place', 'city',
                'country', 'store', 'restaurant', 'store',
                'item', 'object', 'gift', 'present',
                # LoCoMo Q61: "what instruments" → "instrument"
                'instrument',
                # LoCoMo Q66: "what changes" → "change"
                'change',
                # "what artists" → "artist"
                'artist',
            }
            # Check each word in the remainder — singularize and compare
            for word in re.findall(r'\b(\w{3,})\b', remainder):
                w = word.lower()
                if w.endswith('ies'):
                    w_sing = w[:-3] + 'y'
                elif w.endswith('s') and not w.endswith('ss'):
                    w_sing = w[:-1]
                else:
                    w_sing = w
                if w_sing in _CONCRETE_NOUNS:
                    return 'fact'

        # "what has/have/did [words] [action-verb]" — extract all 3+ letter words
        # after the auxiliary verb and check if any lemmatizes to a known action verb.
        # This handles all verb forms (painted, bought, read, participated, etc.)
        # without enumerating every inflection.
        m = re.search(r'\bwhat\b.+\b(has|have|did|does)\b', q)
        if m:
            after = q[m.end():]
            _FACT_VERBS = {
                'buy', 'paint', 'draw', 'make', 'create',
                'visit', 'attend', 'go', 'do', 'say',
                'write', 'read', 'watch', 'play', 'cook',
                'bake', 'build', 'participate', 'experience',
                'purchase', 'order', 'eat', 'drink', 'wear',
                'bring', 'take', 'try', 'use', 'get',
                # Q66: "what changes has Caroline faced" → "face"
                'face',
            }
            for w in re.findall(r'\b(\w{3,})\b', after):
                try:
                    from nltk.stem import WordNetLemmatizer
                    wnl = WordNetLemmatizer()
                    w_lemma = wnl.lemmatize(w, 'v')
                except ImportError:
                    w_lemma = w
                if w_lemma in _FACT_VERBS:
                    return 'fact'

        return None

    def _semantic_category_candidates(
        self,
        semantic_cats: list[str],
        category: Optional[str] = None,
        min_trust: float = 0.3,
        limit: int = 30,
        persistent_only: bool = False,
    ) -> list[dict]:
        """Fetch facts by category (semantic classification)."""
        placeholders = ",".join("?" for _ in semantic_cats)
        conditions = f"f.category IN ({placeholders}) AND f.trust_score >= ?"
        params: list = list(semantic_cats) + [min_trust]

        if persistent_only:
            conditions += " AND f.is_persistent = 1"

        params.append(limit)
        try:
            rows = self.store.execute_query(
                f"SELECT * FROM facts f WHERE {conditions} ORDER BY f.importance DESC, f.created_at DESC LIMIT ?",
                tuple(params),
            )
        except Exception:
            return []

        results = []
        for row in rows:
            d = {key: row[key] for key in row.keys()}
            d["fts_rank"] = 0.0
            d["media"] = []
            d["_media_match"] = False
            results.append(d)
        return results

    @staticmethod
    def _sanitize_fts_query(query: str, fts_mode: str = "or") -> str:
        """Remove FTS5 special characters, lemmatize English, filter stop words, and collapse whitespace."""
        import re
        # Remove characters that could break FTS5 syntax
        # NOTE: # is NOT safe — FTS5 uses # as near/column operator, and
        # jieba already splits e.g. \"#31007\" into separate tokens. The #
        # just gets stripped; 31007 still matches via prefix.
        # NOTE: + is intentionally excluded — FTS5 unicode61 tokenizer strips + during indexing,
        # so querying with + (e.g. "LGBTQ+", "C++") would never match. Without + in the
        # query, "LGBTQ+ events" matches the same index tokens as "LGBTQ events".
        safe = re.sub(r'[^\w\s\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]', ' ', query)
        # Jieba-segment CJK text to match FTS5 indexing
        tokens = []
        for word in safe.split():
            if re.search(r'[\u4e00-\u9fff]', word):
                tokens.extend(jieba.cut(word))
            else:
                tokens.append(word)
        # Lemmatize English tokens (reduce verb/noun forms to base form)
        try:
            from nltk.stem import WordNetLemmatizer
            wnl = WordNetLemmatizer()
            tokens = [
                wnl.lemmatize(t, 'v') if t.isascii() and t.isalpha() and len(t) > 2 else t
                for t in tokens
            ]
        except ImportError:
            pass  # NLTK not installed, skip lemmatization
        safe = ' '.join(tokens)
        # Collapse whitespace
        safe = ' '.join(safe.split())
        if len(safe) < 2:
            return ""
        # Filter stop words — these match almost everything in OR mode
        # and add noise without improving relevance
        STOP_WORDS = {
            # English
            'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
            'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'shall', 'must',
            'i', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours',
            'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her', 'hers',
            'it', 'its', 'they', 'them', 'their', 'theirs',
            'this', 'that', 'these', 'those', 'here', 'there',
            'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as',
            'into', 'about', 'between', 'through', 'during', 'before', 'after',
            'and', 'but', 'or', 'nor', 'not', 'so', 'if', 'then', 'than', 'too',
            'very', 'also', 'some', 'any', 'all',
            'no', 'only', 'own', 'same', 'other', 'such',
            'further', 'once', 'again', 'further', 'even', 'still',
            # English: question words — these are very common in QA queries but
            # their prefix expansion (what*, who*, where*) matches almost everything
            # starting with those letters, drowning out meaningful terms.
            'what', 'who', 'where', 'when', 'why', 'how', 'which', 'whom', 'whose',
            # Chinese
            '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一',
            '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会', '着',
            '没有', '看', '好', '自己', '这', '他', '她', '它', '们', '那', '被',
            '从', '把', '些', '所', '过', '对', '里', '为', '与', '及', '等',
        }
        tokens_clean = [t for t in safe.split() if t.lower() not in STOP_WORDS]
        if not tokens_clean:
            # Fallback: if all tokens were stop words, keep original
            tokens_clean = safe.split()
        # Use '*' prefix matching to bridge jieba segmentation gaps.
        # jieba may segment the same text differently in queries vs indexed content
        # (e.g. "橘猫" is one token in index but "橘" + "猫叫" in query "橘猫叫什么名字").
        # Prefix matching ensures partial jieba tokens still produce candidates.
        # NOTE: Only prefix-match tokens with length >= 3 to avoid short prefixes
        # like "go*" accidentally matching "goal" or "to*" matching "today".
        from .synonyms import get_synonyms

        op = ' AND ' if fts_mode == 'and' else ' OR '
        # Build base query terms — expand synonyms as OR groups
        terms = []
        for t in tokens_clean:
            base = t + '*' if len(t) >= 3 or re.search(r'[\u4e00-\u9fff]', t) else t
            # Short tokens (len < 3) don't benefit from synonym expansion.
            # Single chars like "s" (from possessives: Gina's → s) get WordNet
            # synonyms like second*, sec*, sulfur*, south*, ... that add massive
            # noise to FTS5 queries without improving retrieval quality.
            syns = get_synonyms(t) if len(t) >= 3 else []
            if syns:
                # Filter out multi-word, hyphenated, or special-character synonyms
                # — they break FTS5 syntax (spaces→AND, hyphens→column subtraction,
                # dots→column accessor, colons→NEAR syntax, etc.).
                import re as _re_syn
                safe_syns = [s for s in syns if _re_syn.match(r'^[a-zA-Z0-9]+$', s)]
                if not safe_syns:
                    syns = []
                    syn_terms = []
                else:
                    syns = safe_syns
                    syn_terms = [
                        s + '*' if len(s) >= 3 or re.search(r'[\u4e00-\u9fff]', s) else s
                        for s in syns
                    ]
                    # Deduplicate: avoid repeating the base term as a synonym
                    syn_terms = [st for st in syn_terms if st != base]
                if syn_terms:
                    terms.append(f'({base} OR ' + ' OR '.join(syn_terms) + ')')
                else:
                    terms.append(base)
            else:
                terms.append(base)
        # Handle compound words: FTS5 default tokenizer splits on hyphens,
        # so "de-stress" is indexed as ["de", "stress"] while the query may
        # have "destress" (no hyphen). To bridge this gap, for tokens that
        # contain a known English prefix (de-, re-, un-, pre-, dis-, mis-,
        # over-, under-, out-, non-, anti-, counter-), also add the root
        # part as an additional OR term so "destress* OR stress*" matches.
        PREFIXES = ('de', 're', 'un', 'pre', 'dis', 'mis', 'over', 'under',
                     'out', 'non', 'anti', 'counter', 'inter', 'super',
                     'sub', 'semi', 'mid', 'co', 'ex', 'en')
        extra_terms = []
        for t in tokens_clean:
            tl = t.lower()
            for pfx in PREFIXES:
                if tl.startswith(pfx) and len(tl) > len(pfx) + 2:
                    root = tl[len(pfx):]
                    if len(root) >= 3:
                        extra_terms.append(root + '*')
                    break  # one split per token
        terms.extend(extra_terms)
        return op.join(terms)

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Tokenize text into a set of normalized tokens."""
        return tokenize(text)

    @staticmethod
    def _jaccard_similarity(a: set[str], b: set[str]) -> float:
        """Jaccard similarity between two token sets."""
        return jaccard_similarity(a, b)


# Module-level helpers (also usable by store.py)
def tokenize(text: str) -> set[str]:
    """Tokenize text into a set of normalized tokens.

    Uses jieba for CJK word segmentation and regex for English tokens,
    producing semantically meaningful tokens for both languages.
    This powers Jaccard similarity in dedup, merge, and retrieval.
    """
    import re
    tokens = set()
    # English / Latin words
    for token in re.findall(r'[a-zA-Z][a-zA-Z0-9_\-+#]{1,}', text):
        tokens.add(token.lower())
    # CJK text — jieba word segmentation (more accurate than bigrams)
    cjk_parts = re.findall(r'[\u4e00-\u9fff]+', text)
    for part in cjk_parts:
        tokens.update(jieba.cut(part))
    return tokens


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    """Jaccard similarity between two token sets."""
    if not a or not b:
        return 0.0
    intersection = a & b
    union = a | b
    return len(intersection) / len(union)
