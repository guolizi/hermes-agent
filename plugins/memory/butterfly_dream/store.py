"""SQLite-backed fact store with three-dimensional scoring metadata.

Extends the Holographic store with:
- Importance scoring (1-10, LLM-assigned + user-adjustable)
- Recency tracking (automatic timestamps with configurable decay)
- Entity relationship graph for multi-hop reasoning
- Fact merging / conflict resolution for same-entity facts
"""

import copy
import hashlib
import logging
import os
import re
import shutil
import sqlite3
import threading
from pathlib import Path
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import jieba

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

from .retrieval import tokenize, jaccard_similarity
from nltk.stem import WordNetLemmatizer

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    fact_id            INTEGER PRIMARY KEY AUTOINCREMENT,
    content            TEXT NOT NULL UNIQUE,
    category           TEXT DEFAULT 'general',  -- place/time/person/event/activity/identity/preference/goal/project/tool/possession/state/general
    tags               TEXT DEFAULT '',
    importance      REAL DEFAULT 5.0,          -- 1.0 ~ 10.0, LLM-assigned
    trust_score     REAL DEFAULT 0.5,
    retrieval_count INTEGER DEFAULT 0,
    helpful_count   INTEGER DEFAULT 0,
    is_persistent   INTEGER DEFAULT 0,         -- 1 = long-lived fact, survives prefetch filtering
    content_date    TEXT,                       -- event date from conversation (e.g. '2023-01-19')
    created_at      TEXT DEFAULT (datetime('now')),
    updated_at      TEXT DEFAULT (datetime('now')),
    hrr_vector      BLOB
);

CREATE TABLE IF NOT EXISTS entities (
    entity_id   INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT NOT NULL UNIQUE,
    entity_type TEXT DEFAULT 'unknown',
    aliases     TEXT DEFAULT '',
    created_at  TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS fact_entities (
    fact_id   INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    entity_id INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    PRIMARY KEY (fact_id, entity_id)
);

CREATE TABLE IF NOT EXISTS entity_relations (
    relation_id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_id   INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    target_id   INTEGER NOT NULL REFERENCES entities(entity_id) ON DELETE CASCADE,
    relation    TEXT DEFAULT 'related_to',
    weight      REAL DEFAULT 1.0,
    created_at  TEXT DEFAULT (datetime('now')),
    UNIQUE(source_id, target_id, relation)
);

CREATE TABLE IF NOT EXISTS merge_log (
    merge_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    kept_fact_id   INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    absorbed_fact_id INTEGER REFERENCES facts(fact_id) ON DELETE CASCADE,
    merged_content TEXT,
    merge_reason  TEXT DEFAULT 'auto',
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS media_attachments (
    media_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_id       INTEGER NOT NULL REFERENCES facts(fact_id) ON DELETE CASCADE,
    storage_type  TEXT NOT NULL DEFAULT 'file' CHECK(storage_type IN ('file', 'url')),
    file_path     TEXT NOT NULL,
    mime_type     TEXT NOT NULL,
    file_size     INTEGER NOT NULL DEFAULT 0,
    sha256        TEXT NOT NULL DEFAULT '',
    description   TEXT NOT NULL DEFAULT '',
    caption       TEXT DEFAULT '',
    transcript    TEXT DEFAULT '',
    created_at    TEXT DEFAULT (datetime('now'))
);

CREATE VIRTUAL TABLE IF NOT EXISTS media_attachments_fts
    USING fts5(description, caption, transcript, content=media_attachments, content_rowid=media_id);

CREATE INDEX IF NOT EXISTS idx_media_fact    ON media_attachments(fact_id);
CREATE INDEX IF NOT EXISTS idx_media_sha256  ON media_attachments(sha256) WHERE sha256 != '';
CREATE INDEX IF NOT EXISTS idx_media_path    ON media_attachments(file_path);
CREATE INDEX IF NOT EXISTS idx_media_mime    ON media_attachments(mime_type);
CREATE INDEX IF NOT EXISTS idx_media_created ON media_attachments(created_at DESC);

-- FTS5 sync triggers
CREATE TRIGGER IF NOT EXISTS media_ai AFTER INSERT ON media_attachments BEGIN
    INSERT INTO media_attachments_fts(rowid, description, caption, transcript)
        VALUES (new.media_id, new.description, new.caption, new.transcript);
END;

CREATE TRIGGER IF NOT EXISTS media_ad AFTER DELETE ON media_attachments BEGIN
    INSERT INTO media_attachments_fts(media_attachments_fts, rowid, description, caption, transcript)
        VALUES ('delete', old.media_id, old.description, old.caption, old.transcript);
END;

CREATE TRIGGER IF NOT EXISTS media_au AFTER UPDATE ON media_attachments BEGIN
    INSERT INTO media_attachments_fts(media_attachments_fts, rowid, description, caption, transcript)
        VALUES ('delete', old.media_id, old.description, old.caption, old.transcript);
    INSERT INTO media_attachments_fts(rowid, description, caption, transcript)
        VALUES (new.media_id, new.description, new.caption, new.transcript);
END;

CREATE INDEX IF NOT EXISTS idx_facts_trust       ON facts(trust_score DESC);
CREATE INDEX IF NOT EXISTS idx_facts_importance  ON facts(importance DESC);
CREATE INDEX IF NOT EXISTS idx_facts_created     ON facts(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_facts_content_date ON facts(content_date);
CREATE INDEX IF NOT EXISTS idx_facts_category    ON facts(category);
CREATE INDEX IF NOT EXISTS idx_entities_name   ON entities(name);

CREATE VIRTUAL TABLE IF NOT EXISTS facts_fts
    USING fts5(content, tags, content=facts, content_rowid=fact_id);

CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
END;

CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
    INSERT INTO facts_fts(facts_fts, rowid, content, tags)
        VALUES ('delete', old.fact_id, old.content, old.tags);
    INSERT INTO facts_fts(rowid, content, tags)
        VALUES (new.fact_id, new.content, new.tags);
END;
"""

# Trust adjustment constants
_HELPFUL_DELTA   = 0.05
_UNHELPFUL_DELTA = -0.10
_IMPORTANCE_DELTA = 0.5  # Per feedback on importance
_TRUST_MIN = 0.0
_TRUST_MAX = 1.0
_IMPORTANCE_MIN = 1.0
_IMPORTANCE_MAX = 10.0

_MEDIA_TYPE_DIR = {
    "image/": "im",
    "audio/": "au",
    "video/": "vi",
}

# MIME type → file extension mapping (common aliases)
_EXT_MAP = {
    "jpeg": "jpg",
    "mpeg": "mp3",
    "quicktime": "mov",
    "x-msvideo": "avi",
    "x-matroska": "mkv",
    "webm": "webm",
    "ogg": "ogg",
    "3gpp": "3gp",
    "mp2t": "ts",
    "x-ms-wmv": "wmv",
    "x-flv": "flv",
    "octet-stream": "bin",
}


def _media_type_prefix(mime_type: str) -> str:
    """Map mime type prefix to short directory code."""
    for prefix, code in _MEDIA_TYPE_DIR.items():
        if mime_type.startswith(prefix):
            return code
    return "ot"  # other


# Entity extraction patterns (from Holographic, enhanced for CJK)
_RE_CAPITALIZED  = re.compile(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b')  # multi-word names
_RE_SINGLE_NAME  = re.compile(r'\b([A-Z][a-z]{2,15})\s+(?:is|was|has|had|went|likes|loves|works|lives|said|told|got|made|ran|came|wants|plans|loves|hates|enjoys|prefers|started|attended|joined|created|bought|found|helped|met|played|read|wrote|sang|painted|cooked|drove|flew|swam|ran)\b')  # single name + verb → entity
_RE_DOUBLE_QUOTE = re.compile(r'"([^"]+)"')
_RE_SINGLE_QUOTE = re.compile(r"'([^']+)'")
_RE_AKA = re.compile(
    r'(\w+(?:\s+\w+)*)\s+(?:aka|also known as)\s+(\w+(?:\s+\w+)*)',
    re.IGNORECASE,
)
_RE_CJK_BRACKETS = re.compile(
    r"[「『]([^」』]+)[」』]|《([^》]+)》|\u201c([^\u201d]+)\u201d|\u2018([^\u2019]+)\u2019"
)
_RE_QUOTED_CN = re.compile(r'["\u201c\u2018]([^"\u201d\u2019]{2,})["\u201d\u2019]')


def _jieba_segment(text: str) -> str:
    """Pre-segment text with jieba for FTS5 indexing.

    Inserts spaces between Chinese words so FTS5's unicode61 tokenizer
    can properly tokenize them. English/Latin text passes through unchanged.

    Example:
        "我喜欢猫咪love cats" -> "我 喜欢 猫咪 love cats"
    """
    if not text or not isinstance(text, str):
        return text or ""
    parts = []
    for word in re.split(r'(\s+)', text):
        if not word.strip():
            parts.append(word)
            continue
        if re.search(r'[\u4e00-\u9fff]', word):
            parts.append(" ".join(jieba.cut(word)))
        else:
            parts.append(word)
    return "".join(parts)


class MemoryStore:
    """Thread-safe SQLite-backed fact store with importance + trust tracking."""

    def __init__(self, db_path: str, default_trust: float = 0.5, hrr_dim: int = 1024,
                 compression_config: Optional[dict] = None):
        self._db_path = db_path
        self._default_trust = default_trust
        self._hrr_dim = hrr_dim
        self._media_dir = str(Path(db_path).parent / "media")
        self._compression_config = compression_config
        self._lock = threading.Lock()

        # Ensure directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        # Enable foreign keys so ON DELETE CASCADE works
        self._conn.execute("PRAGMA foreign_keys = ON")
        # WAL mode allows concurrent reads during async writes (extraction threads
        # may write while prefetch reads). busy_timeout prevents SQLITE_BUSY errors.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=5000")

        # Migrate BEFORE schema: existing DBs may lack columns that indexes reference.
        # CREATE TABLE IF NOT EXISTS is a no-op for existing tables, so indexes on
        # new columns would crash if the ALTER TABLE hasn't run yet.
        try:
            self._conn.execute("ALTER TABLE facts ADD COLUMN is_persistent INTEGER DEFAULT 0")
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

        try:
            self._conn.execute("ALTER TABLE facts ADD COLUMN content_date TEXT")
            self._conn.commit()
        except sqlite3.OperationalError:
            pass  # Column already exists

        self._conn.executescript(_SCHEMA)
        self._conn.commit()

        # register jieba_segment SQLite function for FTS5 CJK tokenization
        self._conn.create_function("jieba_segment", 1, _jieba_segment)

        # Rebuild FTS5 triggers to use jieba_segment for CJK word segmentation.
        # Without this, Chinese text stored as e.g. "我喜欢猫咪" gets indexed by
        # FTS5's unicode61 tokenizer as a single token, making it unsearchable.
        # jieba_segment inserts spaces: "我 喜欢 猫咪" -> each word is an FTS5 token.
        self._conn.executescript("""
            DROP TRIGGER IF EXISTS facts_ai;
            CREATE TRIGGER IF NOT EXISTS facts_ai AFTER INSERT ON facts BEGIN
                INSERT INTO facts_fts(rowid, content, tags)
                    VALUES (new.fact_id, jieba_segment(new.content), jieba_segment(new.tags));
            END;

            DROP TRIGGER IF EXISTS facts_ad;
            CREATE TRIGGER IF NOT EXISTS facts_ad AFTER DELETE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, content, tags)
                    VALUES ('delete', old.fact_id, old.content, old.tags);
            END;

            DROP TRIGGER IF EXISTS facts_au;
            CREATE TRIGGER IF NOT EXISTS facts_au AFTER UPDATE ON facts BEGIN
                INSERT INTO facts_fts(facts_fts, rowid, content, tags)
                    VALUES ('delete', old.fact_id, old.content, old.tags);
                INSERT INTO facts_fts(rowid, content, tags)
                    VALUES (new.fact_id, jieba_segment(new.content), jieba_segment(new.tags));
            END;

            DROP TRIGGER IF EXISTS media_ai;
            CREATE TRIGGER IF NOT EXISTS media_ai AFTER INSERT ON media_attachments BEGIN
                INSERT INTO media_attachments_fts(rowid, description, caption, transcript)
                    VALUES (new.media_id, jieba_segment(new.description), jieba_segment(new.caption), jieba_segment(new.transcript));
            END;

            DROP TRIGGER IF EXISTS media_ad;
            CREATE TRIGGER IF NOT EXISTS media_ad AFTER DELETE ON media_attachments BEGIN
                INSERT INTO media_attachments_fts(media_attachments_fts, rowid, description, caption, transcript)
                    VALUES ('delete', old.media_id, old.description, old.caption, old.transcript);
            END;

            DROP TRIGGER IF EXISTS media_au;
            CREATE TRIGGER IF NOT EXISTS media_au AFTER UPDATE ON media_attachments BEGIN
                INSERT INTO media_attachments_fts(media_attachments_fts, rowid, description, caption, transcript)
                    VALUES ('delete', old.media_id, old.description, old.caption, old.transcript);
                INSERT INTO media_attachments_fts(rowid, description, caption, transcript)
                    VALUES (new.media_id, jieba_segment(new.description), jieba_segment(new.caption), jieba_segment(new.transcript));
            END;

            -- Rebuild FTS5 index for existing data with jieba segmentation
            DELETE FROM facts_fts;
            INSERT INTO facts_fts(rowid, content, tags)
                SELECT fact_id, jieba_segment(content), jieba_segment(tags) FROM facts;

            DELETE FROM media_attachments_fts;
            INSERT INTO media_attachments_fts(rowid, description, caption, transcript)
                SELECT media_id, jieba_segment(description), jieba_segment(caption), jieba_segment(transcript)
                FROM media_attachments;
        """)
        self._conn.commit()

        # Ensure media subdirectories exist
        for sub in ("im", "au", "vi", "ot"):
            (Path(self._media_dir) / sub).mkdir(parents=True, exist_ok=True)

    # -- CRUD ------------------------------------------------------------------

    def add_fact(
        self,
        content: str,
        category: str = "general",
        tags: str = "",
        importance: float = 5.0,
        entities: Optional[list[str]] = None,
        merge: bool = True,
        is_persistent: bool = False,
        dedup_threshold: float = 0.0,
        content_date: Optional[str] = None,
    ) -> dict:
        """Store a fact with three-level merging strategy.

        1. Exact content match → merge (keep higher importance/trust)
        2. FTS5 dedup (Jaccard >= threshold) → skip (near-duplicate rephrase)
        3. Shared entities + FTS5 similarity >= threshold → merge content
        4. No match → insert as new fact

        Args:
            dedup_threshold: If > 0, skip insert when Jaccard similarity with
                             an existing fact >= this threshold (0.7 recommended).
                             Used by auto-extraction sources to avoid cross-source
                             duplicates (LLM vs on_memory_write).

        When merging, content is intelligently combined, importance is max'd,
        tags are union'd, and a merge_log entry is created.
        """
        # Type validation
        if not isinstance(content, str) or not content.strip():
            raise ValueError("content must be a non-empty string")
        _VALID_CATEGORIES = {
            "place", "time", "person", "event", "activity",
            "identity", "preference", "goal",
            "project", "tool", "possession", "state", "opinion", "general",
        }
        if not isinstance(category, str) or category not in _VALID_CATEGORIES:
            category = "general"
        if not isinstance(tags, str):
            tags = str(tags) if tags is not None else ""
        if not isinstance(importance, (int, float)):
            importance = 5.0
        importance = max(1.0, min(10.0, float(importance)))

        with self._lock:
            # Extract entities from content
            extracted = self._extract_entities(content)
            if entities:
                extracted.extend(entities)
            extracted = list(dict.fromkeys(extracted))  # dedup

            # Level 1: Exact content match
            existing = self._conn.execute(
                "SELECT fact_id, importance, trust_score, content, tags, is_persistent FROM facts WHERE content = ?",
                (content,),
            ).fetchone()

            if existing:
                return self._merge_exact_match(existing, importance, tags, is_persistent)

            # Level 1.5: Fast dedup (Jaccard ≥ threshold) — skip near-duplicate rephrases
            if dedup_threshold > 0:
                dup = self._find_duplicate(content, dedup_threshold)
                if dup:
                    logger.debug("Dedup: skipped insert '%.60s' (Jaccard >= %.2f) — merged into fact #%d '%.60s'",
                                 content, dedup_threshold, dup["fact_id"], dup.get("content", ""))
                    return dup

            # Level 2: Semantic merge (shared entities + FTS5 similarity)
            if merge and extracted:
                candidate = self._find_merge_candidate(content, extracted, category)
                if candidate:
                    return self._merge_semantic(candidate, content, extracted, category, tags, importance, is_persistent)

            # Level 3: Insert new fact
            return self._insert_new(content, category, tags, importance, extracted, is_persistent, content_date)

    def _merge_exact_match(self, existing: tuple, importance: float, tags: str,
                            is_persistent: bool = False) -> dict:
        """Merge when content is identical."""
        fact_id, old_imp, old_trust, old_content, old_tags, old_persistent = existing
        new_importance = max(old_imp, importance)
        new_trust = max(old_trust, self._default_trust)
        new_persistent = max(old_persistent, 1 if is_persistent else 0)
        # Merge tags
        merged_tags = self._merge_tags(old_tags, tags)
        # Re-encode HRR to reflect updated importance/trust
        hrr_vector = self._encode_hrr(old_content)
        hrr_blob = hrr.phases_to_bytes(hrr_vector) if hrr_vector is not None else None
        self._conn.execute(
            """UPDATE facts SET importance=?, trust_score=?, tags=?,
               is_persistent=?, hrr_vector=?, updated_at=datetime('now') WHERE fact_id=?""",
            (new_importance, new_trust, merged_tags, new_persistent, hrr_blob, fact_id),
        )
        self._conn.commit()
        logger.debug("Merged exact duplicate fact #%d (importance %.1f)", fact_id, new_importance)
        return {"fact_id": fact_id, "content": old_content, "importance": new_importance,
                "is_persistent": bool(new_persistent), "merged": True, "merge_type": "exact"}

    # ── Language-aware dedup helpers ─────────────────────────────────
    _re_cjk = re.compile(r'[\u4e00-\u9fff]')
    _lemmatizer = None  # lazy init

    @classmethod
    def _is_chinese_text(cls, text: str) -> bool:
        """Detect if text is primarily Chinese (any CJK character present)."""
        return bool(cls._re_cjk.search(text))

    @classmethod
    def _tokenize_char2g(cls, text: str) -> set[str]:
        """Character bigrams for Chinese near-duplicate detection."""
        clean = re.sub(r'[^a-z0-9\u4e00-\u9fff\s]', '', text.lower())
        clean = re.sub(r'\s+', ' ', clean).strip()
        return set(clean[i:i+2] for i in range(len(clean)-1))

    @classmethod
    def _tokenize_lemma(cls, text: str) -> set[str]:
        """Lemmatized token set for English near-duplicate detection."""
        if cls._lemmatizer is None:
            cls._lemmatizer = WordNetLemmatizer()
        t = re.sub(r'[^a-zA-Z0-9+#]+', ' ', text.lower()).split()
        lemmas = set()
        for w in t:
            lemmas.add(cls._lemmatizer.lemmatize(w, 'v'))
            lemmas.add(cls._lemmatizer.lemmatize(w, 'n'))
        return lemmas

    def _find_duplicate(self, content: str, threshold: float = 0.7) -> Optional[dict]:
        """Check if a near-duplicate fact already exists via FTS5 + language-aware Jaccard.

        Uses lemmatized Jaccard (English, threshold=0.23) or char2g Jaccard
        (Chinese, threshold=0.03), automatically detected from content.

        Returns dict with fact_id/content/importance/merged or None.
        """
        is_cjk = self._is_chinese_text(content)
        if is_cjk:
            query_tokens = self._tokenize_char2g(content)
            dedup_threshold = 0.03
        else:
            query_tokens = self._tokenize_lemma(content)
            dedup_threshold = 0.5  # 0.23 was too aggressive — collapsed distinct facts
                                   # sharing common words (e.g. "Jon" + "January")
        if len(query_tokens) < 3:
            return None  # Too short for meaningful dedup

        # Build a broad FTS5 query (OR) from key tokens to find candidates.
        # AND is too strict — similar facts may use synonyms ("likes" vs "prefers").
        # Jaccard check below ensures precision; FTS5 just gathers candidates.
        safe = re.sub(r'[^\w\s\u4e00-\u9fff#+]', ' ', content)
        # Jieba-segment CJK tokens to match FTS5 indexed tokens
        segmented = []
        for w in safe.split():
            if self._re_cjk.search(w):
                segmented.extend(jieba.cut(w))
            else:
                segmented.append(w)
        words = [w for w in segmented if len(w) > 1][:5]
        if not words:
            return None
        safe = ' OR '.join(words)

        rows = self._conn.execute(
            """SELECT f.fact_id, f.content, f.importance, f.trust_score, f.is_persistent
               FROM facts_fts JOIN facts f ON facts_fts.rowid = f.fact_id
               WHERE facts_fts MATCH ?
               ORDER BY rank
               LIMIT 5""",
            (safe,),
        ).fetchall()

        for row in rows:
            if is_cjk:
                existing_tokens = self._tokenize_char2g(row["content"])
            else:
                existing_tokens = self._tokenize_lemma(row["content"])
            sim = jaccard_similarity(query_tokens, existing_tokens)
            if sim >= dedup_threshold:
                return {
                    "fact_id": row["fact_id"],
                    "content": row["content"],
                    "importance": row["importance"],
                    "is_persistent": bool(row["is_persistent"]),
                    "merged": True,
                    "merge_type": "dedup",
                }
        return None

    def _find_merge_candidate(
        self, content: str, entities: list[str], category: str,
    ) -> Optional[tuple]:
        """Find best existing fact to merge with, using entity overlap + FTS5.

        Returns (fact_id, content, importance, trust_score, tags, is_persistent) or None.
        """
        # Cap entities to prevent DoS via enormous query
        entities = entities[:20]
        if not entities:
            return None
        shared_facts = self._conn.execute(
            """SELECT f.fact_id, f.content, f.importance, f.trust_score, f.tags,
                      f.is_persistent, COUNT(fe.entity_id) AS entity_count
               FROM facts f
               JOIN fact_entities fe ON f.fact_id = fe.fact_id
               JOIN entities e ON fe.entity_id = e.entity_id
               WHERE e.name IN ({})
               AND f.category = ?
               AND f.fact_id NOT IN (
                   SELECT absorbed_fact_id FROM merge_log
               )
               GROUP BY f.fact_id
               HAVING entity_count >= ?
               ORDER BY entity_count DESC, f.importance DESC
               LIMIT 3""".format(",".join("?" * len(entities))),
            entities + [category, max(1, len(entities) // 2)],
        ).fetchall()

        if not shared_facts:
            return None

        # Strategy B: among entity-sharing facts, pick best FTS5 match
        query_tokens = tokenize(content)
        best = None
        best_score = 0.15  # minimum similarity threshold (jaccard × trust)

        for row in shared_facts:
            fid, fact_content, fact_imp, fact_trust, fact_tags, fact_persistent = (row[0], row[1], row[2], row[3], row[4], row[5])
            content_tokens = tokenize(fact_content)
            tag_tokens = tokenize(fact_tags or "")
            all_tokens = content_tokens | tag_tokens
            jaccard = jaccard_similarity(query_tokens, all_tokens)

            # Weighted score: Jaccard similarity × trust
            score = jaccard * (fact_trust or 0.5)
            if score > best_score:
                best_score = score
                best = (fid, fact_content, fact_imp, fact_trust, fact_tags, fact_persistent)

        return best

    def _merge_semantic(
        self, candidate: tuple, new_content: str, entities: list[str],
        category: str, tags: str, importance: float,
        is_persistent: bool = False,
    ) -> dict:
        """Merge new content into existing fact via content combining."""
        fact_id, existing_content, old_imp, old_trust, old_tags, old_persistent = candidate

        # Combine content intelligently
        merged_content = self._combine_fact_content(existing_content, new_content)
        merged_tags = self._merge_tags(old_tags, tags)
        new_importance = max(old_imp, importance)
        new_trust = max(old_trust, self._default_trust)
        new_persistent = max(old_persistent, 1 if is_persistent else 0)

        # Re-encode HRR with merged content
        hrr_vector = self._encode_hrr(merged_content, entities)
        hrr_blob = hrr.phases_to_bytes(hrr_vector) if hrr_vector is not None else None

        self._conn.execute(
            """UPDATE facts SET content=?, category=?, tags=?, importance=?,
               trust_score=?, is_persistent=?, hrr_vector=?, updated_at=datetime('now')
               WHERE fact_id=?""",
            (merged_content, category, merged_tags, new_importance,
             new_trust, new_persistent, hrr_blob, fact_id),
        )

        # Log the merge
        self._conn.execute(
            """INSERT INTO merge_log (kept_fact_id, absorbed_fact_id, merged_content, merge_reason)
               VALUES (?, NULL, ?, 'semantic')""",
            (fact_id, merged_content),
        )

        # Link any new entities
        for entity_name in entities:
            self._conn.execute(
                "INSERT OR IGNORE INTO entities (name) VALUES (?)", (entity_name,)
            )
            row = self._conn.execute(
                "SELECT entity_id FROM entities WHERE name = ?", (entity_name,)
            ).fetchone()
            if row:
                self._conn.execute(
                    "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                    (fact_id, row[0]),
                )

        self._conn.commit()
        logger.debug("Semantic merge into fact #%d: '%s' ← '%s'",
                     fact_id, existing_content[:60], new_content[:60])
        return {"fact_id": fact_id, "content": merged_content, "importance": new_importance,
                "merged": True, "merge_type": "semantic"}

    @staticmethod
    def _combine_fact_content(existing: str, new: str) -> str:
        """Intelligently combine two fact statements about the same topic.

        Strategies:
        - If one contains the other, keep the longer one
        - If they talk about different aspects, join with separator
        - If they contradict, keep both with [conflict] marker

        Note: contradiction detection is heuristic-only (token-level negation
        and limited antonym check). Full NLP contradiction detection is out of
        scope — this catches clear-cut cases only.
        """
        e_lower = existing.lower().strip()
        n_lower = new.lower().strip()

        # Same content (case-insensitive)
        if e_lower == n_lower:
            return existing

        # One is substring of the other (within similar length)
        if len(e_lower) >= len(n_lower) * 0.7 and n_lower in e_lower:
            return existing
        if len(n_lower) >= len(e_lower) * 0.7 and e_lower in n_lower:
            return new

        # Check for contradiction
        negation_words = {"not", "don't", "doesn't", "didn't", "won't", "can't",
                         "isn't", "aren't", "wasn't", "weren't", "never", "no"}
        cjk_neg_words = {"不喜欢", "不要", "不是", "没有", "不行", "不会", "不能", "拒绝"}
        # Antonym pairs — one fact uses one, the other uses its opposite
        antonym_pairs = [
            ({"love", "like", "enjoy", "prefer", "favorite"},
             {"hate", "dislike", "loathe", "detest", "讨厌", "不喜欢"}),
            ({"喜欢"}, {"讨厌"}),  # CJK antonym pair (不喜欢 → 不+喜欢 by jieba)
        ]
        # Tokenize with jieba for CJK support, fall back to split() for English.
        # Note: jieba splits "不喜欢" into "不" + "喜欢", so CJK negation words
        # are checked via substring matching on the raw text below.
        def _cjk_tokens(text: str) -> set[str]:
            """Tokenize text, using jieba for CJK segments."""
            import re as _re
            tokens = set()
            for word in _re.split(r'(\s+)', text):
                word = word.strip()
                if not word:
                    continue
                if _re.search(r'[\u4e00-\u9fff]', word):
                    tokens.update(jieba.cut(word))
                else:
                    tokens.add(word)
            return tokens

        e_tokens = _cjk_tokens(e_lower)
        n_tokens = _cjk_tokens(n_lower)
        common = e_tokens & n_tokens
        # English negation: exact token match
        e_has_eng_neg = any(w in e_tokens for w in negation_words)
        n_has_eng_neg = any(w in n_tokens for w in negation_words)
        # CJK negation: substring match (jieba splits compound negation)
        e_has_cjk_neg = any(w in e_lower for w in cjk_neg_words)
        n_has_cjk_neg = any(w in n_lower for w in cjk_neg_words)
        e_has_neg = e_has_eng_neg or e_has_cjk_neg
        n_has_neg = n_has_eng_neg or n_has_cjk_neg
        is_contradiction = False

        # Heuristic 1: shared tokens but one negated
        if len(common) >= 2 and e_has_neg != n_has_neg:
            is_contradiction = True

        # Heuristic 2: antonym-like pairs without shared negation
        if not is_contradiction:
            for pos_words, neg_words in antonym_pairs:
                e_pos = bool(e_tokens & pos_words)
                n_pos = bool(n_tokens & pos_words)
                e_neg = bool(e_tokens & neg_words)
                n_neg = bool(n_tokens & neg_words)
                if (e_pos and n_neg) or (e_neg and n_pos):
                    is_contradiction = True
                    break

        if is_contradiction:
            return f"{existing} [冲突] {new}"

        # Different aspects: combine
        # Check if the new info adds details not in existing
        new_words = n_tokens - e_tokens
        if len(new_words) >= 2:
            return f"{existing}；{new}"
        return existing  # No meaningful new info

    @staticmethod
    def _merge_tags(old_tags: str, new_tags: str) -> str:
        """Merge two comma-separated tag strings, deduplicated."""
        all_tags = set()
        for t in old_tags.split(","):
            t = t.strip()
            if t:
                all_tags.add(t)
        for t in new_tags.split(","):
            t = t.strip()
            if t:
                all_tags.add(t)
        return ", ".join(sorted(all_tags))

    def _insert_new(
        self, content: str, category: str, tags: str,
        importance: float, entities: list[str],
        is_persistent: bool = False,
        content_date: Optional[str] = None,
    ) -> dict:
        """Insert a brand-new fact."""
        hrr_vector = self._encode_hrr(content, entities)
        hrr_blob = hrr.phases_to_bytes(hrr_vector) if hrr_vector is not None else None
        cursor = self._conn.execute(
            """INSERT INTO facts (content, category, tags, importance, trust_score, is_persistent, content_date, hrr_vector)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (content, category, tags, importance, self._default_trust,
             1 if is_persistent else 0, content_date, hrr_blob),
        )
        fact_id = cursor.lastrowid
        if entities:
            self._link_entities(fact_id, entities)
        self._conn.commit()
        return {"fact_id": fact_id, "content": content, "importance": importance,
                "is_persistent": is_persistent, "merged": False}

    def get_fact(self, fact_id: int) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM facts WHERE fact_id = ?", (fact_id,)
            ).fetchone()
            if not row:
                return None
            return {key: row[key] for key in row.keys()}

    def update_fact(self, fact_id: int, **kwargs) -> bool:
        allowed = {"content", "category", "tags", "importance", "trust_score"}
        updates = {k: v for k, v in kwargs.items() if k in allowed}
        if not updates:
            return False
        updates["updated_at"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        # Build safe column list — keys are already whitelisted above
        columns = list(updates.keys())
        set_clause = ", ".join(f"{col}=?" for col in columns)
        values = [updates[col] for col in columns] + [fact_id]
        with self._lock:
            self._conn.execute(
                f"UPDATE facts SET {set_clause} WHERE fact_id=?", values
            )
            self._conn.commit()
            return True

    def remove_fact(self, fact_id: int) -> bool:
        with self._lock:
            cursor = self._conn.execute("DELETE FROM facts WHERE fact_id = ?", (fact_id,))
            self._conn.commit()
            return cursor.rowcount > 0

    # -- Media attachments ----------------------------------------------------

    def attach_media(self, fact_id: int, source_path: str, mime_type: str,
                     description: str = "", caption: str = "",
                     transcript: str = "") -> dict:
        """Attach a media file to a fact.

        - Compresses the file first if compression is enabled (configurable)
        - Copies file to content-addressed path under media_dir
        - Validates path safety (prevents traversal)
        - Computes SHA-256 for dedup
        - Inserts into media_attachments table
        - Optionally re-bundles HRR vector with description

        Returns dict with media_id, file_path, sha256.
        """
        # 1. Verify fact exists
        fact = self.get_fact(fact_id)
        if not fact:
            raise ValueError(f"Fact {fact_id} not found")

        # 2. Validate source file
        source_path = str(source_path)
        if not os.path.isfile(source_path):
            raise FileNotFoundError(f"Source file not found: {source_path}")

        # 2b. Compress file if enabled
        effective_source = source_path
        effective_mime = mime_type
        if self._compression_config:
            try:
                from .media_compressor import compress_media, DEFAULT_COMPRESSION_CONFIG
                # Deep merge user config over defaults (avoid mutating global)
                merged = copy.deepcopy(DEFAULT_COMPRESSION_CONFIG)
                if isinstance(self._compression_config, dict):
                    for key, val in self._compression_config.items():
                        if key in ("image", "video", "audio") and isinstance(val, dict):
                            merged.setdefault(key, {})
                            merged[key].update(val)
                        else:
                            merged[key] = val

                result_path, result_mime = compress_media(
                    source_path, mime_type,
                    output_dir=self._media_dir,
                    config=merged,
                )
                if result_path:
                    effective_source = result_path
                    effective_mime = result_mime
            except Exception:
                logger.warning("Compression error for %s, using original", source_path,
                               exc_info=True)

        # 3. Compute hash (chunked, memory-safe for large files)
        sha256_hash = hashlib.sha256()
        file_size = 0
        with open(effective_source, "rb") as f:
            while True:
                chunk = f.read(64 * 1024)  # 64KB chunks
                if not chunk:
                    break
                sha256_hash.update(chunk)
                file_size += len(chunk)
        sha256_val = sha256_hash.hexdigest()

        # 4. Determine target path (content-addressed)
        type_code = _media_type_prefix(effective_mime)
        ext = effective_mime.split("/")[-1]
        # Strip params (e.g. image/png; charset=utf-8 → png)
        ext = ext.split(";")[0].strip()
        # Strip +xml / +json suffix (e.g. svg+xml → svg)
        ext = ext.split("+")[0]
        ext = _EXT_MAP.get(ext, ext)

        rel_dir = f"{type_code}/{sha256_val[:2]}"
        filename = f"{sha256_val}.{ext}"
        rel_path = f"{rel_dir}/{filename}"

        # 5. Resolve absolute path and validate it stays within media_dir
        media_root = Path(self._media_dir).resolve()
        abs_dir = (media_root / rel_dir).resolve()
        abs_path = (media_root / rel_path).resolve()

        # Security: verify resolved path is inside media_root
        if not str(abs_path).startswith(str(media_root) + os.sep):
            # Clean up temp compressed file if applicable
            if effective_source != source_path:
                try:
                    os.unlink(effective_source)
                except OSError:
                    pass
            raise ValueError(f"Path traversal detected: {rel_path}")

        # 6. Create directory and copy file (if not already there = dedup)
        abs_dir.mkdir(parents=True, exist_ok=True)
        if not abs_path.exists():
            shutil.copy2(effective_source, str(abs_path))

        # 7. Insert into DB
        with self._lock:
            # Check for existing same-sha attachment on this fact (dedup within fact)
            existing = self._conn.execute(
                "SELECT media_id FROM media_attachments WHERE fact_id=? AND sha256=?",
                (fact_id, sha256_val),
            ).fetchone()
            if existing:
                logger.info("Media dedup: fact_id=%d sha256=%s rel_path=%s (same fact+sha)",
                            fact_id, sha256_val, rel_path)
                # Clean up temp file before early return
                if effective_source != source_path:
                    try:
                        os.unlink(effective_source)
                    except OSError:
                        pass
                return {"media_id": existing[0], "file_path": rel_path,
                        "sha256": sha256_val, "dedup": True}

            # Re-verify file exists (GC might have removed it between step 6 and here)
            if not abs_path.exists():
                abs_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy2(effective_source, str(abs_path))

            cursor = self._conn.execute(
                """INSERT INTO media_attachments
                   (fact_id, file_path, mime_type, file_size, sha256,
                    description, caption, transcript)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (fact_id, rel_path, effective_mime, file_size, sha256_val,
                 description, caption, transcript),
            )
            media_id = cursor.lastrowid

            # 8. Re-bundle HRR vector with media description
            if hrr._HAS_NUMPY and description:
                try:
                    fact_hrr = self._conn.execute(
                        "SELECT hrr_vector FROM facts WHERE fact_id=?",
                        (fact_id,),
                    ).fetchone()
                    if fact_hrr and fact_hrr[0] is not None:
                        existing_vec = hrr.bytes_to_phases(fact_hrr[0])
                        media_vec = hrr.encode_text(description, self._hrr_dim)
                        new_vec = hrr.bundle(existing_vec, media_vec)
                        new_blob = hrr.phases_to_bytes(new_vec)
                        self._conn.execute(
                            "UPDATE facts SET hrr_vector=? WHERE fact_id=?",
                            (new_blob, fact_id),
                        )
                except Exception:
                    pass  # non-critical

            self._conn.commit()

        # Clean up temp compressed file (after lock so TOCTOU path can still use it)
        if effective_source != source_path:
            try:
                os.unlink(effective_source)
            except OSError:
                pass

        logger.info("Media attached: media_id=%d fact_id=%d sha256=%s type=%s size=%d path=%s",
                     media_id, fact_id, sha256_val, mime_type, file_size, rel_path)

        # 9. Generate thumbnail (best-effort, non-critical)
        try:
            from .media_utils import generate_thumbnail
            generate_thumbnail(str(abs_path), mime_type, str(media_root))
        except Exception:
            pass  # thumbnail failure is non-fatal

        return {"media_id": media_id, "file_path": rel_path,
                "sha256": sha256_val, "dedup": False}

    def detach_media(self, media_id: int) -> bool:
        """Remove media attachment from DB. Does NOT delete file from disk."""
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM media_attachments WHERE media_id=?", (media_id,),
            )
            self._conn.commit()
            success = cursor.rowcount > 0
        if success:
            logger.info("Media detached: media_id=%d", media_id)
        else:
            logger.warning("Media detach failed: media_id=%d not found", media_id)
        return success

    def get_fact_media(self, fact_id: int) -> list[dict]:
        """Get all media attachments for a fact, ordered by creation time."""
        with self._lock:
            rows = self._conn.execute(
                """SELECT * FROM media_attachments WHERE fact_id=?
                   ORDER BY created_at DESC""",
                (fact_id,),
            ).fetchall()
            result = [{key: row[key] for key in row.keys()} for row in rows]
        logger.debug("get_fact_media(fact_id=%d): %d records", fact_id, len(result))
        return result

    def media_orphans(self) -> list[str]:
        """Find files on disk not referenced in DB.

        Walks the media directory tree and returns paths of files
        whose relative path doesn't exist in media_attachments.
        """
        media_root = Path(self._media_dir)
        if not media_root.exists():
            return []

        # Load DB paths under a brief lock, then walk disk without holding it
        with self._lock:
            db_paths = {
                row[0] for row in self._conn.execute(
                    "SELECT file_path FROM media_attachments"
                ).fetchall()
            }

        orphans = []
        for fpath in media_root.rglob("*"):
            if not fpath.is_file() or fpath.is_symlink():
                continue
            rel = str(fpath.relative_to(media_root))
            if rel not in db_paths:
                orphans.append(rel)
        if orphans:
            logger.info("media_orphans: %d orphan file(s) found (e.g. %s)",
                         len(orphans), orphans[0])
        else:
            logger.debug("media_orphans: no orphan files")
        return orphans

    def media_cleanup(self, dry_run: bool = True) -> dict:
        """Remove orphaned media files from disk via media_utils.

        Args:
            dry_run: If True, only report orphans without deleting.

        Returns:
            Dict with deleted, skipped, freed_bytes, dry_run, errors.
        """
        from .media_utils import media_cleanup as _do_cleanup
        return _do_cleanup(self, dry_run=dry_run)

    # -- List / Count ---------------------------------------------------------

    def list_facts(self, limit: int = 50, offset: int = 0,
                    persistent_only: bool = False) -> list[dict]:
        with self._lock:
            if persistent_only:
                rows = self._conn.execute(
                    "SELECT * FROM facts WHERE is_persistent = 1 ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT * FROM facts ORDER BY created_at DESC LIMIT ? OFFSET ?",
                    (limit, offset),
                ).fetchall()
            return [{key: r[key] for key in r.keys()} for r in rows]

    def count_facts(self, persistent_only: bool = False) -> int:
        with self._lock:
            if persistent_only:
                return self._conn.execute("SELECT COUNT(*) FROM facts WHERE is_persistent = 1").fetchone()[0]
            return self._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]

    def search_by_date(self, date: str, limit: int = 50) -> list[dict]:
        """Find facts with a specific content_date (YYYY-MM-DD).

        Args:
            date: ISO date string to match (exact or prefix for month/year queries).
            limit: Max facts to return.
        """
        with self._lock:
            # Exact match first, then prefix (for "2023-01" matching "2023-01-19")
            rows = self._conn.execute(
                """SELECT * FROM facts WHERE content_date = ?
                   UNION
                   SELECT * FROM facts WHERE content_date LIKE ? AND content_date != ?
                   ORDER BY content_date LIMIT ?""",
                (date, date + "%", date, limit),
            ).fetchall()
            return [{key: r[key] for key in r.keys()} for r in rows]

    # -- Feedback --------------------------------------------------------------

    def record_feedback(self, fact_id: int, helpful: bool) -> dict:
        with self._lock:
            row = self._conn.execute(
                "SELECT trust_score, importance, helpful_count, retrieval_count FROM facts WHERE fact_id=?",
                (fact_id,),
            ).fetchone()
            if not row:
                return {"error": "fact not found"}
            trust, importance, helpful_count, retrieval_count = (
                row["trust_score"], row["importance"], row["helpful_count"], row["retrieval_count"]
            )
            if helpful:
                trust = min(_TRUST_MAX, trust + _HELPFUL_DELTA)
                importance = min(_IMPORTANCE_MAX, importance + _IMPORTANCE_DELTA)
                helpful_count += 1
            else:
                trust = max(_TRUST_MIN, trust + _UNHELPFUL_DELTA)
                importance = max(_IMPORTANCE_MIN, importance - _IMPORTANCE_DELTA)
            self._conn.execute(
                """UPDATE facts SET trust_score=?, importance=?, helpful_count=?,
                   retrieval_count=?, updated_at=datetime('now') WHERE fact_id=?""",
                (trust, importance, helpful_count, retrieval_count, fact_id),
            )
            self._conn.commit()
            return {"fact_id": fact_id, "trust_score": trust, "importance": importance}

    # -- Entity management -----------------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        """Extract entity candidates from text."""
        entities = set()
        for pattern in (_RE_CAPITALIZED, _RE_SINGLE_NAME, _RE_DOUBLE_QUOTE, _RE_SINGLE_QUOTE,
                        _RE_CJK_BRACKETS, _RE_QUOTED_CN):
            for match in pattern.finditer(text):
                for group in match.groups():
                    if group and len(group) > 1:
                        entities.add(group.strip())
        # AKA patterns
        for match in _RE_AKA.finditer(text):
            entities.add(match.group(1).strip())
            entities.add(match.group(2).strip())
        # Filter out garbage: possessive fragments ("s Web", "s favorite...")
        # and common non-entity words
        _STOP_ENTITIES = {
            'the', 'this', 'that', 'these', 'those', 'here', 'there',
            'when', 'where', 'what', 'which', 'who', 'how', 'why',
            'user', 'assistant', 'memory', 'context', 'based', 'following',
            'however', 'therefore', 'meanwhile', 'otherwise', 'instead',
        }
        return [
            e for e in entities
            if len(e) >= 2
            and not e.startswith("s ")      # possessive garbage
            and not e.startswith("'s ")      # possessive garbage
            and e.lower() not in _STOP_ENTITIES
        ]

    def _link_entities(self, fact_id: int, entity_names: list[str]) -> None:
        """Associate entities with a fact, creating them if needed."""
        for name in entity_names:
            # Upsert entity
            self._conn.execute(
                "INSERT INTO entities (name) VALUES (?) ON CONFLICT(name) DO NOTHING",
                (name,),
            )
            row = self._conn.execute(
                "SELECT entity_id FROM entities WHERE name = ?", (name,)
            ).fetchone()
            if row:
                # Link fact ↔ entity
                self._conn.execute(
                    "INSERT OR IGNORE INTO fact_entities (fact_id, entity_id) VALUES (?, ?)",
                    (fact_id, row[0]),
                )

    def get_entity_facts(self, entity_name: str, limit: int = 20) -> list[dict]:
        """Get all facts linked to an entity."""
        with self._lock:
            # Escape SQL LIKE wildcards in entity name
            safe_entity = entity_name.replace("%", "\\%").replace("_", "\\_")
            rows = self._conn.execute(
                """SELECT f.* FROM facts f
                   JOIN fact_entities fe ON f.fact_id = fe.fact_id
                   JOIN entities e ON fe.entity_id = e.entity_id
                   WHERE e.name = ? OR e.aliases LIKE ? ESCAPE '\\'
                   ORDER BY f.importance DESC, f.trust_score DESC
                   LIMIT ?""",
                (entity_name, f"%{safe_entity}%", limit),
            ).fetchall()
            return [{key: r[key] for key in r.keys()} for r in rows]

    def get_fact_ids_for_entities(self, entity_names: list[str]) -> set[int]:
        """Return set of fact_ids linked to any of the given entity names."""
        if not entity_names:
            return set()
        placeholders = ",".join("?" for _ in entity_names)
        try:
            rows = self._conn.execute(
                f"""SELECT DISTINCT fe.fact_id FROM fact_entities fe
                    JOIN entities e ON fe.entity_id = e.entity_id
                    WHERE e.name IN ({placeholders})""",
                entity_names,
            ).fetchall()
            return {r[0] for r in rows}
        except Exception:
            return set()

    def get_entity_timeline(self, entity_name: str, limit: int = 20,
                            min_importance: float = 0.0) -> list[dict]:
        """Get all facts linked to an entity, sorted chronologically (oldest first).

        Unlike get_entity_facts (sorted by importance), this returns facts in
        temporal order so the agent can trace how preferences, decisions, or
        project architecture evolved over time.

        Args:
            entity_name: Entity to query.
            limit: Max results.
            min_importance: Minimum importance filter (0 = no filter).
        """
        with self._lock:
            safe_entity = entity_name.replace("%", "\\%").replace("_", "\\_")
            rows = self._conn.execute(
                """SELECT f.* FROM facts f
                   JOIN fact_entities fe ON f.fact_id = fe.fact_id
                   JOIN entities e ON fe.entity_id = e.entity_id
                   WHERE (e.name = ? OR e.aliases LIKE ? ESCAPE '\\')
                   AND f.importance >= ?
                   ORDER BY f.created_at ASC
                   LIMIT ?""",
                (entity_name, f"%{safe_entity}%", min_importance, limit),
            ).fetchall()
            return [{key: r[key] for key in r.keys()} for r in rows]

    def get_related_entities(self, entity_name: str, depth: int = 2) -> list[dict]:
        """BFS traversal for related entities up to `depth` hops."""
        with self._lock:
            visited = set()
            queue = deque()
            queue.append((entity_name, 0))
            results = []
            while queue:
                name, d = queue.popleft()
                if name in visited or d > depth:
                    continue
                visited.add(name)
                rows = self._conn.execute(
                    """SELECT DISTINCT e2.name FROM entities e1
                       JOIN fact_entities fe1 ON e1.entity_id = fe1.entity_id
                       JOIN fact_entities fe2 ON fe1.fact_id = fe2.fact_id AND fe2.entity_id != fe1.entity_id
                       JOIN entities e2 ON fe2.entity_id = e2.entity_id
                       WHERE e1.name = ?""",
                    (name,),
                ).fetchall()
                for row in rows:
                    related_name = row[0]
                    if related_name not in visited:
                        results.append({"source": name, "target": related_name, "depth": d + 1})
                        queue.append((related_name, d + 1))
            return results

    # -- Entity summary (S6.4) -------------------------------------------------

    def get_entity_summary(self, entity_name: str, limit: int = 50) -> dict:
        """Return a structured summary card for an entity.

        Groups facts by category, identifies current state (latest per category),
        detects contradictions, and returns a timeline — all without extra LLM cost.

        Returns:
            dict with entity, facts_count, by_category, current_state,
                 timeline, conflicts, related_entities.
        """
        facts = self.get_entity_facts(entity_name, limit=limit)
        if not facts:
            return {
                "entity": entity_name, "facts_count": 0,
                "by_category": {}, "current_state": {},
                "timeline": [], "conflicts": [],
                "related_entities": [],
            }

        # Timeline: sorted chronologically
        timeline = sorted(facts, key=lambda f: f.get("created_at", ""))

        # Group by category
        by_category: dict[str, list[dict]] = {}
        for f in timeline:
            cat = f.get("category", "general")
            by_category.setdefault(cat, []).append(f)

        # Current state: latest fact in each category
        current_state = {}
        for cat, cat_facts in by_category.items():
            latest = max(cat_facts, key=lambda f: f.get("created_at", ""))
            current_state[cat] = {
                "content": latest.get("content", ""),
                "importance": latest.get("importance", 5),
                "trust_score": latest.get("trust_score", 0.5),
                "updated_at": latest.get("created_at", ""),
            }

        # Detect conflicts within entity
        conflicts = self._find_entity_conflicts(timeline)

        # Related entities
        related = self.get_related_entities(entity_name, depth=1)

        return {
            "entity": entity_name,
            "facts_count": len(timeline),
            "by_category": {k: len(v) for k, v in by_category.items()},
            "current_state": current_state,
            "timeline": timeline,
            "conflicts": conflicts,
            "related_entities": [r["target"] for r in related],
        }

    def _find_entity_conflicts(self, facts: list[dict]) -> list[dict]:
        """Find contradictory fact pairs within a list of facts."""
        conflicts = []
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                c1 = facts[i].get("content", "")
                c2 = facts[j].get("content", "")
                if self._is_contradictory(c1, c2):
                    conflicts.append({
                        "fact_id_a": facts[i]["fact_id"],
                        "content_a": c1,
                        "fact_id_b": facts[j]["fact_id"],
                        "content_b": c2,
                    })
                    if len(conflicts) >= 10:
                        return conflicts
        return conflicts

    @staticmethod
    def _is_contradictory(a: str, b: str) -> bool:
        """Rough heuristic: check for negation markers between similar statements.

        Handles both English (whitespace-delimited) and CJK (no word boundaries)
        by using tokenize() for the common-token check (uses jieba word-level)
        and checking English negation at token level, CJK at substring level.
        """
        from .retrieval import tokenize
        a_lower = a.lower()
        b_lower = b.lower()
        eng_neg = {"not", "don't", "doesn't", "didn't", "won't", "can't",
                   "isn't", "aren't", "wasn't", "weren't", "never", "no"}
        cjk_neg = {"不喜欢", "不要", "不是", "没有", "不行"}
        # Use tokenize() for the common-token check — uses jieba word-level
        a_tok = tokenize(a)
        b_tok = tokenize(b)
        common = a_tok & b_tok
        if len(common) < 3:
            return False
        # English negation: token-level (word boundaries via split)
        a_tokens = set(a_lower.split())
        b_tokens = set(b_lower.split())
        has_eng_a = any(n in a_tokens for n in eng_neg)
        has_eng_b = any(n in b_tokens for n in eng_neg)
        # CJK negation: substring-level (no whitespace word boundaries)
        has_cjk_a = any(n in a_lower for n in cjk_neg)
        has_cjk_b = any(n in b_lower for n in cjk_neg)
        return (has_eng_a or has_cjk_a) != (has_eng_b or has_cjk_b)

    # -- HRR encoding ----------------------------------------------------------

    def _encode_hrr(self, content: str, entities: list[str] = None) -> Optional["np.ndarray"]:
        """Encode fact content + entities into HRR phase vector."""
        if not hrr._HAS_NUMPY:
            return None
        try:
            if entities:
                return hrr.encode_fact(content, entities, self._hrr_dim)
            return hrr.encode_text(content, self._hrr_dim)
        except Exception:
            return None

    def compute_hrr_similarity(self, fact_id: int, query: str) -> float:
        """HRR similarity between a stored fact and a query string."""
        if not hrr._HAS_NUMPY:
            return 0.5
        with self._lock:
            row = self._conn.execute(
                "SELECT hrr_vector FROM facts WHERE fact_id = ? AND hrr_vector IS NOT NULL",
                (fact_id,),
            ).fetchone()
            if not row:
                return 0.5
            try:
                fact_vec = hrr.bytes_to_phases(row[0])
                query_vec = hrr.encode_text(query, self._hrr_dim)
                return (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0
            except Exception:
                return 0.5

    # -- Close ----------------------------------------------------------------

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    # -- Public connection access for retriever --------------------------------

    def execute_query(self, sql: str, params: tuple = ()) -> list:
        """Execute a read-only SQL query and return all rows.
        Used by ThreeDimRetriever to access FTS5 search results."""
        with self._lock:
            return self._conn.execute(sql, params).fetchall()
