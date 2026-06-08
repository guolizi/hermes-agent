"""Synonym dictionary for FTS5 query expansion in retrieval.

Three-tier architecture:
  Tier 1 (Manual override): Small curated dict for high-polysemy English words
    and Chinese synonyms (OMW coverage is uneven).
  Tier 2 (Auto-expansion via WordNet/OMW): Lazy, on-demand, cached.
  Tier 3 (User custom): synonyms_custom.json always beats auto.

``get_synonyms(token)`` lookup order:
  1. Manual dict (highest precision)
  2. Custom JSON (user overrides)
  3. WordNet auto (lazy, cached)

Design goals:
  - No dependency on heavy NLP models (just WordNet data files, ~6 MB).
  - Lazy WordNet initialization → fast import time.
  - On-demand lookup → only expand words that actually appear in queries.
  - Quality gate: skip words with >15 WordNet synsets (too noisy).
"""

from __future__ import annotations

import json
import os
import unicodedata
from typing import Final

# ---------------------------------------------------------------------------
# Tier 1: Manual override (always beats any auto-expansion)
# ---------------------------------------------------------------------------
# English entries: high-polysemy words (>15 WordNet synsets) or words whose
# WordNet synonyms would be too noisy for FTS5 query expansion.
# Chinese entries: OMW coverage is uneven; manual entries are more reliable.

_MANUAL: dict[str, list[str]] = {
    # --- English: high-polysemy generic verbs ---
    "play":     ["game", "sport", "activity"],
    "make":     ["create", "build", "produce", "craft"],
    "give":     ["offer", "provide", "present"],
    "change":   ["transition", "shift", "transformation", "difference"],
    "work":     ["job", "career", "employment", "occupation"],
    "get":      ["receive", "obtain", "acquire"],
    "support":  ["help", "assist", "aid"],
    "have":     ["own", "possess"],
    "feel":     ["experience", "sense", "go through"],
    "study":    ["learn", "research", "education", "read"],
    "fix":      ["repair", "mend", "restore"],
    # --- English: words with no useful WordNet synonyms ---
    "concert":  ["show", "performance", "gig", "live"],
    # --- English: additional high-value curated pairs ---
    "recent":   ["last", "past", "recently"],
    "pursue":   ["chase", "strive", "go after", "follow"],
    "use":      ["utilize", "employ"],
    "hate":     ["dislike", "despise", "detest"],
    "family":   ["kid", "children", "parent"],
    "parent":   ["mother", "father", "mom", "dad"],
    "travel":   ["trip", "vacation", "journey", "tour"],
    # --- English: words where key synonyms are deep in WordNet synsets ---
    "hurt":     ["injured", "pain", "ache", "wound"],
    "music":    ["song", "concert", "band", "artist", "musical"],
    "buy":      ["purchase", "acquire", "get"],
    # --- English: "visit" WordNet misses "trip" — "took a trip" ≠ "visited"
    "visit":    ["trip", "see", "travel", "go to"],
    "visited":  ["trip", "see", "travel", "go to"],
    # --- English: "event" has wrong WordNet sense (case/consequence/effect/outcome)
    #              — add the "organized activity" synonyms instead
    "event":    ["fair", "competition", "festival", "gathering", "conference", "workshop", "showcase"],
    "events":   ["fair", "competition", "festival", "gathering", "conference", "workshop", "showcase"],
    # --- English: "mentorship" FTS5 doesn't stem to "mentor" or "mentored" ---
    "mentorship": ["mentor", "mentored", "guidance", "advice"],
    # ===== Chinese =====
    "家庭":   ["家人", "孩子", "子女", "亲戚"],
    "家人":   ["家庭", "孩子", "子女"],
    "孩子":   ["学生", "青年", "少年", "小孩", "子女"],
    "朋友":   ["好友", "哥们", "伙伴", "闺蜜"],
    "参加":   ["参与", "加入", "出席", "报名"],
    "参与":   ["参加", "加入", "出席"],
    "加入":   ["参加", "参与", "报名"],
    "活动":   ["爱好", "运动", "娱乐", "消遣"],
    "爱好":   ["活动", "兴趣", "嗜好", "喜欢"],
    "喜欢":   ["爱好", "喜爱", "热爱", "钟情"],
    "玩":     ["游戏", "运动", "娱乐"],
    "帮助":   ["支持", "协助", "帮忙", "援助"],
    "支持":   ["帮助", "协助", "支援"],
    "时间":   ["日期", "时候", "何时"],
    "计划":   ["打算", "目标", "希望", "准备"],
    "最近":   ["近期", "前几天", "上次"],
    "经常":   ["常常", "总是", "频繁"],
    "地方":   ["地点", "位置", "场所", "哪里"],
    "去":     ["到", "前往", "出发"],
    "旅行":   ["旅游", "度假", "出行", "游玩"],
    "说":     ["告诉", "讲", "提到", "聊"],
    "聊天":   ["讨论", "交流", "谈话", "聊"],
    "讨论":   ["聊天", "交流", "商量", "谈"],
    "开心":   ["快乐", "高兴", "愉快", "幸福"],
    "难过":   ["伤心", "不开心", "沮丧", "低落"],
    "感觉":   ["觉得", "感受", "感到"],
    "想":     ["希望", "打算", "想要", "愿望"],
    "工作":   ["上班", "职业", "事业", "打工"],
    "学习":   ["学", "读书", "研究", "上学"],
    "买":     ["购买", "购置", "入手"],
    "看":     ["阅读", "观看", "浏览", "读"],
    "做":     ["干", "搞", "弄", "进行"],
    "吃":     ["喝", "品尝", "用餐"],
    "给":     ["送", "提供", "递"],
}

# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

#: WordNet words with more synsets than this are skipped (too polysemous).
_MAX_AUTO_SYNSETS: Final[int] = 15
#: Max auto-expanded synonyms per word.
_MAX_AUTO_SYNS: Final[int] = 10

#: Path to optional user-custom synonym file (JSON dict of word → list).
_CUSTOM_PATH: Final[str] = os.path.join(
    os.path.dirname(__file__), "synonyms_custom.json",
)

# ---------------------------------------------------------------------------
# Tier 3: User custom synonyms
# ---------------------------------------------------------------------------

def _load_custom() -> dict[str, list[str]]:
    """Load user-custom synonyms from JSON, merged on top of manual."""
    if not os.path.exists(_CUSTOM_PATH):
        return {}
    try:
        with open(_CUSTOM_PATH) as f:
            return dict(json.load(f))
    except (json.JSONDecodeError, OSError):
        return {}


_CUSTOM: dict[str, list[str]] = _load_custom()

# ---------------------------------------------------------------------------
# Tier 2: WordNet / OMW auto-expansion (lazy, on-demand, cached)
# ---------------------------------------------------------------------------

_WORDNET_CACHE: dict[str, list[str] | None] = {}
"""Cache for WordNet lookups. ``None`` = already checked and absent."""

_WORDNET_READY: bool | None = None
"""Lazy-initialised flag: ``None`` = unchecked, ``True`` = available."""


def _ensure_wordnet() -> bool:
    """Check WordNet + OMW availability once (idempotent)."""
    global _WORDNET_READY
    if _WORDNET_READY is not None:
        return _WORDNET_READY
    try:
        import nltk  # noqa: F401
        nltk.data.path.insert(0, os.path.expanduser("~/nltk_data"))
        from nltk.corpus import wordnet as wn  # noqa: F401
        _WORDNET_READY = True
    except (ImportError, LookupError):
        _WORDNET_READY = False
    return _WORDNET_READY


def _is_cjk(text: str) -> bool:
    """Check if *text* contains CJK Unified Ideographs."""
    return any("\u4e00" <= c <= "\u9fff" for c in text)


def _lookup_wordnet(word: str) -> list[str]:
    """Look up synonyms via WordNet (English) / OMW (Chinese).

    Returns an empty list when:
    - WordNet is unavailable.
    - The word has more than ``_MAX_AUTO_SYNSETS`` synsets (too noisy).
    - No useful synonyms are found.
    """
    if not _ensure_wordnet():
        return []

    from nltk.corpus import wordnet as wn

    synsets: list[object] = []
    if _is_cjk(word):
        synsets = wn.synsets(word, lang="cmn")  # type: ignore[assignment]
    if not synsets:
        synsets = wn.synsets(word)  # type: ignore[assignment]
    if not synsets:
        return []

    if len(synsets) > _MAX_AUTO_SYNSETS:
        return []  # Too polysemous → too noisy

    # Preserve WordNet synset order (first synsets = more common senses).
    # This avoids the problem of sorting by length which can push out
    # relevant synonyms like "injure" (longer) in favour of "ache" (shorter).
    seen: set[str] = set()
    ordered: list[str] = []
    for s in synsets:
        for lemma in s.lemma_names():  # type: ignore[union-attr]
            clean = lemma.lower().replace("_", " ")
            if clean != word and len(clean) > 1 and clean not in seen:
                seen.add(clean)
                ordered.append(clean)
        if len(ordered) >= _MAX_AUTO_SYNS:
            break

    return ordered[:_MAX_AUTO_SYNS]


def _get_wordnet_syns(word: str) -> list[str]:
    """Cached WordNet synonym lookup (idempotent)."""
    lower = word.lower()
    cached = _WORDNET_CACHE.get(lower)
    if cached is not None:
        return list(cached)  # Return a copy
    if cached is None and lower in _WORDNET_CACHE:
        return []  # Previously confirmed absent
    syns = _lookup_wordnet(lower)
    _WORDNET_CACHE[lower] = syns if syns else None
    return list(syns) if syns else []


# ---------------------------------------------------------------------------
# Resolved synonym map (lazily built, for introspection / backward compat)
# ---------------------------------------------------------------------------

_SYNS_BY_PRIORITY: tuple[dict[str, list[str]], ...] = (
    _MANUAL,
    _CUSTOM,
)
"""Lookup order: manual (highest) → custom. WordNet is appended lazily."""


def _build_merged(word: str) -> list[str]:
    """Resolve *word* across all tiers: manual → custom → WordNet."""
    lower = word.lower()

    # Tiers 1-3: manual + custom (static)
    for src in _SYNS_BY_PRIORITY:
        if lower in src:
            return list(src[lower])

    # Tier 2: WordNet auto (lazy)
    return _get_wordnet_syns(lower)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_synonyms(token: str) -> list[str]:
    """Return the synonym list for *token*, or an empty list.

    Lookup is case-insensitive.  Lookup order:
      1. Manual override dict (highest precision)
      2. ``synonyms_custom.json`` (user edits)
      3. WordNet / OMW auto-expansion (lazy, cached)

    Returns a copy so callers can safely mutate the result list.
    """
    return list(_build_merged(token))


def add_synonym(word: str, synonyms: list[str]) -> None:
    """Add or overwrite a synonym entry at runtime (testing convenience).

    The entry is stored in the manual tier, beating both custom and
    WordNet expansions.
    """
    _MANUAL[word.lower()] = [s.lower() for s in synonyms]
