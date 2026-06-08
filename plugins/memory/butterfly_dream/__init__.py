"""🦋 Butterfly Dream — 3-Dimensional Memory Plugin for Hermes Agent.

A MemoryProvider plugin that scores facts across three dimensions:
  Relevance (semantic), Recency (temporal decay), Importance (LLM-assigned).

庄周梦蝶 — 记忆如蝶，翩跹于时间、意义与关联的三维空间。
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from hermes_cli.config import cfg_get
from hermes_constants import get_hermes_home

from .store import MemoryStore
from .retrieval import ThreeDimRetriever

logger = logging.getLogger(__name__)

# Known provider base URLs (can be overridden via {PROVIDER}_BASE_URL env)
_DEFAULT_BASE_URLS = {
    "deepseek": "https://api.deepseek.com/v1",
    "openai": "https://api.openai.com/v1",
    "anthropic": "https://api.anthropic.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1",
    "minimax": "https://api.minimax.chat/v1",
    "glm": "https://open.bigmodel.cn/api/paas/v4",
    "ollama": "http://localhost:11434/v1",
}


# ---------------------------------------------------------------------------
# Date normalization helper
# ---------------------------------------------------------------------------

def _normalize_date(raw) -> Optional[str]:
    """Normalize various date formats to YYYY-MM-DD string.

    Handles:
    - Already ISO: "2023-01-19" → "2023-01-19"
    - Common formats: "January 19, 2023", "19 January 2023", "Jan 19, 2023"
    - Chinese: "2023年1月19日", "2023/01/19", "2023.01.19"
    - Month + Year: "February 2023" → "2023-02-01"
    - Standalone year: "2023" → "2023-01-01"
    - Chinese: "2023年1月19日", "2023/01/19", "2023.01.19"
    - Returns None if no valid date found or input is None/empty.
    """
    if not raw or not isinstance(raw, str):
        return None
    raw = raw.strip()
    if not raw:
        return None

    # Already ISO format
    if re.match(r'^\d{4}-\d{2}-\d{2}$', raw):
        return raw

    # Try common English formats (full date)
    from datetime import datetime as _dt
    for fmt in (
        "%B %d, %Y",    # January 19, 2023
        "%b %d, %Y",    # Jan 19, 2023
        "%d %B %Y",     # 19 January 2023
        "%d %b %Y",     # 19 Jan 2023
        "%Y/%m/%d",     # 2023/01/19
        "%Y.%m.%d",     # 2023.01.19
        "%m/%d/%Y",     # 01/19/2023 (US)
        "%d/%m/%Y",     # 19/01/2023 (EU)
    ):
        try:
            return _dt.strptime(raw, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue

    # Month + Year (no day) → default to 1st of month
    for fmt in (
        "%B %Y",        # February 2023
        "%b %Y",        # Feb 2023
    ):
        try:
            return _dt.strptime(raw, fmt).strftime("%Y-%m-01")
        except ValueError:
            continue

    # Standalone 4-digit year → January 1st
    m = re.match(r'^(\d{4})$', raw)
    if m:
        return f"{m.group(1)}-01-01"

    # Chinese date: 2023年1月19日
    m = re.match(r'(\d{4})年(\d{1,2})月(\d{1,2})日?', raw)
    if m:
        return f"{m.group(1)}-{int(m.group(2)):02d}-{int(m.group(3)):02d}"

    return None  # Unparseable — skip


def _extract_date_from_content(content: str) -> Optional[str]:
    """Fallback: extract a date from fact content text when LLM omits content_date.

    Tries common patterns like "on 19 January, 2023", "January 19, 2023",
    "2023-01-19", "February 2023", "in 2023", "2023年1月19日", etc.
    Returns YYYY-MM-DD or None.
    """
    if not content:
        return None

    # Pattern: "on <date>"
    m = re.search(r'\bon\s+([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})', content)
    if m:
        return _normalize_date(m.group(1))

    # Pattern: standalone date at various positions
    for pattern in [
        r'(\d{1,2}\s+[A-Z][a-z]+\s+\d{4})',       # 19 January 2023
        r'([A-Z][a-z]+\s+\d{1,2},?\s+\d{4})',      # January 19, 2023
        r'(\d{4}[-/]\d{1,2}[-/]\d{1,2})',           # 2023-01-19 or 2023/01/19
        r'(\d{4}年\d{1,2}月\d{1,2}日?)',              # 2023年1月19日
    ]:
        m = re.search(pattern, content)
        if m:
            result = _normalize_date(m.group(1))
            if result:
                return result

    # Pattern: Month + Year (no day) with time preposition
    # e.g. "in February 2023", "around March 2023", "since June 2023"
    m = re.search(
        r'\b(?:in|during|around|since|until|by|before|after)\s+'
        r'([A-Z][a-z]+\s+\d{4})\b',
        content
    )
    if m:
        result = _normalize_date(m.group(1))
        if result:
            return result

    # Pattern: standalone 4-digit year with time preposition
    # e.g. "in 2023", "during 2022"
    m = re.search(r'\b(?:in|during|since|around)\s+(\d{4})\b', content)
    if m:
        result = _normalize_date(m.group(1))
        if result:
            return result

    return None


# ---------------------------------------------------------------------------
# LLM extraction prompt (enhanced with importance scoring)
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """You are a meticulous fact extractor. Extract EVERY concrete detail from conversations as separate facts.

GOLDEN RULES (in order):
1. Better to have 20 facts too many than miss 1 important one
2. Extract EVERY specific detail as its own standalone fact — do NOT merge different details
3. Entity names, numbers, dates must be preserved exactly as mentioned
4. "partner is pregnant" and "values family highly" are TWO separate facts at different specificity levels
5. Use the ACTUAL names from the conversation content — never "User", "Assistant", "the user", "the assistant", "Participant 1", or "Participant 2". If the speakers introduce themselves as "Evan" and "Sam", use "Evan" and "Sam" in every fact.

Output strictly a JSON array of objects. Each object has:
- "content": str, the fact (plain text, one complete sentence)
- "category": str — one of: place, time, person, event, activity, preference, identity, goal, project, tool, possession, state, opinion, general
- "entities": [str] — at least the primary subject
- "tags": str — comma-separated keywords
- "importance": int 1-10 (7-9 major life events, 4-6 significant details/plans/preferences, 1-3 minor)
- "is_persistent": bool
- "content_date": "YYYY-MM-DD" or null. CRITICAL: This must be the EVENT's actual date, not the conversation date. If the event is "last week", "a few days ago", or "yesterday", compute the real date relative to the session date shown in [Date: ...] and put the computed date here.
"""


# ---------------------------------------------------------------------------
# Tool schemas (extended with importance & scenario support)
# ---------------------------------------------------------------------------

FACT_STORE_SCHEMA = {
    "name": "fact_store",
    "description": (
        "Three-dimensional memory with algebraic reasoning. "
        "Use for deep recall across relevance, recency, and importance.\\n\\n"
        "ACTIONS:\\n"
        "• add — Store a fact the user would expect you to remember.\\n"
        "• search — 3D keyword/semantic search ('editor config', 'deploy process').\\n"
        "  Pass scenario='chat'|'technical'|'longterm'|'qa' to tune weights.\\n"
        "• probe — Entity recall: ALL facts about a person/thing.\\n"
        "• related — What connects to an entity? Structural adjacency.\\n"
        "• reason — Compositional: facts connected to MULTIPLE entities simultaneously.\\n"
        "• contradict — Memory hygiene: find facts making conflicting claims.\\n"
        "• timeline — Entity facts sorted chronologically (oldest first).\\n"
        "  Trace how preferences/decisions evolved over time.\\n"
        "• summarize — Entity summary card: current state per category,\\n"
        "  timeline, conflicts, and related entities. No extra LLM call.\\n"
        "• update/remove/list — CRUD operations.\\n\\n"
        "IMPORTANT: Before answering questions about the user, ALWAYS probe or reason first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "probe", "related", "reason", "contradict", "timeline", "summarize", "update", "remove", "list"],
            },
            "content": {"type": "string", "description": "Fact content (required for 'add')."},
            "query": {"type": "string", "description": "Search query (required for 'search')."},
            "entity": {"type": "string", "description": "Entity name for 'probe'/'related'/'timeline'/'summarize'."},
            "entities": {
                "type": "array", "items": {"type": "string"},
                "description": "Entity names for 'reason'.",
            },
            "fact_id": {"type": "integer", "description": "Fact ID for 'update'/'remove'."},
            "category": {"type": "string", "enum": ["place", "time", "person", "event", "activity", "identity", "preference", "goal", "project", "tool", "possession", "state", "opinion", "general"]},
            "tags": {"type": "string", "description": "Comma-separated tags."},
            "importance": {
                "type": "integer", "description": "Importance 1-10 (used for 'add').",
            },
            "trust_delta": {"type": "number", "description": "Trust adjustment for 'update'."},
            "min_trust": {"type": "number", "description": "Minimum trust filter (default: 0.3)."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
            "min_importance": {
                "type": "number",
                "description": "Minimum importance filter for 'timeline' (default: 0 = no filter).",
            },
            "scenario": {
                "type": "string",
                "enum": ["chat", "technical", "longterm", "qa", "balanced"],
                "description": "Retrieval weight scenario (default: 'balanced').",
            },
            "persistent_only": {
                "type": "boolean",
                "description": "If true, only return facts marked as persistent (long-lived).",
            },
        },
        "required": ["action"],
    },
}

FACT_FEEDBACK_SCHEMA = {
    "name": "fact_feedback",
    "description": (
        "Rate a fact after using it. Mark 'helpful' if accurate, 'unhelpful' if outdated. "
        "This trains both trust and importance — good facts rise, bad facts sink."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["helpful", "unhelpful"]},
            "fact_id": {"type": "integer", "description": "The fact ID to rate."},
        },
        "required": ["action", "fact_id"],
    },
}

MEDIA_ATTACH_SCHEMA = {
    "name": "media_attach",
    "description": (
        "Attach a media file (image, audio, video) to an existing fact. "
        "The file is content-addressed (SHA-256 dedup) and its description "
        "becomes FTS5-searchable. Optionally integrates the description into "
        "the parent fact's HRR vector for semantic retrieval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "fact_id": {"type": "integer", "description": "Fact ID to attach media to."},
            "file_path": {"type": "string", "description": "Absolute path to the media file on disk."},
            "mime_type": {"type": "string", "description": "MIME type, e.g. 'image/jpeg', 'audio/ogg', 'video/mp4'."},
            "description": {"type": "string", "description": "Text description for FTS5 search (required for searchability)."},
            "caption": {"type": "string", "description": "Optional human-readable caption."},
            "transcript": {"type": "string", "description": "Optional transcription text (for speech/screenshots)."},
        },
        "required": ["fact_id", "file_path", "mime_type"],
    },
}

MEDIA_DETACH_SCHEMA = {
    "name": "media_detach",
    "description": "Remove a media attachment from the database. Does NOT delete the file from disk.",
    "parameters": {
        "type": "object",
        "properties": {
            "media_id": {"type": "integer", "description": "Media attachment ID to remove."},
        },
        "required": ["media_id"],
    },
}

MEDIA_ORPHANS_SCHEMA = {
    "name": "media_orphans",
    "description": (
        "List media files stored on disk that have no corresponding database "
        "reference. Use this to identify files that can be safely cleaned up."
    ),
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
    },
}

MEDIA_CLEANUP_SCHEMA = {
    "name": "media_cleanup",
    "description": (
        "Remove orphaned media files from disk. Orphans are files in the "
        "media directory that have no corresponding database record. "
        "Use dry_run=True first to see what would be deleted, "
        "then dry_run=False to actually delete."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "dry_run": {
                "type": "boolean",
                "description": "If True, only report what would be deleted without removing anything",
                "default": True,
            },
        },
        "required": [],
    },
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

# Truncation limits for LLM extraction
_MAX_EXTRACT_CHARS = 3_000_000
_EXTRACT_HEAD_CHARS = 1_500_000
_EXTRACT_TAIL_CHARS = 1_498_000
_MAX_MSG_CHARS = 1000

# Trivial message patterns — skip LLM extraction for low-information content
_TRIVIAL_PATTERNS = re.compile(
    r'^(?:'
    r'ok|okay|好的|好|嗯|嗯嗯|嗯呢|是|是的|对|对的|可以|行|知道|明白了|收到|没问题'
    r'|thanks|thank you|thx|ty|tks|谢谢|多谢|感谢'
    r'|got it|gotcha|understood|理解|懂了|了解|明白'
    r'|yes|yep|yeah|yup|no|nope|nop|不是|不对'
    r'|hello|hi|hey|嗨|你好|您好|hi~|hello~'
    r'|👍|👌|😊|😄|😁|🙏|💪|❤️'
    r'|我也觉得|确实|不错|nice|great|good|好的吧|好吧|哈哈|呵呵|嘿嘿'
    r'|试一下|试试|先这样|就这样|差不多了'
    r')[ !~。！～,.?？\s]*$',
    re.IGNORECASE,
)

# Circuit breaker defaults
_CB_DEFAULT_MAX_FAILURES = 3
_CB_DEFAULT_COOLDOWN = 120  # seconds

# Reflection prompt — periodic meta-analysis of stored facts
_REFLECTION_SYSTEM_PROMPT = """You are a memory analysis assistant. Analyze the stored facts about a user and generate higher-level insights (meta-facts).

Given the existing facts, identify:
1. **Patterns**: Recurring preferences, habits, or behaviors
2. **Contradictions**: Facts that seem to conflict with each other
3. **Gaps**: Topics where more information would be valuable
4. **Evolution**: How preferences or decisions have changed over time

Rules:
- Be concrete and specific. Each meta-fact must be grounded in the actual facts.
- Skip obvious observations ("the user has multiple preferences").
- Return empty array if no meaningful insight can be drawn.

Return a JSON array of meta-fact objects, each with:
- "content": the insight statement (max 400 chars)
- "category": one of "user_pref", "project", "tool", "general"
- "tags": comma-separated tags
- "importance": integer 1-10 (how valuable is this insight?)
"""

_REFLECTION_FREQUENCY = 5  # Run reflection every N extraction cycles


def _load_plugin_config() -> dict:
    """Load butterfly-dream config.

    Priority:
    1. $HERMES_HOME/butterfly_config.yaml  (dedicated butterfly config)
    2. config.yaml → plugins.butterfly-dream  (legacy hermes config, backward compat)

    Returns flat dict of merged config values (dedicated wins on overlap).
    """
    from hermes_constants import get_hermes_home
    hermes_home = get_hermes_home()
    config = {}

    # 1. Try dedicated butterfly config file
    dedicated_path = hermes_home / "butterfly_config.yaml"
    if dedicated_path.exists():
        try:
            import yaml
            with open(dedicated_path, encoding="utf-8-sig") as f:
                config = yaml.safe_load(f) or {}
        except Exception:
            logger.debug("ButterflyDream: failed to load %s", dedicated_path)

    # 2. Legacy fallback: hermes config.yaml → plugins.butterfly-dream
    legacy_path = hermes_home / "config.yaml"
    if legacy_path.exists():
        try:
            import yaml
            with open(legacy_path, encoding="utf-8-sig") as f:
                all_config = yaml.safe_load(f) or {}
            legacy = cfg_get(all_config, "plugins", "butterfly-dream", default={}) or {}
            # Merge: dedicated values override legacy for same keys,
            # but legacy-only keys are preserved
            merged = dict(legacy)
            merged.update(config)
            config = merged
        except Exception:
            pass

    return config


# ---------------------------------------------------------------------------
# Dedicated butterfly debug logger (writes to $HERMES_HOME/logs/butterfly.log)
# ---------------------------------------------------------------------------

_butterfly_logger: logging.Logger | None = None


def _init_butterfly_logger(log_dir: str) -> logging.Logger:
    """Initialize the dedicated butterfly debug logger.

    Writes DEBUG+ to ``<log_dir>/butterfly.log`` with a clean
    timestamp|LEVEL|message format, completely separate from Hermes logging.
    Safe to call multiple times — only the first call sets up the handler.
    """
    global _butterfly_logger
    if _butterfly_logger is not None:
        return _butterfly_logger

    _butterfly_logger = logging.getLogger("butterfly")
    _butterfly_logger.propagate = False

    try:
        from pathlib import Path
        log_path = Path(log_dir) / "butterfly.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_path), encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "%(asctime)s|%(levelname)s|%(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        fh.setFormatter(fmt)
        _butterfly_logger.addHandler(fh)
        _butterfly_logger.setLevel(logging.DEBUG)
    except Exception:
        _butterfly_logger.addHandler(logging.NullHandler())
        _butterfly_logger.setLevel(logging.DEBUG)

    return _butterfly_logger


def _resolve_provider_credentials(provider: str) -> tuple[str, str]:
    """Resolve (base_url, api_key) for a given provider name."""
    prefix = provider.upper().replace("-", "_")
    api_key = os.environ.get(f"{prefix}_API_KEY", "")
    base_url = os.environ.get(f"{prefix}_BASE_URL", _DEFAULT_BASE_URLS.get(provider, ""))
    return base_url.rstrip("/"), api_key


def _detect_has_chinese(text: str) -> bool:
    """Check if text contains any CJK (Chinese/Japanese/Korean) characters."""
    import re
    return bool(re.search(r'[\u4e00-\u9fff\u3400-\u4dbf\uf900-\ufaff]', text))


def _call_extraction_llm(
    messages_text: str,
    provider: str,
    model: str,
    timeout: int = 90,
    system_prompt: str | None = None,
    dlog: logging.Logger | None = None,  # dedicated butterfly logger (butterfly.log)
) -> list[dict]:
    """Call the extraction LLM and return parsed fact objects with importance.

    Returns list of {"content", "category", "tags", "importance"}.
    Returns empty list on any error (fail-safe).

    Args:
        system_prompt: Optional override for the system prompt.
                       Defaults to _EXTRACTION_SYSTEM_PROMPT.
    """
    log = dlog or logger  # prefer butterfly.log, fallback to Hermes root
    import time as _time

    base_url, api_key = _resolve_provider_credentials(provider)
    if not api_key:
        log.warning("ButterflyDream LLM extract: no API key for '%s'", provider)
        return []
    if not base_url:
        log.warning("ButterflyDream LLM extract: no base URL for '%s'", provider)
        return []
    if not model:
        log.warning("ButterflyDream LLM extract: no model specified")
        return []

    # Detect conversation language and reinforce output language
    base_prompt = system_prompt or _EXTRACTION_SYSTEM_PROMPT
    if _detect_has_chinese(messages_text):
        lang_hint = "\n\nCRITICAL: The conversation contains Chinese (中文). You MUST output all extracted facts in Chinese (中文), matching the conversation's primary language."
    else:
        lang_hint = "\n\nCRITICAL: The conversation is in English. You MUST output all extracted facts in English, matching the conversation's language."

    url = f"{base_url}/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": base_prompt + lang_hint},
            {"role": "user", "content": f"Conversation turns:\n\n{messages_text}\n\n=== STEP 1: EXTRACT EVERY FACT ===\nRead through each turn. For EVERY concrete detail, write a separate standalone fact.\nDo NOT merge, do NOT filter, do NOT skip. Extract ALL specific information.\n\n=== STEP 2: DEDUPLICATE ===\nNow review your facts. If two facts say the EXACT same thing, keep one.\nIf two facts are about different aspects of the same topic, KEEP BOTH.\n\n=== STEP 3: FORMAT ===\nAdd category, importance, entities, tags, is_persistent, content_date to each fact.\n\n=== STEP 4: FINAL CHECKS (8) ===\n1. COVERAGE: Re-read the conversation. Is EVERY concrete detail extracted?\n2. SPECIFICITY: Entity names must be specific — \"a bonsai tree\" not \"an item\"\n3. NUMBERS: All numbers preserved exactly — \"3 kids\" not \"several kids\"\n4. DATES: content_date must be the EVENT's date, not the conversation date. For \"last week\", \"a few days ago\", \"yesterday\" — compute the actual date from the session date. Never leave relative dates unresolved.\n5. RELATIONS: \"X suggested Y to Z\" relations must be preserved — who did/said what to whom\n6. NAMES: No fact may contain \"User\", \"Assistant\", \"Participant 1\", \"Participant 2\", or any generic role label. Every entity name must be a real name from the conversation content.\n7. SUBJECT: Every fact must contain a person's name (Evan, Sam, etc.) as subject or explicit owner. Avoid \"The\"-starting facts — rewrite \"The road trip included...\" to \"Evan's road trip included...\" or \"The painting instructor\" to \"Evan's painting instructor\".\n8. JSON: Valid JSON with no trailing commas, proper escaping\n\nOutput ONLY the JSON array, no extra text."},
        ],
        "temperature": 0.2,
        "max_tokens": 16384,
    }
    # Some providers support response_format for guaranteed JSON output,
    # but not all (e.g. owl-alpha). Only add it for known-good providers.
    if provider in ("openai", "deepseek"):
        payload["response_format"] = {"type": "json_object"}

    body = json.dumps(payload).encode("utf-8")

    # Retry logic with exponential backoff
    max_retries = 4
    backoff_delays = [5, 10, 15, 20]  # seconds
    parsed = None

    for attempt in range(max_retries):
        req = urllib.request.Request(
            url, data=body,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                response_data = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, OSError, json.JSONDecodeError) as e:
            log.warning("ButterflyDream LLM extract request failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                delay = backoff_delays[attempt]
                log.info("ButterflyDream LLM extract: retrying in %ds...", delay)
                _time.sleep(delay)
                continue
            return []

        try:
            content = response_data["choices"][0]["message"]["content"]
            # Strip markdown code fences (```json ... ``` or ``` ... ```)
            content = content.strip()
            if content.startswith("```"):
                # Remove first line (```json or ```)
                first_nl = content.index("\n")
                content = content[first_nl + 1:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            parsed = json.loads(content)
            # Success - break out of retry loop
            break
        except (KeyError, IndexError, json.JSONDecodeError, ValueError) as e:
            log.warning("ButterflyDream LLM extract: parse failed (attempt %d/%d): %s", attempt + 1, max_retries, e)
            if attempt < max_retries - 1:
                delay = backoff_delays[attempt]
                log.info("ButterflyDream LLM extract: retrying in %ds...", delay)
                _time.sleep(delay)
                continue
            return []

    if parsed is None:
        return []

    if isinstance(parsed, dict):
        # Try common wrapper keys
        for key in ("facts", "memories", "extractions", "results", "insights",
                     "patterns", "data", "items", "output", "response", "content"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break
        # If still a dict, try the first list-valued key
        if isinstance(parsed, dict):
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
                # Handle nested dicts (e.g. {"fact_1": {...}, "fact_2": {...}})
                if isinstance(v, dict):
                    parsed = list(parsed.values())
                    break
        # Single-fact dict: {"content": "...", "category": "...", ...}
        # Wrap in a list so downstream validation picks it up
        if isinstance(parsed, dict) and "content" in parsed:
            parsed = [parsed]

    if isinstance(parsed, list):
        pass  # good
    elif isinstance(parsed, dict):
        log.debug("LLM extract dict keys: %s", list(parsed.keys())[:5])
        log.debug("LLM extract dict sample: %s", str(parsed)[:200])
        log.warning("ButterflyDream LLM extract: unexpected format: %s", type(parsed).__name__)
        return []

    # Validate and normalize
    facts = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if len(content) < 10:
            continue
        content = content[:400]
        category = str(item.get("category", "general")).strip()
        _VALID_CATEGORIES = {
            "place", "time", "person", "event", "activity",
            "identity", "preference", "goal",
            "project", "tool", "possession", "state", "opinion", "general",
        }
        if category not in _VALID_CATEGORIES:
            category = "general"
        tags = str(item.get("tags", "")).strip()
        importance = int(item.get("importance", 5))
        importance = max(1, min(10, importance))
        raw_persistent = item.get("is_persistent", False)
        if isinstance(raw_persistent, str):
            is_persistent = raw_persistent.lower() in ("true", "1", "yes")
        else:
            is_persistent = bool(raw_persistent)
        # Extract content_date (LLM returns ISO date or null)
        content_date = item.get("content_date")
        if content_date is not None:
            content_date = str(content_date).strip() or None
        # Extract entities from LLM output (if provided)
        entities = item.get("entities")
        if entities is not None and not isinstance(entities, list):
            entities = None  # ignore malformed

        facts.append({
            "content": content,
            "category": category,
            "tags": tags,
            "importance": importance,
            "is_persistent": is_persistent,
            "content_date": content_date,
            "entities": entities,
        })

    return facts


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class ButterflyDreamMemoryProvider(MemoryProvider):
    """Three-dimensional memory: Relevance × Recency × Importance.

    Builds on Holographic's fact store and HRR vector encoding, adding:
    - LLM-assigned importance scoring during extraction
    - Exponential recency decay with configurable half-life
    - Scenario-aware weight presets for retrieval
    - Entity relationship graph for multi-hop reasoning
    """

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store: Optional[MemoryStore] = None
        self._retriever: Optional[ThreeDimRetriever] = None
        self._min_trust = float(self._config.get("min_trust_threshold", 0.3))

        # LLM extraction config
        llm_cfg = self._config.get("extraction_model", {})
        self._extraction_provider = str(llm_cfg.get("provider", "deepseek"))
        self._extraction_model = str(llm_cfg.get("model", "deepseek-v4-flash"))

        # Extraction state
        self._llm_extract_enabled = self._config.get("llm_extract", False)
        self._last_extracted_idx = 0

        # Trivial message filter (Supermemory / ByteRover pattern)
        self._trivial_filter_enabled = self._config.get("trivial_filter", True)

        # Circuit breaker state (Mem0 pattern)
        cb_cfg = self._config.get("circuit_breaker", {})
        self._cb_max_failures = int(cb_cfg.get("max_failures", _CB_DEFAULT_MAX_FAILURES))
        self._cb_cooldown = int(cb_cfg.get("cooldown_seconds", _CB_DEFAULT_COOLDOWN))
        self._extraction_failures = 0
        self._cooldown_until = 0.0

        # Reflection state (GenAgents pattern)
        self._reflection_enabled = self._config.get("reflection", True)
        self._extraction_count = 0

        # Session date for extraction context (None = auto-detect datetime.now)
        self._session_date: Optional[str] = None

        # Turn-based sync_turn extraction
        self._extract_interval = int(self._config.get("extract_interval", 20))
        self._turn_counter = 0

        # Debug logging toggle (butterfly config, not hermes global)
        self._debug_logging = self._config.get("debug_logging", False)

        # Prefetch limit: how many facts to inject into system prompt per turn
        self._prefetch_limit = int(self._config.get("prefetch_limit", 10))

        # Thread safety for async extraction state
        self._extraction_lock = threading.Lock()
        # Track async extraction threads for safe shutdown
        self._extract_threads: list[threading.Thread] = []

    @property
    def name(self) -> str:
        return "butterfly-dream"

    def is_available(self) -> bool:
        return True

    def save_config(self, values, hermes_home):
        """Write config to config.yaml under plugins.butterfly-dream."""
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["butterfly-dream"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass

    def get_config_schema(self):
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/memories/butterfly_memory.db"
        return [
            {"key": "db_path", "description": "SQLite database path", "default": _default_db},
            {"key": "llm_extract", "description": "LLM-based fact extraction with importance scoring", "default": "true", "choices": ["true", "false"]},
            {"key": "default_trust", "description": "Default trust score for new facts", "default": "0.5"},
            {"key": "min_trust_threshold", "description": "Minimum trust threshold for retrieval", "default": "0.3"},
            {"key": "recency_half_life_days", "description": "Days for recency score to decay by half", "default": "30"},
            {"key": "hrr_dim", "description": "HRR vector dimensions", "default": "1024"},
            {"key": "extract_interval", "description": "Extract facts every N turns via sync_turn (0=disable)", "default": "20"},
            {"key": "debug_logging", "description": "Enable debug logs for search/query pipeline", "default": "false", "choices": ["true", "false"]},
            {"key": "prefetch_limit", "description": "Number of facts to inject into system prompt each turn", "default": "10"},
            {"key": "compression", "description": "Media compression settings (YAML block: enabled, image.quality, video.bitrate, etc.)", "default": "{enabled: true}"},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        # Clean up previous instance if re-initializing (e.g. session restart)
        if self._store:
            self.shutdown()

        # Prefer kwargs hermes_home for proper profile isolation, fall back to
        # get_hermes_home() for backwards compatibility.
        _hermes_home = str(kwargs.get("hermes_home")) if kwargs.get("hermes_home") else str(get_hermes_home())
        _default_db = _hermes_home + "/memories/butterfly_memory.db"
        db_path = self._config.get("db_path", _default_db)
        # Expand $HERMES_HOME
        if isinstance(db_path, str):
            db_path = db_path.replace("$HERMES_HOME", _hermes_home)
            db_path = db_path.replace("${HERMES_HOME}", _hermes_home)

        default_trust = float(self._config.get("default_trust", 0.5))
        hrr_dim = int(self._config.get("hrr_dim", 1024))
        half_life = float(self._config.get("recency_half_life_days", 30.0))

        # Retrieval weights from config
        ret_cfg = self._config.get("retrieval", {})
        rel_w = float(ret_cfg.get("relevance_weight", 0.4))
        rec_w = float(ret_cfg.get("recency_weight", 0.3))
        imp_w = float(ret_cfg.get("importance_weight", 0.3))

        # Normalize weights to sum to 1.0
        total = rel_w + rec_w + imp_w
        if total > 0:
            rel_w /= total
            rec_w /= total
            imp_w /= total
        custom_weights = {
            "relevance": rel_w,
            "recency": rec_w,
            "importance": imp_w,
        }

        self._store = MemoryStore(
            db_path=db_path,
            default_trust=default_trust,
            hrr_dim=hrr_dim,
            compression_config=self._config.get("compression", None),
        )
        # Initialize dedicated butterfly debug logger
        self._dlog = _init_butterfly_logger(_hermes_home + "/logs")
        self._retriever = ThreeDimRetriever(
            store=self._store,
            half_life_days=half_life,
            hrr_dim=hrr_dim,
            custom_weights=custom_weights,
            debug_logging=self._config.get("debug_logging", False),
            dlog=self._dlog,
        )
        self._session_id = session_id
        self._last_extracted_idx = 0
        self._turn_counter = 0

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        try:
            total = self._store.count_facts()
        except Exception:
            total = 0
        if total == 0:
            return (
                "# 🦋 Butterfly Dream Memory\n"
                "Active. Empty fact store — proactively add facts the user would expect you to remember.\n"
                "Use fact_store(action='add') to store durable facts with three-dimensional scoring.\n"
                "Use fact_feedback to rate facts after using them (trains trust + importance)."
            )
        return (
            f"# 🦋 Butterfly Dream Memory\n"
            f"Active. {total} facts stored with 3D scoring (Relevance × Recency × Importance).\n"
            f"Use fact_store to search, probe entities, reason across entities, or add facts.\n"
            f"Use fact_feedback to rate facts after using them (trains trust scores)."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Prefetch relevant facts before agent processes a user message."""
        if not self._retriever or not query:
            return ""
        try:
            results = self._retriever.search(
                query, min_trust=self._min_trust, limit=self._prefetch_limit, scenario="balanced",
            )
            if not results:
                return ""

            # Look up entity labels for retrieved facts (same format as eval)
            fids = tuple(r["fact_id"] for r in results if r.get("fact_id"))
            fact_entities: dict[int, str] = {}
            if fids and self._store:
                placeholders = ",".join("?" * len(fids))
                entity_rows = self._store.execute_query(
                    f"""SELECT fe.fact_id, GROUP_CONCAT(e.name, ', ') as entities
                        FROM fact_entities fe
                        JOIN entities e ON fe.entity_id = e.entity_id
                        WHERE fe.fact_id IN ({placeholders})
                        GROUP BY fe.fact_id""",
                    fids,
                )
                fact_entities = {r["fact_id"]: r["entities"] for r in entity_rows}

            lines = []
            for r in results:
                content = r.get("content", "")
                if not content:
                    continue
                fid = r.get("fact_id")
                ents = fact_entities.get(fid, "")
                entity_tag = f"[{ents}] " if ents else ""
                date = r.get("content_date", "")
                if date:
                    lines.append(f"{entity_tag}[{date}] {content}")
                else:
                    lines.append(f"{entity_tag}{content}")

            if self._debug_logging:
                self._dlog.debug(
                    "prefetch: %d facts for query='%.80s'",
                    len(lines), query,
                )
            return "## 🦋 Butterfly Dream Memory\n" + "\n".join(lines)
        except Exception as e:
            self._dlog.debug("prefetch failed: %s", e)
            return ""

    def _start_extract_thread(self, target, name):
        """Safely start and track a daemon extraction thread.

        Starts the thread first so a failed start() won't leave a zombie
        entry in _extract_threads. Prunes finished threads under lock to
        prevent unbounded list growth.
        """
        t = threading.Thread(target=target, daemon=True, name=name)
        t.start()
        with self._extraction_lock:
            self._extract_threads.append(t)
            self._extract_threads[:] = [t for t in self._extract_threads if t.is_alive()]

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", messages: list | None = None) -> None:
        """Batch extraction every N turns (config: extract_interval, default 20).

        Runs async extraction on unprocessed messages when the turn counter
        reaches the interval boundary. Designed for high-volume conversations
        where on_pre_compress/on_session_end alone would miss too much.
        """
        if not self._llm_extract_enabled or not self._store or not messages:
            return

        # Read+write _last_extracted_idx under a single lock to keep
        # state consistent with on_pre_compress / on_session_end.
        with self._extraction_lock:
            if self._extract_interval <= 0:
                return
            self._turn_counter += 1
            if self._turn_counter % self._extract_interval != 0:
                return
            new_msgs = messages[self._last_extracted_idx:]
            if len(new_msgs) < 2:
                return
            self._last_extracted_idx = max(self._last_extracted_idx, len(messages))

        msgs_copy = list(new_msgs)

        def _extract_async():
            try:
                facts = self._run_llm_extraction(msgs_copy)
                if facts:
                    logger.info("ButterflyDream sync_turn extracted %d facts (interval=%d)",
                                len(facts), self._extract_interval)
                    self._dlog.info("sync_turn: extracted %d facts (interval=%d)",
                                    len(facts), self._extract_interval)
            except Exception as e:
                self._dlog.debug("sync_turn extraction failed: %s", e)

        self._start_extract_thread(_extract_async, "butterfly-sync")

    def on_pre_compress(self, messages: list) -> str:
        """Extract facts before context compression discards messages — includes importance scoring.

        Returns "" because Butterfly Dream stores facts into its own DB rather
        than contributing text to the compression summary prompt. The extracted
        facts are available to the agent via normal 3D retrieval on subsequent turns.
        """
        if not self._llm_extract_enabled or not self._store or not messages:
            return ""
        # Mark consumed before async thread; under same lock as sync_turn/on_session_end
        # to keep _last_extracted_idx consistent.
        with self._extraction_lock:
            self._last_extracted_idx = max(self._last_extracted_idx, len(messages))
        msgs_copy = list(messages)

        def _extract_async():
            try:
                facts = self._run_llm_extraction(msgs_copy)
                if facts:
                    logger.info("ButterflyDream pre-compress extracted %d facts", len(facts))
                    self._dlog.info("pre-compress: extracted %d facts", len(facts))
            except Exception as e:
                self._dlog.debug("pre-compress extraction failed: %s", e)

        self._start_extract_thread(_extract_async, "butterfly-compress")
        return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA,
                MEDIA_ATTACH_SCHEMA, MEDIA_DETACH_SCHEMA, MEDIA_ORPHANS_SCHEMA,
                MEDIA_CLEANUP_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "fact_store":
            return self._handle_fact_store(args)
        elif tool_name == "fact_feedback":
            return self._handle_fact_feedback(args)
        elif tool_name == "media_attach":
            return self._handle_media_attach(args)
        elif tool_name == "media_detach":
            return self._handle_media_detach(args)
        elif tool_name == "media_orphans":
            return self._handle_media_orphans(args)
        elif tool_name == "media_cleanup":
            return self._handle_media_cleanup(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Final extraction at session end with importance scoring (async)."""
        if not self._store or not messages:
            return
        if self._llm_extract_enabled:
            # Read+write _last_extracted_idx under a single lock
            with self._extraction_lock:
                new_msgs = messages[self._last_extracted_idx:]
                if new_msgs:
                    self._last_extracted_idx = max(self._last_extracted_idx, len(messages))
            if new_msgs:
                msgs_copy = list(new_msgs)

                def _extract_async():
                    try:
                        facts = self._run_llm_extraction(msgs_copy)
                        if facts:
                            logger.info("ButterflyDream session-end extracted %d facts", len(facts))
                            self._dlog.info("session-end: extracted %d facts", len(facts))
                        # Reflection: check after extraction for accurate count
                        if self._reflection_enabled:
                            with self._extraction_lock:
                                count = self._extraction_count
                            if count > 0 and count % _REFLECTION_FREQUENCY == 0:
                                try:
                                    self._run_reflection()
                                except Exception as e:
                                    self._dlog.debug("reflection failed: %s", e)
                    except Exception as e:
                        self._dlog.debug("session-end extraction failed: %s", e)

                self._start_extract_thread(_extract_async, "butterfly-close")

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        **kwargs,
    ) -> None:
        """Reset extraction counters on session switch.

        On /reset or /new (reset=True), zero out both the turn counter and
        the extraction index so the next sync_turn boundary starts fresh.
        On /resume or /branch (reset=False), keep counters — the same logical
        conversation continues under a new session id.
        """
        if reset:
            with self._extraction_lock:
                self._turn_counter = 0
                self._last_extracted_idx = 0
            logger.debug("ButterflyDream session_switch(reset=True): counters reset")
            self._dlog.debug("session_switch: counters reset")

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes as facts with default importance."""
        if action == "add" and self._store and content:
            try:
                category = "user_pref" if target == "user" else "general"
                # Importance: user profile writes get higher default (7), general gets 5
                importance = 7 if target == "user" else 5
                # User profile writes (USER.md) are cross-session by nature
                is_persistent = target == "user"
                self._store.add_fact(content, category=category, importance=importance,
                                     is_persistent=is_persistent, dedup_threshold=0.7)
            except Exception as e:
                logger.debug("ButterflyDream memory_write mirror failed: %s", e)
                self._dlog.debug("memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        # Wait for async extraction threads to finish (they may be writing to the store)
        with self._extraction_lock:
            threads = list(self._extract_threads)
            self._extract_threads.clear()
        for t in threads:
            if t.is_alive():
                t.join(timeout=5)
        if self._store:
            self._store.close()
        self._store = None
        self._retriever = None

    # -- LLM extraction (enhanced with importance) -----------------------------

    def _run_llm_extraction(self, messages: list) -> list[dict]:
        """Extract facts with importance scoring via LLM.

        Returns list of stored facts (with fact_id).
        """
        # Circuit breaker: skip if in cooldown
        if not self._circuit_breaker_ok():
            self._dlog.debug("ButterflyDream: extraction skipped (circuit breaker cooldown)")
            return []

        lines = []
        
        # Session date header: helps LLM resolve relative dates ("yesterday", "last week")
        from datetime import datetime as _dt
        session_date_str = self._session_date or _dt.now().strftime("%Y-%m-%d")
        lines.append(f"[Session date: {session_date_str}]")
        
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content.strip()) < 10:
                continue
            # Trivial message filter: skip "ok", "thanks", etc.
            if self._trivial_filter_enabled and self._is_trivial_content(content.strip()):
                continue
            if role in ("user", "assistant"):
                # Use neutral labels to avoid LLM picking up "User"/"Assistant" as entity names.
                # The LLM must infer actual names (Evan, Sam, etc.) from conversation content.
                label_num = 1 if role == "user" else 2
                lines.append(f"[Participant {label_num}]: {content[:_MAX_MSG_CHARS]}")

        if len(lines) < 2:
            return []

        text = "\n\n".join(lines)
        if len(text) > _MAX_EXTRACT_CHARS:
            head = text[:_EXTRACT_HEAD_CHARS]
            tail = text[-_EXTRACT_TAIL_CHARS:]
            text = head + "\n\n... [truncated] ...\n\n" + tail

        facts = _call_extraction_llm(
            messages_text=text,
            provider=self._extraction_provider,
            model=self._extraction_model,
            dlog=self._dlog,
        )

        # Debug: log raw LLM output
        if facts:
            self._dlog.debug("ButterflyDream: LLM extracted %d raw facts", len(facts))
            for i, f in enumerate(facts):
                self._dlog.debug("  raw[%d] content='%.80s' cat=%s imp=%s date=%s",
                             i, f.get("content", ""), f.get("category", ""),
                             f.get("importance", "?"), f.get("content_date", "?"))
        else:
            self._dlog.debug("ButterflyDream: LLM returned 0 facts")

        # Circuit breaker: track result
        success = bool(facts)
        self._mark_extraction_result(success)

        if not facts:
            return []

        stored = []
        for fact in facts:
            try:
                result = self._store.add_fact(
                    content=fact["content"],
                    category=fact.get("category", "general"),
                    tags=fact.get("tags", ""),
                    importance=fact.get("importance", 5),
                    entities=fact.get("entities", None),
                    is_persistent=fact.get("is_persistent", False),
                    dedup_threshold=0.7,
                    content_date=_normalize_date(fact.get("content_date")) or _extract_date_from_content(fact.get("content", "")),
                )
                stored.append(result)
                # Debug: log storage result
                is_new = "inserted" if result.get("fact_id") and not result.get("merged") else \
                         "merged" if result.get("merge_type") else "existing"
                self._dlog.debug("  stored[%d] fact_id=%d %s: '%.80s'",
                             len(stored) - 1, result.get("fact_id", -1), is_new, result.get("content", ""))
            except Exception as e:
                self._dlog.debug("ButterflyDream store fact failed for '%.60s': %s", fact.get("content", ""), e)

        # Track extraction count for reflection trigger
        if stored:
            with self._extraction_lock:
                self._extraction_count += 1

        return stored

    # -- Helper methods (trivial filter, circuit breaker, reflection) -----------

    @staticmethod
    def _is_trivial_content(content: str) -> bool:
        """Check if a message is trivial (greeting, acknowledgment, etc.).

        Uses regex patterns for both English and Chinese trivial phrases.
        Returns True for messages that don't warrant memory extraction.
        """
        return bool(_TRIVIAL_PATTERNS.match(content.strip()))

    def _circuit_breaker_ok(self) -> bool:
        """Check if extraction can proceed, or if circuit breaker is active.

        Returns True if extraction should proceed.
        If cooldown has expired, resets the failure counter and allows through.
        """
        with self._extraction_lock:
            if self._extraction_failures < self._cb_max_failures:
                return True
            now = time.time()
            if now >= self._cooldown_until:
                self._extraction_failures = 0
                self._cooldown_until = 0.0
                logger.info("ButterflyDream circuit breaker: cooldown expired, resetting")
                self._dlog.info("circuit breaker: cooldown expired, resetting")
                return True
            return False

    def _mark_extraction_result(self, success: bool) -> None:
        """Record extraction outcome for circuit breaker tracking."""
        with self._extraction_lock:
            if success:
                self._extraction_failures = 0
                self._cooldown_until = 0.0
            else:
                self._extraction_failures += 1
                if self._extraction_failures >= self._cb_max_failures:
                    now = time.time()
                    self._cooldown_until = now + self._cb_cooldown
                    logger.warning(
                        "ButterflyDream circuit breaker: %d consecutive failures, "
                        "cooling down for %ds",
                        self._extraction_failures, self._cb_cooldown,
                    )
                    self._dlog.warning(
                        "circuit breaker: %d consecutive failures, cooling down for %ds",
                        self._extraction_failures, self._cb_cooldown,
                    )

    def _run_reflection(self) -> None:
        """LLM meta-analysis of stored facts → generate pattern insights.

        Fetches all stored facts (up to 100), sends to LLM for analysis,
        stores returned meta-facts as regular facts with high importance.
        Triggers every _REFLECTION_FREQUENCY extractions.
        """
        if not self._store:
            return
        # Respect circuit breaker: don't call LLM during cooldown
        if not self._circuit_breaker_ok():
            logger.debug("ButterflyDream reflection: skipped (circuit breaker cooldown)")
            self._dlog.debug("reflection: skipped (circuit breaker cooldown)")
            return
        try:
            all_facts = self._store.list_facts(limit=100)
        except Exception:
            return
        if not all_facts:
            return

        # Build fact summary for LLM
        fact_lines = []
        for f in all_facts:
            content = f.get("content", "")
            cat = f.get("category", "")
            imp = f.get("importance", 5)
            if content:
                fact_lines.append(f"[{cat}|imp={imp}] {content[:200]}")
        if len(fact_lines) < 3:
            return  # Not enough facts to reflect on

        fact_text = "\n".join(fact_lines)

        # Call LLM for reflection
        meta_facts = _call_extraction_llm(
            messages_text="Analyze these stored facts for patterns and insights:\n\n" + fact_text,
            provider=self._extraction_provider,
            model=self._extraction_model,
            system_prompt=_REFLECTION_SYSTEM_PROMPT,
            dlog=self._dlog,
        )

        # Track result in circuit breaker (shared with extraction)
        success = bool(meta_facts)
        self._mark_extraction_result(success)

        if not meta_facts:
            self._dlog.debug("reflection: no insights generated")
            return

        stored = 0
        for fact in meta_facts:
            try:
                # Reflection insights get higher base importance
                base_imp = fact.get("importance", 7)
                if base_imp < 5:
                    base_imp = 5  # Floor reflection importance
                self._store.add_fact(
                    content=fact["content"],
                    category=fact.get("category", "general"),
                    tags="reflection," + fact.get("tags", ""),
                    importance=base_imp,
                    entities=fact.get("entities", None),
                    is_persistent=True,
                    dedup_threshold=0.7,
                )
                stored += 1
            except Exception as e:
                self._dlog.debug("reflection store failed: %s", e)

        if stored:
            logger.info("ButterflyDream reflection: stored %d meta-facts", stored)
            self._dlog.info("reflection: stored %d meta-facts", stored)

    # -- Tool handlers ---------------------------------------------------------

    def _handle_fact_store(self, args: dict) -> str:
        action = args.get("action", "")
        try:
            if action == "add":
                return self._handle_add(args)
            elif action == "search":
                return self._handle_search(args)
            elif action == "probe":
                return self._handle_probe(args)
            elif action == "related":
                return self._handle_related(args)
            elif action == "reason":
                return self._handle_reason(args)
            elif action == "contradict":
                return self._handle_contradict(args)
            elif action == "timeline":
                return self._handle_timeline(args)
            elif action == "summarize":
                return self._handle_summarize(args)
            elif action == "update":
                return self._handle_update(args)
            elif action == "remove":
                return self._handle_remove(args)
            elif action == "list":
                return self._handle_list(args)
            else:
                return json.dumps({"error": f"Unknown action: {action}"})
        except Exception as e:
            logger.error("ButterflyDream fact_store error: %s", e, exc_info=True)
            self._dlog.error("fact_store error: %s", e, exc_info=True)
            return json.dumps({"error": str(e)})
    
    def _handle_add(self, args: dict) -> str:
        content = args.get("content", "").strip()
        if not content:
            return json.dumps({"error": "content is required"})
        category = args.get("category", "general")
        tags = args.get("tags", "")
        importance = args.get("importance", 5)
        if not isinstance(importance, int):
            try:
                importance = int(importance)
            except (ValueError, TypeError):
                importance = 5
        importance = max(1, min(10, importance))
        result = self._store.add_fact(content, category=category, tags=tags, importance=importance)
        return json.dumps(result)

    def _handle_search(self, args: dict) -> str:
        query = args.get("query", "").strip()
        if not query:
            return json.dumps({"error": "query is required"})
        min_trust = float(args.get("min_trust", self._min_trust))
        limit = int(args.get("limit", 10))
        scenario = args.get("scenario", "balanced")
        persistent_only = bool(args.get("persistent_only", False))
        results = self._retriever.search(
            query, min_trust=min_trust, limit=limit, scenario=scenario,
            persistent_only=persistent_only,
        )
        if self._debug_logging:
            self._dlog.debug(
                "handle_search: query='%.80s' limit=%d scenario=%s → %d results",
                query, limit, scenario, len(results),
            )
        return json.dumps(results, default=str)

    def _handle_probe(self, args: dict) -> str:
        entity = args.get("entity", "").strip()
        if not entity:
            return json.dumps({"error": "entity is required"})
        limit = int(args.get("limit", 20))
        facts = self._store.get_entity_facts(entity, limit=limit)
        if self._debug_logging:
            self._dlog.debug(
                "handle_probe: entity='%.60s' limit=%d → %d facts",
                entity, limit, len(facts),
            )
        return json.dumps(facts, default=str)

    def _handle_related(self, args: dict) -> str:
        entity = args.get("entity", "").strip()
        if not entity:
            return json.dumps({"error": "entity is required"})
        depth = int(args.get("depth", 2))
        relations = self._store.get_related_entities(entity, depth=depth)
        if self._debug_logging:
            self._dlog.debug(
                "handle_related: entity='%.60s' depth=%d → %d relations",
                entity, depth, len(relations),
            )
        return json.dumps(relations, default=str)

    def _handle_reason(self, args: dict) -> str:
        entities = args.get("entities", [])
        if not entities:
            return json.dumps({"error": "entities is required"})
        limit = int(args.get("limit", 10))
        # Gather facts shared by all specified entities
        shared = None
        for entity_name in entities[:5]:  # cap at 5 entities
            facts = self._store.get_entity_facts(entity_name, limit=50)
            fact_ids = {f.get("fact_id") for f in facts if f.get("fact_id")}
            if shared is None:
                shared = fact_ids
            else:
                shared &= fact_ids
        if not shared:
            if self._debug_logging:
                self._dlog.debug(
                    "handle_reason: entities=%s → 0 shared facts",
                    entities[:5],
                )
            return json.dumps([])
        # Fetch full facts
        results = []
        for fid in sorted(shared)[:limit]:
            fact = self._store.get_fact(fid)
            if fact:
                results.append(fact)
        if self._debug_logging:
            self._dlog.debug(
                "handle_reason: entities=%s → %d shared facts",
                entities[:5], len(results),
            )
        return json.dumps(results, default=str)
    def _handle_contradict(self, args: dict) -> str:
        """Find facts with conflicting claims (same entity, opposing content).

        Uses SQL to find entity-sharing fact pairs, then checks for
        contradiction heuristics on the result set (up to 200 pairs).
        """
        contradict_pairs = []
        # Find pairs of facts that share at least one entity
        pairs = self._store.execute_query(
            """SELECT e.name, f1.fact_id, f1.content, f2.fact_id, f2.content
               FROM entities e
               JOIN fact_entities fe1 ON e.entity_id = fe1.entity_id
               JOIN facts f1 ON fe1.fact_id = f1.fact_id
               JOIN fact_entities fe2 ON e.entity_id = fe2.entity_id AND fe2.fact_id > fe1.fact_id
               JOIN facts f2 ON fe2.fact_id = f2.fact_id
               WHERE f1.content < f2.content
               ORDER BY e.name
               LIMIT 200"""
        )
        for row in pairs:
            name = row[0]
            id1, c1, id2, c2 = row[1], row[2], row[3], row[4]
            if self._is_contradictory(c1, c2):
                contradict_pairs.append({
                    "entity": name,
                    "fact_id_a": id1,
                    "content_a": c1,
                    "fact_id_b": id2,
                    "content_b": c2,
                })
        if self._debug_logging:
            self._dlog.debug(
                "handle_contradict: %d pairs checked, %d contradictions found",
                len(pairs), len(contradict_pairs),
            )
        return json.dumps(contradict_pairs[:20], default=str)

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

    def _handle_update(self, args: dict) -> str:
        fact_id = args.get("fact_id")
        if fact_id is None:
            return json.dumps({"error": "fact_id is required"})
        try:
            fact_id = int(fact_id)
        except (ValueError, TypeError):
            return json.dumps({"error": "invalid fact_id"})
        kwargs = {}
        for key in ("content", "category", "tags", "importance"):
            if key in args:
                kwargs[key] = args[key]
        # trust_delta is a relative adjustment — resolve to absolute trust_score
        if "trust_delta" in args:
            current = self._store.get_fact(fact_id)
            if current:
                delta = float(args["trust_delta"])
                kwargs["trust_score"] = max(0.0, min(1.0, current["trust_score"] + delta))
            else:
                return json.dumps({"error": "fact not found", "fact_id": fact_id})
        if self._store.update_fact(fact_id, **kwargs):
            return json.dumps({"success": True, "fact_id": fact_id})
        return json.dumps({"error": "fact not found", "fact_id": fact_id})

    def _handle_remove(self, args: dict) -> str:
        fact_id = args.get("fact_id")
        if fact_id is None:
            return json.dumps({"error": "fact_id is required"})
        try:
            fact_id = int(fact_id)
        except (ValueError, TypeError):
            return json.dumps({"error": "invalid fact_id"})
        if self._store.remove_fact(fact_id):
            return json.dumps({"success": True, "fact_id": fact_id})
        return json.dumps({"error": "fact not found", "fact_id": fact_id})

    def _handle_list(self, args: dict) -> str:
        limit = int(args.get("limit", 50))
        offset = int(args.get("offset", 0))
        persistent_only = bool(args.get("persistent_only", False))
        facts = self._store.list_facts(limit=limit, offset=offset, persistent_only=persistent_only)
        return json.dumps(facts, default=str)

    def _handle_timeline(self, args: dict) -> str:
        """Return facts linked to an entity sorted chronologically (oldest first)."""
        entity = args.get("entity", "").strip()
        if not entity:
            return json.dumps({"error": "entity is required for timeline"})
        limit = int(args.get("limit", 20))
        min_importance = float(args.get("min_importance", 0))
        facts = self._store.get_entity_timeline(entity, limit=limit,
                                                min_importance=min_importance)
        if self._debug_logging:
            self._dlog.debug(
                "handle_timeline: entity='%.60s' limit=%d → %d facts",
                entity, limit, len(facts),
            )
        return json.dumps(facts, default=str)

    def _handle_summarize(self, args: dict) -> str:
        """Return a structured summary card for an entity."""
        entity = args.get("entity", "").strip()
        if not entity:
            return json.dumps({"error": "entity is required for summarize"})
        limit = int(args.get("limit", 50))
        summary = self._store.get_entity_summary(entity, limit=limit)
        if self._debug_logging:
            self._dlog.debug(
                "handle_summarize: entity='%.60s' limit=%d → %d fields",
                entity, limit, len(summary),
            )
        return json.dumps(summary, default=str)

    def _handle_fact_feedback(self, args: dict) -> str:
        action = args.get("action", "")
        fact_id = args.get("fact_id")
        if fact_id is None or action not in ("helpful", "unhelpful"):
            return json.dumps({"error": "fact_id and action (helpful/unhelpful) are required"})
        try:
            fact_id = int(fact_id)
        except (ValueError, TypeError):
            return json.dumps({"error": "invalid fact_id"})
        result = self._store.record_feedback(fact_id, helpful=(action == "helpful"))
        return json.dumps(result)

    def _handle_media_attach(self, args: dict) -> str:
        fact_id = args.get("fact_id")
        file_path = args.get("file_path", "").strip()
        mime_type = args.get("mime_type", "").strip()
        if not fact_id or not file_path or not mime_type:
            return json.dumps({"error": "fact_id, file_path, and mime_type are required"})
        try:
            fact_id = int(fact_id)
        except (ValueError, TypeError):
            return json.dumps({"error": "invalid fact_id"})
        if not os.path.isfile(file_path):
            return json.dumps({"error": f"file not found: {file_path}"})
        try:
            result = self._store.attach_media(
                fact_id=fact_id,
                source_path=file_path,
                mime_type=mime_type,
                description=args.get("description", ""),
                caption=args.get("caption", ""),
                transcript=args.get("transcript", ""),
            )
            return json.dumps(result, default=str)
        except (ValueError, FileNotFoundError) as e:
            return json.dumps({"error": str(e)})

    def _handle_media_detach(self, args: dict) -> str:
        media_id = args.get("media_id")
        if media_id is None:
            return json.dumps({"error": "media_id is required"})
        try:
            media_id = int(media_id)
        except (ValueError, TypeError):
            return json.dumps({"error": "invalid media_id"})
        if self._store.detach_media(media_id):
            return json.dumps({"success": True, "media_id": media_id})
        return json.dumps({"error": "media not found", "media_id": media_id})

    def _handle_media_orphans(self, args: dict) -> str:
        orphans = self._store.media_orphans()
        return json.dumps({"orphans": orphans, "count": len(orphans)}, default=str)

    def _handle_media_cleanup(self, args: dict) -> str:
        dry_run = args.get("dry_run", True)
        if not isinstance(dry_run, bool):
            dry_run = True
        result = self._store.media_cleanup(dry_run=dry_run)
        return json.dumps(result, default=str)


# ---------------------------------------------------------------------------
# Plugin registration
# ---------------------------------------------------------------------------


_plugin_instance: Optional[ButterflyDreamMemoryProvider] = None


def register(ctx) -> None:
    """Register the butterfly-dream memory provider with the plugin system."""
    global _plugin_instance
    if _plugin_instance is not None:
        logger.warning("ButterflyDream already registered, skipping")
        return
    config = _load_plugin_config()
    _plugin_instance = ButterflyDreamMemoryProvider(config=config)
    ctx.register_memory_provider(_plugin_instance)
