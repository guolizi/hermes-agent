"""hermes-memory-store — holographic memory plugin using MemoryProvider interface.

Registers as a MemoryProvider plugin, giving the agent structured fact storage
with entity resolution, trust scoring, and HRR-based compositional retrieval.

Original plugin by dusterbloom (PR #2351), adapted to the MemoryProvider ABC.

Config in $HERMES_HOME/config.yaml (profile-scoped):

  plugins:
    hermes-memory-store:
      db_path: $HERMES_HOME/memory_store.db   # omit to use the default
      auto_extract: false                       # legacy regex-based extraction
      llm_extract: false                        # LLM-based extraction (overrides auto_extract)
      default_trust: 0.5
      min_trust_threshold: 0.3
      temporal_decay_half_life: 0
      extraction_model:
        provider: deepseek                      # provider name (resolves API key from env)
        model: deepseek-v4-flash                # model for extraction (cheap models work well)
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import urllib.error
import urllib.request
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from .store import MemoryStore
from .retrieval import FactRetriever
from hermes_cli.config import cfg_get

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
    "ollama": "http://localhost:11434/v1",
}

# ---------------------------------------------------------------------------
# LLM extraction prompt
# ---------------------------------------------------------------------------

_EXTRACTION_SYSTEM_PROMPT = """You are a memory extraction assistant. Your task is to analyze conversation turns and extract facts worth remembering.

Extract facts about:
1. User preferences, habits, and personal information
2. Project decisions, architecture choices, and technical rationale
3. Tool configurations, setup steps, and environment details
4. Key conventions and agreements made during the conversation
5. Any other information that would be useful to remember across sessions

Rules:
- Only extract concrete, specific facts. Skip small talk and greetings.
- Prefer concise, self-contained statements.
- If nothing worth extracting, return an empty array.
- Deduplicate: don't extract the same fact multiple times.

Return a JSON array of objects, each with:
- "content": the fact statement (plain text, max 400 chars)
- "category": one of "user_pref", "project", "tool", "general"
- "tags": optional comma-separated tags

Example:
[
  {"content": "User prefers to use VS Code for Python development", "category": "user_pref", "tags": "editor,python"},
  {"content": "Project uses FastAPI with SQLAlchemy for backend", "category": "project", "tags": "backend,stack"},
  {"content": "Hermes gateway runs on port 8080 with 3 platforms connected", "category": "tool", "tags": "hermes,config"}
]"""


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

FACT_STORE_SCHEMA = {
    "name": "fact_store",
    "description": (
        "Deep structured memory with algebraic reasoning. "
        "Use alongside the memory tool — memory for always-on context, "
        "fact_store for deep recall and compositional queries.\n\n"
        "ACTIONS (simple → powerful):\n"
        "• add — Store a fact the user would expect you to remember.\n"
        "• search — Keyword lookup ('editor config', 'deploy process').\n"
        "• probe — Entity recall: ALL facts about a person/thing.\n"
        "• related — What connects to an entity? Structural adjacency.\n"
        "• reason — Compositional: facts connected to MULTIPLE entities simultaneously.\n"
        "• contradict — Memory hygiene: find facts making conflicting claims.\n"
        "• update/remove/list — CRUD operations.\n\n"
        "IMPORTANT: Before answering questions about the user, ALWAYS probe or reason first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "probe", "related", "reason", "contradict", "update", "remove", "list"],
            },
            "content": {"type": "string", "description": "Fact content (required for 'add')."},
            "query": {"type": "string", "description": "Search query (required for 'search')."},
            "entity": {"type": "string", "description": "Entity name for 'probe'/'related'."},
            "entities": {"type": "array", "items": {"type": "string"}, "description": "Entity names for 'reason'."},
            "fact_id": {"type": "integer", "description": "Fact ID for 'update'/'remove'."},
            "category": {"type": "string", "enum": ["user_pref", "project", "tool", "general"]},
            "tags": {"type": "string", "description": "Comma-separated tags."},
            "trust_delta": {"type": "number", "description": "Trust adjustment for 'update'."},
            "min_trust": {"type": "number", "description": "Minimum trust filter (default: 0.3)."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["action"],
    },
}

FACT_FEEDBACK_SCHEMA = {
    "name": "fact_feedback",
    "description": (
        "Rate a fact after using it. Mark 'helpful' if accurate, 'unhelpful' if outdated. "
        "This trains the memory — good facts rise, bad facts sink."
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


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def _load_plugin_config() -> dict:
    from hermes_constants import get_hermes_home
    config_path = get_hermes_home() / "config.yaml"
    if not config_path.exists():
        return {}
    try:
        import yaml
        with open(config_path, encoding="utf-8-sig") as f:
            all_config = yaml.safe_load(f) or {}
        return cfg_get(all_config, "plugins", "hermes-memory-store", default={}) or {}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# LLM extraction helpers
# ---------------------------------------------------------------------------

def _resolve_provider_credentials(provider: str) -> tuple[str, str]:
    """Resolve (base_url, api_key) for a given provider name.

    Reads from environment first, falls back to known defaults.
    Returns (base_url or "", api_key or "").
    """
    prefix = provider.upper().replace("-", "_")
    api_key = os.environ.get(f"{prefix}_API_KEY", "")
    base_url = os.environ.get(f"{prefix}_BASE_URL", _DEFAULT_BASE_URLS.get(provider, ""))
    return base_url.rstrip("/"), api_key


def _call_extraction_llm(
    messages_text: str,
    provider: str,
    model: str,
    timeout: int = 30,
) -> list[dict]:
    """Call the extraction LLM and return parsed fact objects.

    Returns list of {"content": str, "category": str, "tags": str}.
    Returns empty list on any error (fail-safe).
    """
    base_url, api_key = _resolve_provider_credentials(provider)
    if not api_key:
        logger.warning("Holographic LLM extract: no API key found for provider '%s'", provider)
        return []
    if not base_url:
        logger.warning("Holographic LLM extract: no base URL for provider '%s'", provider)
        return []
    if not model:
        logger.warning("Holographic LLM extract: no model specified")
        return []

    url = f"{base_url}/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract facts from these conversation turns:\n\n{messages_text}"},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.1,
        "max_tokens": 2048,
    }

    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=body,
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
        logger.warning("Holographic LLM extract request failed: %s", e)
        return []

    try:
        content = response_data["choices"][0]["message"]["content"]
        parsed = json.loads(content)
    except (KeyError, IndexError, json.JSONDecodeError) as e:
        logger.warning("Holographic LLM extract: failed to parse response: %s", e)
        return []

    if isinstance(parsed, dict):
        # Some providers wrap in {"facts": [...]}
        for key in ("facts", "memories", "extractions", "results"):
            if key in parsed and isinstance(parsed[key], list):
                parsed = parsed[key]
                break

    if not isinstance(parsed, list):
        logger.warning("Holographic LLM extract: unexpected response format: %s", type(parsed).__name__)
        return []

    # Validate and normalize each fact
    facts = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        content = str(item.get("content", "")).strip()
        if len(content) < 10:
            continue
        content = content[:400]
        category = str(item.get("category", "general")).strip()
        if category not in ("user_pref", "project", "tool", "general"):
            category = "general"
        tags = str(item.get("tags", "")).strip()
        facts.append({"content": content, "category": category, "tags": tags})

    return facts


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class HolographicMemoryProvider(MemoryProvider):
    """Holographic memory with structured facts, entity resolution, and HRR retrieval."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store = None
        self._retriever = None
        self._min_trust = float(self._config.get("min_trust_threshold", 0.3))

        # LLM extraction config
        llm_cfg = self._config.get("extraction_model", {})
        self._extraction_provider = str(llm_cfg.get("provider", "deepseek"))
        self._extraction_model = str(llm_cfg.get("model", "deepseek-v4-flash"))

        # Extraction state
        self._llm_extract_enabled = self._config.get("llm_extract", False)
        self._last_extracted_idx = 0  # index into conversation messages

    @property
    def name(self) -> str:
        return "holographic"

    def is_available(self) -> bool:
        return True  # SQLite is always available, numpy is optional

    def save_config(self, values, hermes_home):
        """Write config to config.yaml under plugins.hermes-memory-store."""
        from pathlib import Path
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            existing = {}
            if config_path.exists():
                with open(config_path, encoding="utf-8-sig") as f:
                    existing = yaml.safe_load(f) or {}
            existing.setdefault("plugins", {})
            existing["plugins"]["hermes-memory-store"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass

    def get_config_schema(self):
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/memory_store.db"
        return [
            {"key": "db_path", "description": "SQLite database path", "default": _default_db},
            {"key": "auto_extract", "description": "Auto-extract facts at session end and pre-compress (regex, with Chinese patterns)", "default": "false", "choices": ["true", "false"]},
            {"key": "llm_extract", "description": "LLM-based fact extraction (overrides auto_extract)", "default": "false", "choices": ["true", "false"]},
            {"key": "default_trust", "description": "Default trust score for new facts", "default": "0.5"},
            {"key": "hrr_dim", "description": "HRR vector dimensions", "default": "1024"},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        _hermes_home = str(get_hermes_home())
        _default_db = _hermes_home + "/memory_store.db"
        db_path = self._config.get("db_path", _default_db)
        # Expand $HERMES_HOME in user-supplied paths so config values like
        # "$HERMES_HOME/memory_store.db" or "~/.hermes/memory_store.db" both
        # resolve to the active profile's directory.
        if isinstance(db_path, str):
            db_path = db_path.replace("$HERMES_HOME", _hermes_home)
            db_path = db_path.replace("${HERMES_HOME}", _hermes_home)
        default_trust = float(self._config.get("default_trust", 0.5))
        hrr_dim = int(self._config.get("hrr_dim", 1024))
        hrr_weight = float(self._config.get("hrr_weight", 0.3))
        temporal_decay = int(self._config.get("temporal_decay_half_life", 0))

        self._store = MemoryStore(db_path=db_path, default_trust=default_trust, hrr_dim=hrr_dim)
        self._retriever = FactRetriever(
            store=self._store,
            temporal_decay_half_life=temporal_decay,
            hrr_weight=hrr_weight,
            hrr_dim=hrr_dim,
        )
        self._session_id = session_id
        self._last_extracted_idx = 0

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        try:
            total = self._store._conn.execute(
                "SELECT COUNT(*) FROM facts"
            ).fetchone()[0]
        except Exception:
            total = 0
        if total == 0:
            return (
                "# Holographic Memory\n"
                "Active. Empty fact store — proactively add facts the user would expect you to remember.\n"
                "Use fact_store(action='add') to store durable structured facts about people, projects, preferences, decisions.\n"
                "Use fact_feedback to rate facts after using them (trains trust scores)."
            )
        return (
            f"# Holographic Memory\n"
            f"Active. {total} facts stored with entity resolution and trust scoring.\n"
            f"Use fact_store to search, probe entities, reason across entities, or add facts.\n"
            f"Use fact_feedback to rate facts after using them (trains trust scores)."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._retriever or not query:
            return ""
        try:
            results = self._retriever.search(query, min_trust=self._min_trust, limit=5)
            if not results:
                return ""
            lines = []
            for r in results:
                trust = r.get("trust_score", r.get("trust", 0))
                lines.append(f"- [{trust:.1f}] {r.get('content', '')}")
            return "## Holographic Memory\n" + "\n".join(lines)
        except Exception as e:
            logger.debug("Holographic prefetch failed: %s", e)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", messages: list | None = None) -> None:
        """No-op: extraction only fires on pre-compress and session-end."""
        pass

    def on_pre_compress(self, messages: list) -> str:
        """Extract facts before context compression discards messages.

        Runs async (background thread) so it doesn't block compression.
        """
        if not self._llm_extract_enabled or not self._store or not messages:
            return ""

        # Fire-and-forget: don't block compression waiting for LLM
        msgs_copy = list(messages)

        def _extract_async():
            try:
                facts = self._run_llm_extraction(msgs_copy)
                if facts:
                    logger.info("Holographic pre-compress extracted %d facts", len(facts))
            except Exception as e:
                logger.debug("Holographic pre-compress async extraction failed: %s", e)

        t = threading.Thread(target=_extract_async, daemon=True, name="holographic-compress")
        t.start()
        return ""  # No text to inject into compression prompt

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        if tool_name == "fact_store":
            return self._handle_fact_store(args)
        elif tool_name == "fact_feedback":
            return self._handle_fact_feedback(args)
        return tool_error(f"Unknown tool: {tool_name}")

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """Final extraction at session end.

        Highest priority: ALWAYS extracts remaining messages, regardless of
        whether sync_turn or on_pre_compress already ran this session.
        """
        if not self._store or not messages:
            return
        if self._llm_extract_enabled:
            # LLM-based extraction for all unprocessed messages
            new_msgs = messages[self._last_extracted_idx:]
            if new_msgs:
                facts = self._run_llm_extraction(new_msgs)
                if facts:
                    logger.info("Holographic session-end LLM extracted %d facts", len(facts))
        elif self._auto_extract_enabled():
            # Legacy regex-based extraction
            self._auto_extract_facts(messages)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes as facts."""
        if action == "add" and self._store and content:
            try:
                category = "user_pref" if target == "user" else "general"
                self._store.add_fact(content, category=category)
            except Exception as e:
                logger.debug("Holographic memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        self._store = None
        self._retriever = None

    def _auto_extract_enabled(self) -> bool:
        value = self._config.get("auto_extract", False)
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    # -- LLM extraction --------------------------------------------------------

    def _run_llm_extraction(self, messages: list) -> list[dict]:
        """Extract facts from a list of conversation messages via LLM.

        Returns list of stored facts (with fact_id). Empty on error/ nothing to extract.
        """
        # Build compact text from messages
        lines = []
        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content.strip()) < 10:
                continue
            if role in ("user", "assistant"):
                label = "User" if role == "user" else "Assistant"
                lines.append(f"{label}: {content[:1000]}")

        if len(lines) < 2:
            return []

        text = "\n\n".join(lines)

        # Truncate to avoid excessive token usage (roughly 8K chars is safe)
        if len(text) > 24000:
            # Keep first and last messages
            head = text[:12000]
            tail = text[-10000:]
            text = head + "\n\n... [truncated] ...\n\n" + tail

        facts = _call_extraction_llm(
            messages_text=text,
            provider=self._extraction_provider,
            model=self._extraction_model,
        )

        if not facts:
            return []

        stored = []
        for fact in facts:
            try:
                fact_id = self._store.add_fact(
                    fact["content"],
                    category=fact.get("category", "general"),
                    tags=fact.get("tags", ""),
                )
                stored.append({"fact_id": fact_id, **fact})
            except Exception:
                # Duplicate (UNIQUE constraint) or other DB error — skip silently
                pass

        if stored:
            logger.info("Holographic LLM extracted %d new facts (from %d candidates)", len(stored), len(facts))
        return stored

    # -- Tool handlers -------------------------------------------------------

    def _handle_fact_store(self, args: dict) -> str:
        try:
            action = args["action"]
            store = self._store
            retriever = self._retriever

            if action == "add":
                fact_id = store.add_fact(
                    args["content"],
                    category=args.get("category", "general"),
                    tags=args.get("tags", ""),
                )
                return json.dumps({"fact_id": fact_id, "status": "added"})

            elif action == "search":
                results = retriever.search(
                    args["query"],
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", self._min_trust)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "probe":
                results = retriever.probe(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "related":
                results = retriever.related(
                    args["entity"],
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "reason":
                entities = args.get("entities", [])
                if not entities:
                    return tool_error("reason requires 'entities' list")
                results = retriever.reason(
                    entities,
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "contradict":
                results = retriever.contradict(
                    category=args.get("category"),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"results": results, "count": len(results)})

            elif action == "update":
                updated = store.update_fact(
                    int(args["fact_id"]),
                    content=args.get("content"),
                    trust_delta=float(args["trust_delta"]) if "trust_delta" in args else None,
                    tags=args.get("tags"),
                    category=args.get("category"),
                )
                return json.dumps({"updated": updated})

            elif action == "remove":
                removed = store.remove_fact(int(args["fact_id"]))
                return json.dumps({"removed": removed})

            elif action == "list":
                facts = store.list_facts(
                    category=args.get("category"),
                    min_trust=float(args.get("min_trust", 0.0)),
                    limit=int(args.get("limit", 10)),
                )
                return json.dumps({"facts": facts, "count": len(facts)})

            else:
                return tool_error(f"Unknown action: {action}")

        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def _handle_fact_feedback(self, args: dict) -> str:
        try:
            fact_id = int(args["fact_id"])
            helpful = args["action"] == "helpful"
            result = self._store.record_feedback(fact_id, helpful=helpful)
            return json.dumps(result)
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    # -- Legacy regex extraction (auto_extract mode) --------------------------

    def _auto_extract_facts(self, messages: list) -> None:
        _PREF_PATTERNS = [
            re.compile(r'\bI\s+(?:prefer|like|love|use|want|need)\s+(.+)', re.IGNORECASE),
            re.compile(r'\bmy\s+(?:favorite|preferred|default)\s+\w+\s+is\s+(.+)', re.IGNORECASE),
            re.compile(r'\bI\s+(?:always|never|usually)\s+(.+)', re.IGNORECASE),
            re.compile(r'(?:我|我的|以后|后续).{0,24}(?:喜欢|偏好|更喜欢|习惯|通常|默认|希望|想要|需要|不喜欢|讨厌).+'),
            re.compile(r'(?:我叫|我的名字是|我是|我主要|我负责|我从事).+'),
            re.compile(r'(?:请|帮我).{0,24}(?:默认|以后|后续|一直|优先).+'),
            re.compile(r'(?:不要|别|不需要|不想).+'),
        ]
        _PROJECT_PATTERNS = [
            re.compile(r'\bwe\s+(?:decided|agreed|chose)\s+(?:to\s+)?(.+)', re.IGNORECASE),
            re.compile(r'\bthe\s+project\s+(?:uses|needs|requires)\s+(.+)', re.IGNORECASE),
            re.compile(r'(?:项目|仓库|代码库|系统|服务|框架|Hermes|hermes|agent_assisstant).{0,40}(?:使用|采用|依赖|需要|要求|部署|运行|工作目录|入口|端口|配置|保存|存储).+'),
            re.compile(r'(?:决定|确定|约定|统一|以后|后续).{0,40}(?:使用|采用|保留|删除|不再|迁移|改成|放在|写入).+'),
            re.compile(r'.{0,24}(?:工作目录|部署目录|配置文件|记忆文件|数据库|服务名|端口).{0,24}(?:是|在|为|叫).+'),
        ]

        extracted = 0
        seen_norms = self._existing_fact_norms()
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if not isinstance(content, str) or len(content) < 5:
                continue

            for sentence in self._candidate_fact_sentences(content):
                category = None
                if any(pattern.search(sentence) for pattern in _PROJECT_PATTERNS):
                    category = "project"
                elif any(pattern.search(sentence) for pattern in _PREF_PATTERNS):
                    category = "user_pref"

                if category and self._add_extracted_fact(sentence, category, seen_norms):
                    extracted += 1

        if extracted:
            logger.info("Auto-extracted %d facts from conversation (regex)", extracted)

    def _candidate_fact_sentences(self, content: str) -> List[str]:
        sentences: List[str] = []
        for line in re.split(r'\n+', content):
            for match in re.finditer(r'[^。！？!?；;]+[。！？!?；;]?', line):
                sentence = re.sub(r'\s+', ' ', match.group(0).strip())
                if len(sentence) < 5 or self._looks_like_question(sentence):
                    continue
                sentences.append(sentence)
        return sentences

    def _looks_like_question(self, sentence: str) -> bool:
        stripped = sentence.strip()
        if stripped.endswith(("?", "？")):
            return True
        return any(marker in stripped for marker in (
            "什么", "为什么", "怎么", "如何", "哪里", "哪儿", "是否", "是不是", "吗",
        ))

    def _add_extracted_fact(self, content: str, category: str, seen_norms: set[str]) -> bool:
        if not self._store:
            return False
        content = re.sub(r'\s+', ' ', content).strip()
        content = content.strip('。！？!?；;，,、')
        if len(content) < 5:
            return False

        norm = self._normalize_fact_text(content)
        if not norm or norm in seen_norms:
            return False

        try:
            self._store.add_fact(content[:400], category=category)
            seen_norms.add(norm)
            return True
        except Exception as e:
            logger.debug("Holographic auto-extract add failed: %s", e)
            return False

    def _existing_fact_norms(self) -> set[str]:
        if not self._store:
            return set()
        try:
            rows = self._store._conn.execute("SELECT content FROM facts").fetchall()
            return {self._normalize_fact_text(row["content"]) for row in rows}
        except Exception as e:
            logger.debug("Holographic auto-extract dedupe load failed: %s", e)
            return set()

    def _normalize_fact_text(self, content: str) -> str:
        content = re.sub(r'\s+', ' ', content).strip().casefold()
        return content.strip('。！？!?；;，,、. ')


# ---------------------------------------------------------------------------
# Plugin entry point
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Register the holographic memory provider with the plugin system."""
    config = _load_plugin_config()
    provider = HolographicMemoryProvider(config=config)
    ctx.register_memory_provider(provider)
