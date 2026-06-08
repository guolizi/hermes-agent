# Butterfly Dream Memory Provider

🦋 Three-dimensional memory for Hermes Agent — scores facts across **Relevance**, **Recency**, and **Importance** for context-aware recall.

Built on top of the Holographic plugin with LLM extraction, entity tracking, media attachments, and CJK-aware FTS5 search.

## Requirements

None — fully local. Uses SQLite (always available). NumPy optional for HRR vector algebra.

## Setup

```bash
hermes memory setup    # select "butterfly_dream"
```

Or manually:

```bash
hermes config set memory.provider butterfly_dream
```

## Config

Configuration in `$HERMES_HOME/config.yaml` under `memory.providers.butterfly_dream`,
or in `~/.hermes/butterfly_config.yaml` (takes priority).

| Key | Default | Description |
|-----|---------|-------------|
| `db_path` | `$HERMES_HOME/memories/butterfly_memory.db` | SQLite database path |
| `llm_extract` | `false` | Enable LLM-based fact extraction at session end |
| `auto_extract` | `false` | Legacy regex-based extraction (Chinese + English patterns) |
| `prefetch_limit` | `10` | Number of facts to inject into the system prompt each turn |
| `default_trust` | `0.5` | Default trust score for new facts |
| `min_trust` | `0.3` | Minimum trust threshold for retrieval |
| `recency_half_life_days` | `30` | Days for recency score to decay by half |
| `hrr_dim` | `1024` | HRR vector dimensions |
| `extract_interval` | `20` | Extract facts every N turns (0 = disable periodic extraction) |
| `debug_logging` | `false` | Enable debug logs to `$HERMES_HOME/logs/butterfly.log` |

### Extraction Model

Configured under `memory.providers.butterfly_dream`:

| Key | Default | Description |
|-----|---------|-------------|
| `extraction_model` | — | Model name (e.g. `deepseek-v4-flash`) |
| `extraction_provider` | — | Provider name (e.g. `deepseek`) |
| `extraction_base_url` | — | Optional API base URL override |

### Media Compression

Set under `memory.providers.butterfly_dream.compression`:

| Key | Default | Description |
|-----|---------|-------------|
| `enabled` | `true` | Enable media compression on attachment |
| `image.quality` | `85` | JPEG quality (1–100) |
| `image.max_dim` | `1920` | Max image width/height (preserves aspect ratio) |
| `video.bitrate` | `2M` | Video bitrate (ffmpeg) |
| `video.max_fps` | `30` | Max video framerate |
| `audio.bitrate` | `128k` | Audio bitrate (ffmpeg MP3) |
| `audio.sample_rate` | `44100` | Audio sample rate |

## Tools

| Tool | Description |
|------|-------------|
| `fact_store` | 10 actions: `add`, `search`, `probe`, `related`, `reason`, `contradict`, `timeline`, `summarize`, `update`, `remove`, `list` |
| `fact_feedback` | Rate facts as `helpful` or `unhelpful` (trains trust + importance scores) |
| `media_attach` | Attach an image/audio/video file to a stored fact |
| `media_detach` | Remove a media attachment (leaves file on disk) |
| `media_orphans` | List media files on disk with no DB reference |
| `media_cleanup` | Remove orphaned media files (supports `dry_run`) |

## Retrieval Pipeline

1. **FTS5 search** — Full-text search with CJK-aware jieba segmentation, synonym expansion, and stop-word filtering
2. **Semantic category boost** — Query intent detection (`time`, `place`, `person`, `activity`, etc.) boosts matching fact categories
3. **Entity boost** — Facts linked to entities mentioned in the query get a relevance bump
4. **Entity mismatch penalty** — Facts about entities NOT mentioned in the query are penalized
5. **Three-dimensional scoring** — `Relevance × Recency × Importance × Trust`
6. **Category diversity re-ranking** — Prevents any single category from dominating results
7. **Entity diversity re-ranking** — Swaps in facts about minority entities when one entity dominates

## Data Flow

```
User message → prefetch(query) → ThreeDimRetriever.search()
                                        ↓
                              ┌─────────────────┐
                              │  FTS5 candidates │ ← facts_fts (populated by jieba-segmented content)
                              │  + semantic cats │ ← category-based fallback
                              └────────┬────────┘
                                       ↓
                              ┌─────────────────┐
                              │  3D scoring      │
                              │  relevance(fts)  │
                              │  recency(decay)  │
                              │  importance(raw) │
                              └────────┬────────┘
                                       ↓
                              ┌─────────────────┐
                              │ diversity re-    │
                              │ rank (category   │
                              │  + entity)       │
                              └────────┬────────┘
                                       ↓
                              injected into prompt
                              as `## 🦋 Butterfly Dream Memory`
```

## Extraction

Butterfly Dream can extract facts from conversations automatically:

- **LLM extraction** (recommended) — triggered at session end and pre-compress, extracts structured facts with category, tags, importance, and content-date
- **Regex extraction** (legacy) — pattern-based fallback for user preferences (`I like...`, `我喜欢...`) and project decisions (`We decided...`, `项目使用...`), with Chinese and English support

Extraction runs asynchronously in a background thread to avoid blocking the conversation.

## Architecture

```
butterfly_dream/
├── __init__.py       # MemoryProvider plugin — lifecycle, extraction, tool dispatch
├── store.py          # MemoryStore — SQLite schema, CRUD, dedup, merge, FTS5
├── retrieval.py      # ThreeDimRetriever — FTS5 + 3D scoring + diversity
├── synonyms.py       # Synonym expansion for FTS5 query enhancement
├── holographic.py    # HRR (Holographic Reduced Representations) vector encoding
├── media_compressor.py  # Image/audio/video compression via Pillow + ffmpeg
└── media_utils.py    # SHA-256 hashing, MIME detection helpers
```

## Debug Logging

Enable `debug_logging: true` in config to get detailed search/extraction logs at:

```
$HERMES_HOME/logs/butterfly.log
```

Includes per-query candidate counts, scoring breakdown, extraction events, and timing info.
