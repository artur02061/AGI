# Code Review: Kristina 6.0 — Bugs, Problems, Stubs, Workarounds

## 1. BUGS (real errors that will break execution)

### 1.1. `_save_to_vector_memory` uses cumulative stats instead of per-request
**File:** `core/agent.py:429-433`

```python
importance = min(3, 1 + self.stats["total_tool_calls"])
```

`self.stats["total_tool_calls"]` is a **global** session counter. After the second tool call ever, importance is always 3.

### 1.2. `had_errors` is also cumulative
**File:** `core/agent.py:433`

```python
'had_errors': self.stats["total_errors"] > 0,
```

After the first error in a session, **all** subsequent dialogues will be marked `had_errors=True`.

### 1.3. Episode rotation deletes by importance, ignoring recency
**File:** `core/memory.py:124-126`

```python
self.episodic.sort(key=lambda x: x['importance'])
self.episodic = self.episodic[100:]
```

Fresh but low-importance episodes get deleted before old high-importance ones. No timestamp consideration.

### 1.4. `asyncio.get_event_loop()` deprecated since Python 3.10
**Files:** `core/memory.py:110`, `bridge.py:129`

Raises `DeprecationWarning` in 3.10+ and `RuntimeError` in 3.12 without a running loop. `asyncio.run()` inside existing loop context also raises `RuntimeError`.

### 1.5. Wrong strptime format for ISO weeks
**File:** `modules/memory/summarizer.py:208, 346`

```python
dt = datetime.strptime(f"{year}-W{w:02d}-1", "%Y-W%W-%w")
```

`%W` is Monday-first week (0-53), not ISO week. Need `%G-W%V-%u` for ISO weeks. Produces wrong dates.

### 1.6. Executor capability check blocks dynamic tools
**File:** `core/agents/executor.py:72`

`self.capabilities` is a static list. Any tools added at runtime but not in capabilities will be rejected.

### 1.7. Executor calls tools with positional args
**File:** `core/agents/executor.py:80`

```python
result = await tool(*args)
```

But tools have **named** parameters. Works accidentally for single-arg tools, fragile for multi-arg.

### 1.8. Sync `requests` blocks event loop in async tools
**Files:** `tools/web_tools.py:181`, `tools/web_tools.py:249`

`requests.get()` blocks the async event loop. `httpx` is in requirements but not used here.

### 1.9. Sync `ollama.embeddings()` blocks event loop
**File:** `modules/rag/vector_store.py:285`

Synchronous embedding call blocks all async tasks during every `add_dialogue` and `search`.

### 1.10. Double embedding cache
**Files:** `bridge.py:85` + `modules/rag/vector_store.py:96`

Two independent embedding caches (`EmbeddingCacheAdapter` and `VectorMemory.embedding_cache`) store the same data. Double memory usage, no consistency.

---

## 2. ARCHITECTURE PROBLEMS

### 2.1. `MemorySystem` is dead code
`core/memory.py` has a full `MemorySystem` class. `bridge.py` has `MemoryAdapter`. Only `MemoryAdapter` is used. `MemorySystem` is never instantiated.

### 2.2. `identity.analyze_interaction()` is never called
Evolution logic exists but is never invoked in the processing pipeline.

### 2.3. MetaCognition strategies are disconnected
`select_strategy()`, `record_strategy_outcome()`, `estimate_confidence()`, `record_outcome()` — none are called from orchestrator or agent.

### 2.4. VAD emotional engine doesn't update Identity
`VADEmotionalEngine` updates in `process_input`, but `identity.update_mood()` is never called with VAD results. `identity.current_mood` stays "neutral" forever.

### 2.5. `SelfAwareness.get_narrative_summary()` is unused
Inner narrative is generated but never injected into prompts.

### 2.6. Orchestrator triple-calls director
analyze_request() + execute() (if primary=director) + synthesize_response() = 3 LLM calls per request. Synthesis over director's own output is redundant.

### 2.7. `_build_task` for executor always sets `tool=None`
Director's analysis result (which agent/what task) is lost when building executor tasks.

---

## 3. STUBS

- `MemoryAdapter.load()` — empty (`bridge.py:205`)
- `ThreadTrackerAdapter.save()` / `load()` — empty (`bridge.py:355-356`)
- `ContextCompressor` — created but never used (`main.py:97`)
- `EmotionAnalyzer` — result discarded (`main.py:269`)
- `response_cache` — naive, no invalidation, caches tool-call results
- `AnalystAgent._web_search_and_analyze()` — dead code, duplicated inline
- `_analyze_data` signature mismatch — expects Dict, gets str

---

## 4. WORKAROUNDS

- `config.py` backward compat via `__getattr__` with 60+ mappings
- `_COMPUTED_MAP` with hardcoded lambda values
- `ThreadTrackerAdapter._v4_current_thread` dual state
- `_execute_tool` kwargs→positional fallback
- `main.py` docstring says v5.0, code says v6.0
- DDGS dual-library fallback
- ChromaDB triple fallback
- Pickle→JSON one-time migration code
- Hardcoded `Path.home() / "Desktop"`

---

## 5. SECURITY CONCERNS

- `file_access_mode: "unrestricted"` by default
- No user confirmation before file create/write/delete via LLM
- `blocked_extensions` only covers Windows executables
- MD5 for cache keys (not critical but unnecessary)

---

## SUMMARY

The project is architecturally ambitious but has the classic "Frankenstein" problem: modules were designed independently but **not wired together**. MetaCognition, SelfAwareness, VAD, Identity, ContextCompressor, and EmotionAnalyzer all initialize but don't form a unified pipeline. Most critical bugs: blocking sync calls in async context and incorrect cumulative stats.
