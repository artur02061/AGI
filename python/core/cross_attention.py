"""
Кристина 7.3 — Cross-Attention с памятью (Memory-Augmented Attention)

ЗАЧЕМ:
  Self-Attention (WISH-004) смотрит внутри фразы.
  Cross-Attention позволяет "смотреть" на ВНЕШНЮЮ память при генерации.

  Query = текущий вопрос пользователя
  Key/Value = релевантные записи из ChromaDB

  Это RAG, но ВНУТРИ модели, а не как пост-процессинг.

КАК РАБОТАЕТ:
  ┌────────────────────────────────────────────────┐
  │ Текущий контекст (вопрос пользователя)         │
  │ X = [token_emb_1, token_emb_2, ...]           │
  └──────────────────┬─────────────────────────────┘
                     │ Q = X @ Wq
                     ↓
  ┌────────────────────────────────────────────────┐
  │ Cross-Attention                                │
  │                                                │
  │   Q (из текущего контекста)                    │
  │   K = memory_vectors @ Wk (из ChromaDB)        │
  │   V = memory_vectors @ Wv (из ChromaDB)        │
  │                                                │
  │   Attn = softmax(Q @ K.T / √d) @ V            │
  │                                                │
  │   → Вектор, обогащённый памятью                │
  └──────────────────┬─────────────────────────────┘
                     ↓
  ┌────────────────────────────────────────────────┐
  │ Gate: α * self_attn + (1-α) * cross_attn       │
  │ α обучаемый — модель решает сколько "памяти"   │
  └────────────────────────────────────────────────┘

ИНТЕГРАЦИЯ:
  - Используется в orchestrator при Tier 2-3
  - Обогащает контекст ДО генерации ответа
  - Работает с vector_memory (ChromaDB + bge-m3)
"""

import math
import json
import time
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

from utils.logging import get_logger
import config

logger = get_logger("cross_attention")


# ═══════════════════════════════════════════════════════════════
#               ЛИНЕЙНАЯ АЛГЕБРА (минимум, чистый Python)
# ═══════════════════════════════════════════════════════════════

def _matmul_mv(M: List[List[float]], v: List[float]) -> List[float]:
    """Матрица × вектор: M[m×n] @ v[n] → r[m]"""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _softmax(values: List[float]) -> List[float]:
    if not values:
        return []
    max_val = max(values)
    exps = [math.exp(min(v - max_val, 80)) for v in values]
    total = sum(exps) + 1e-10
    return [e / total for e in exps]


def _vec_add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def _vec_scale(v: List[float], s: float) -> List[float]:
    return [x * s for x in v]


def _random_matrix(rows: int, cols: int) -> List[List[float]]:
    import random
    scale = math.sqrt(2.0 / (rows + cols))
    return [[random.gauss(0, scale) for _ in range(cols)] for _ in range(rows)]


def _layer_norm(x: List[float], eps: float = 1e-5) -> List[float]:
    n = len(x)
    mean = sum(x) / n
    var = sum((xi - mean) ** 2 for xi in x) / n
    inv_std = 1.0 / math.sqrt(var + eps)
    return [(xi - mean) * inv_std for xi in x]


# ═══════════════════════════════════════════════════════════════
#               CROSS-ATTENTION HEAD
# ═══════════════════════════════════════════════════════════════


class CrossAttentionHead:
    """
    Одна голова cross-attention:
      Q из текущего контекста, K/V из памяти.

    d_model: размерность входных векторов контекста
    d_memory: размерность векторов памяти (ChromaDB = 1024 для bge-m3)
    d_head: размерность проекции (d_model // n_heads)
    """

    def __init__(self, d_model: int, d_memory: int, d_head: int):
        self.d_model = d_model
        self.d_memory = d_memory
        self.d_head = d_head

        # Проекции
        self.Wq = _random_matrix(d_model, d_head)    # Query: из контекста
        self.Wk = _random_matrix(d_memory, d_head)    # Key: из памяти
        self.Wv = _random_matrix(d_memory, d_head)    # Value: из памяти

        self._scale = 1.0 / math.sqrt(d_head)

    def forward(
        self,
        query: List[float],
        memory_keys: List[List[float]],
        memory_values: List[List[float]],
    ) -> Tuple[List[float], List[float]]:
        """
        Cross-attention: один запрос к N записям памяти.

        Args:
            query: вектор запроса [d_model]
            memory_keys: матрица ключей [N × d_memory]
            memory_values: матрица значений [N × d_memory]

        Returns:
            (output [d_head], attention_weights [N])
        """
        # Q = query @ Wq → [d_head]
        q = _matmul_mv(list(zip(*self.Wq)), query) if self.Wq else query[:self.d_head]
        # Более правильно: q[j] = sum(query[i] * Wq[i][j])
        q = [sum(query[i] * self.Wq[i][j] for i in range(min(len(query), self.d_model)))
             for j in range(self.d_head)]

        # K = memory @ Wk → [N × d_head]
        # V = memory @ Wv → [N × d_head]
        n_mem = len(memory_keys)
        if n_mem == 0:
            return [0.0] * self.d_head, []

        keys = []
        vals = []
        for m_idx in range(n_mem):
            mk = memory_keys[m_idx]
            mv = memory_values[m_idx]
            # k[j] = sum(mk[i] * Wk[i][j])
            k = [sum(mk[i] * self.Wk[i][j]
                     for i in range(min(len(mk), self.d_memory)))
                 for j in range(self.d_head)]
            v = [sum(mv[i] * self.Wv[i][j]
                     for i in range(min(len(mv), self.d_memory)))
                 for j in range(self.d_head)]
            keys.append(k)
            vals.append(v)

        # Attention scores: Q · K^T / √d
        scores = [_dot(q, k) * self._scale for k in keys]
        weights = _softmax(scores)

        # Output: weighted sum of values
        output = [0.0] * self.d_head
        for m_idx in range(n_mem):
            w = weights[m_idx]
            for j in range(self.d_head):
                output[j] += w * vals[m_idx][j]

        return output, weights


# ═══════════════════════════════════════════════════════════════
#               MULTI-HEAD CROSS-ATTENTION
# ═══════════════════════════════════════════════════════════════


class MultiHeadCrossAttention:
    """
    Multi-Head Cross-Attention с гейтированием.

    Несколько голов "смотрят" на разные аспекты памяти,
    затем gate решает сколько памяти подмешать.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_memory: int = 1024,
        n_heads: int = 4,
    ):
        self.d_model = d_model
        self.d_memory = d_memory
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Головы cross-attention
        self.heads = [
            CrossAttentionHead(d_model, d_memory, self.d_head)
            for _ in range(n_heads)
        ]

        # Output projection: concat(heads) → d_model
        self.Wo = _random_matrix(d_model, d_model)

        # Gate: обучаемый скаляр α ∈ [0, 1]
        # α = sigmoid(gate_w · [context; memory_attn] + gate_b)
        self.gate_w = [0.0] * (d_model * 2)  # Инициализируем нулями → gate ≈ 0.5
        self.gate_b = 0.0

        # Статистика
        self._total_queries = 0
        self._avg_gate = 0.5

    def forward(
        self,
        context_vec: List[float],
        memory_vectors: List[List[float]],
    ) -> Tuple[List[float], Dict]:
        """
        Cross-attention: обогащает контекст памятью.

        Args:
            context_vec: вектор текущего контекста [d_model]
            memory_vectors: записи из ChromaDB [N × d_memory]

        Returns:
            (enriched_vec [d_model], info dict)
        """
        if not memory_vectors:
            return context_vec, {"gate": 0.0, "n_memories": 0, "weights": []}

        self._total_queries += 1

        # Multi-head attention
        head_outputs = []
        all_weights = []

        for head in self.heads:
            out, weights = head.forward(
                query=context_vec,
                memory_keys=memory_vectors,
                memory_values=memory_vectors,
            )
            head_outputs.append(out)
            all_weights.append(weights)

        # Concat heads → [d_model]
        concat = []
        for h_out in head_outputs:
            concat.extend(h_out)

        # Output projection
        attn_output = [
            sum(concat[i] * self.Wo[i][j]
                for i in range(min(len(concat), self.d_model)))
            for j in range(self.d_model)
        ]

        # Layer norm on attention output
        attn_output = _layer_norm(attn_output)

        # Gate: сколько памяти подмешать
        gate_input = context_vec[:self.d_model] + attn_output[:self.d_model]
        gate_logit = sum(
            self.gate_w[i] * gate_input[i]
            for i in range(min(len(self.gate_w), len(gate_input)))
        ) + self.gate_b
        gate = 1.0 / (1.0 + math.exp(-max(-10, min(10, gate_logit))))  # sigmoid

        # Update running average
        self._avg_gate = 0.95 * self._avg_gate + 0.05 * gate

        # Blend: enriched = (1-gate)*context + gate*memory_attn
        enriched = [
            (1.0 - gate) * context_vec[i] + gate * attn_output[i]
            for i in range(min(len(context_vec), len(attn_output)))
        ]

        # Усредняем веса по головам для интерпретации
        n_mem = len(memory_vectors)
        avg_weights = [0.0] * n_mem
        for head_w in all_weights:
            for i in range(min(len(head_w), n_mem)):
                avg_weights[i] += head_w[i] / self.n_heads

        info = {
            "gate": round(gate, 3),
            "n_memories": n_mem,
            "weights": [round(w, 3) for w in avg_weights[:10]],  # Top 10
            "avg_gate": round(self._avg_gate, 3),
        }

        return enriched, info


# ═══════════════════════════════════════════════════════════════
#               MEMORY-AUGMENTED CONTEXT
# ═══════════════════════════════════════════════════════════════


class MemoryAugmentedContext:
    """
    Высокоуровневый модуль: обогащает контекст запроса памятью.

    Использование:
        mac = MemoryAugmentedContext(vector_memory, sentence_embeddings)

        # При обработке запроса
        enriched = mac.enrich(
            user_input="Как создать CSV-парсер?",
            context_embedding=[...],  # от sentence_embeddings
        )

        if enriched:
            context_vec = enriched["context_vec"]  # Обогащённый вектор
            memories = enriched["memories"]         # Какие записи использованы
            gate = enriched["gate"]                 # Сколько памяти подмешано
    """

    def __init__(
        self,
        vector_memory=None,
        sentence_embeddings=None,
        d_model: int = 128,
        d_memory: int = 1024,
        n_heads: int = 4,
        max_memories: int = 5,
        db_path: Path = None,
    ):
        self._vector_memory = vector_memory
        self._sentence = sentence_embeddings
        self._max_memories = max_memories

        # Cross-Attention модуль
        self.cross_attention = MultiHeadCrossAttention(
            d_model=d_model,
            d_memory=d_memory,
            n_heads=n_heads,
        )

        # Persistence
        self._db_path = db_path or (config.config.data_dir / "cross_attention.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

        # Статистика
        self._total_enrichments = 0
        self._useful_enrichments = 0  # gate > 0.3
        self._load_stats()

        logger.info(
            f"🔗 CrossAttention: d_model={d_model}, d_memory={d_memory}, "
            f"heads={n_heads}, enrichments={self._total_enrichments}"
        )

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cross_attn_stats (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS cross_attn_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query TEXT NOT NULL,
                n_memories INTEGER,
                gate REAL,
                top_memory TEXT,
                created_at REAL NOT NULL
            )
        """)
        self._conn.commit()

    def _load_stats(self):
        row = self._conn.execute(
            "SELECT value FROM cross_attn_stats WHERE key = 'total_enrichments'"
        ).fetchone()
        if row:
            self._total_enrichments = int(row["value"])
        row = self._conn.execute(
            "SELECT value FROM cross_attn_stats WHERE key = 'useful_enrichments'"
        ).fetchone()
        if row:
            self._useful_enrichments = int(row["value"])

    def _save_stats(self):
        for key, val in [
            ("total_enrichments", str(self._total_enrichments)),
            ("useful_enrichments", str(self._useful_enrichments)),
        ]:
            self._conn.execute("""
                INSERT INTO cross_attn_stats (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?
            """, (key, val, val))
        self._conn.commit()

    def enrich(
        self,
        user_input: str,
        context_embedding: List[float] = None,
        n_results: int = None,
    ) -> Optional[Dict]:
        """
        Обогащает контекст запроса релевантной памятью.

        1. Ищет N ближайших записей в vector_memory (ChromaDB)
        2. Пропускает через cross-attention
        3. Возвращает обогащённый вектор

        Returns:
            Dict с полями:
            - context_vec: обогащённый вектор [d_model]
            - memories: список использованных записей
            - gate: доля памяти (0=нет, 1=только память)
            - weights: attention weights на каждую запись
        """
        if not self._vector_memory:
            return None

        n = n_results or self._max_memories

        # 1. Получаем context embedding
        if context_embedding is None and self._sentence:
            context_embedding = self._sentence.encode(user_input)

        if context_embedding is None:
            return None

        # Проецируем до d_model если нужно
        d_model = self.cross_attention.d_model
        if len(context_embedding) > d_model:
            # Простой downsample: берём первые d_model элементов
            # (в продакшене нужна обучаемая проекция)
            ctx_vec = context_embedding[:d_model]
        elif len(context_embedding) < d_model:
            ctx_vec = context_embedding + [0.0] * (d_model - len(context_embedding))
        else:
            ctx_vec = list(context_embedding)

        # 2. Ищем в ChromaDB
        try:
            search_results = self._vector_memory.search(
                query=user_input,
                n_results=n,
            )
        except Exception as e:
            logger.debug(f"CrossAttention: memory search failed: {e}")
            return None

        if not search_results or not search_results.get("documents"):
            return None

        # Извлекаем вектора записей
        memory_vectors = []
        memory_texts = []
        memory_metadatas = []

        documents = search_results.get("documents", [[]])[0]
        embeddings = search_results.get("embeddings", [[]])[0] if search_results.get("embeddings") else []
        metadatas = search_results.get("metadatas", [[]])[0]
        distances = search_results.get("distances", [[]])[0]

        for i, doc in enumerate(documents):
            if i < len(embeddings) and embeddings[i]:
                memory_vectors.append(embeddings[i])
            else:
                # Если нет эмбеддинга — кодируем через sentence_embeddings
                if self._sentence:
                    emb = self._sentence.encode(doc)
                    if emb:
                        memory_vectors.append(emb)
                        continue
                continue  # Пропускаем без вектора

            memory_texts.append(doc)
            if i < len(metadatas):
                memory_metadatas.append(metadatas[i])

        if not memory_vectors:
            return None

        # 3. Cross-Attention
        enriched_vec, info = self.cross_attention.forward(
            context_vec=ctx_vec,
            memory_vectors=memory_vectors,
        )

        # 4. Статистика
        self._total_enrichments += 1
        if info["gate"] > 0.3:
            self._useful_enrichments += 1

        # Логируем
        top_memory = memory_texts[0][:100] if memory_texts else ""
        self._conn.execute("""
            INSERT INTO cross_attn_log (query, n_memories, gate, top_memory, created_at)
            VALUES (?, ?, ?, ?, ?)
        """, (user_input[:200], len(memory_vectors), info["gate"], top_memory, time.time()))

        if self._total_enrichments % 20 == 0:
            self._save_stats()

        logger.debug(
            f"🔗 CrossAttn: gate={info['gate']:.2f}, "
            f"memories={info['n_memories']}, "
            f"top='{top_memory[:40]}...'"
        )

        return {
            "context_vec": enriched_vec,
            "memories": [
                {"text": memory_texts[i][:200] if i < len(memory_texts) else "",
                 "weight": info["weights"][i] if i < len(info["weights"]) else 0.0,
                 "distance": distances[i] if i < len(distances) else 1.0}
                for i in range(len(memory_vectors))
            ],
            "gate": info["gate"],
            "weights": info["weights"],
            "avg_gate": info["avg_gate"],
        }

    def get_stats(self) -> Dict:
        return {
            "total_enrichments": self._total_enrichments,
            "useful_enrichments": self._useful_enrichments,
            "useful_rate": round(
                self._useful_enrichments / max(self._total_enrichments, 1) * 100, 1
            ),
            "avg_gate": round(self.cross_attention._avg_gate, 3),
            "d_model": self.cross_attention.d_model,
            "d_memory": self.cross_attention.d_memory,
            "n_heads": self.cross_attention.n_heads,
        }

    def close(self):
        self._save_stats()
        self._conn.close()
