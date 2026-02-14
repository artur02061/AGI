"""
Кристина 7.2 — SentenceEmbeddings (от слов к фразам)

ЗАЧЕМ:
  Word2Vec понимает слова по отдельности:
    "не" = [0.1, -0.3, ...]
    "работает" = [0.4, 0.2, ...]

  Но "не работает" != "работает" + "не"!
  Нужно понимать ФРАЗЫ целиком.

АРХИТЕКТУРА (3 уровня, от простого к сложному):

  Level 1: Weighted Average
    sentence_vec = Σ(word_vec * idf_weight) / N
    Просто, но теряет порядок слов.

  Level 2: Positional Encoding
    sentence_vec = Σ(word_vec + pos_encoding(i)) / N
    Учитывает ПОЗИЦИЮ слова во фразе (как в трансформерах).

  Level 3: Learned Attention Pooling
    attention_weights = softmax(W @ word_vecs)
    sentence_vec = Σ(attention_weight_i * word_vec_i)
    Учится какие слова ВАЖНЕЕ в каждом контексте.

ОБУЧЕНИЕ:
  Level 1-2: Не требуют обучения (используют готовые Word2Vec).
  Level 3: Обучается на парах (вопрос, ответ) — похожие пары
           должны давать похожие sentence vectors.

ИНТЕГРАЦИЯ:
  - NeuralEngine.understand_sentence() → теперь возвращает sentence vector
  - VectorStore → можно искать по sentence embeddings вместо bge-m3
  - IntentRouter → семантический поиск паттернов
"""

import math
import json
import sqlite3
import time
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import Counter

from utils.logging import get_logger
import config

logger = get_logger("sentence_embeddings")

# ═══════════════════════════════════════════════════════════════
#               МАТЕМАТИКА
# ═══════════════════════════════════════════════════════════════


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    """Косинусное сходство двух векторов"""
    if len(v1) != len(v2) or not v1:
        return 0.0
    dot = sum(a * b for a, b in zip(v1, v2))
    norm1 = math.sqrt(sum(a * a for a in v1))
    norm2 = math.sqrt(sum(b * b for b in v2))
    if norm1 < 1e-10 or norm2 < 1e-10:
        return 0.0
    return dot / (norm1 * norm2)


def _vec_add(v1: List[float], v2: List[float]) -> List[float]:
    return [a + b for a, b in zip(v1, v2)]


def _vec_scale(v: List[float], s: float) -> List[float]:
    return [a * s for a in v]


def _vec_normalize(v: List[float]) -> List[float]:
    norm = math.sqrt(sum(a * a for a in v))
    if norm < 1e-10:
        return v
    return [a / norm for a in v]


def _softmax(values: List[float]) -> List[float]:
    """Стабильная softmax"""
    if not values:
        return []
    max_val = max(values)
    exps = [math.exp(v - max_val) for v in values]
    total = sum(exps)
    if total < 1e-10:
        return [1.0 / len(values)] * len(values)
    return [e / total for e in exps]


class SentenceEmbeddings:
    """
    Понимание фраз на основе Word2Vec эмбеддингов.

    Три уровня агрегации:
    1. Weighted Average (IDF-взвешенное среднее)
    2. Positional Encoding (с учётом позиции)
    3. Attention Pooling (обучаемое внимание)

    Использование:
        se = SentenceEmbeddings(neural_engine)
        vec = se.encode("Привет, как дела?")          # Level 1
        vec = se.encode("Привет, как дела?", level=2)  # Level 2
        vec = se.encode("Привет, как дела?", level=3)  # Level 3

        sim = se.similarity("Привет!", "Здравствуй!")   # 0.87
    """

    def __init__(self, neural_engine, db_path: Path = None):
        """
        Args:
            neural_engine: NeuralEngine instance (для Word2Vec эмбеддингов)
        """
        self._engine = neural_engine
        self._db_path = db_path or (config.config.data_dir / "sentence_embeddings.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")

        self._create_tables()

        # IDF кеш (Inverse Document Frequency)
        self._idf_cache: Dict[str, float] = {}
        self._doc_count = 0

        # Attention weights (Level 3) — обучаемые
        self._embedding_dim = 128  # Совпадает с NeuralEngine EMBEDDING_DIM (v7.3: 64→128)
        self._attention_w: Optional[List[float]] = None  # Вектор внимания

        self._load_state()

        logger.info(
            f"📐 SentenceEmbeddings: dim={self._embedding_dim}, "
            f"idf_words={len(self._idf_cache)}, "
            f"attention={'trained' if self._attention_w else 'untrained'}"
        )

    def _create_tables(self):
        cur = self._conn.cursor()

        # IDF статистика (в скольких документах встречается слово)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS word_doc_freq (
                word TEXT PRIMARY KEY,
                doc_count INTEGER DEFAULT 1,
                updated_at REAL NOT NULL
            )
        """)

        # Общее количество документов
        cur.execute("""
            CREATE TABLE IF NOT EXISTS meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Attention weights (Level 3)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS attention_weights (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                weights TEXT NOT NULL,
                training_steps INTEGER DEFAULT 0,
                updated_at REAL NOT NULL
            )
        """)

        # Кеш sentence embeddings (для часто используемых фраз)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS embedding_cache (
                text_hash TEXT PRIMARY KEY,
                text TEXT NOT NULL,
                embedding TEXT NOT NULL,
                level INTEGER NOT NULL,
                created_at REAL NOT NULL
            )
        """)

        self._conn.commit()

    def _load_state(self):
        """Загружает IDF и attention weights"""
        # IDF
        rows = self._conn.execute(
            "SELECT word, doc_count FROM word_doc_freq"
        ).fetchall()
        for row in rows:
            self._idf_cache[row["word"]] = row["doc_count"]

        # Doc count
        meta = self._conn.execute(
            "SELECT value FROM meta WHERE key = 'doc_count'"
        ).fetchone()
        self._doc_count = int(meta["value"]) if meta else 0

        # Attention weights
        att = self._conn.execute(
            "SELECT weights FROM attention_weights WHERE id = 1"
        ).fetchone()
        if att:
            try:
                self._attention_w = json.loads(att["weights"])
            except (json.JSONDecodeError, TypeError):
                self._attention_w = None

    # ═══════════════════════════════════════════════════════════════
    #               КОДИРОВАНИЕ ФРАЗ
    # ═══════════════════════════════════════════════════════════════

    def encode(
        self,
        text: str,
        level: int = 2,
    ) -> Optional[List[float]]:
        """
        Кодирует текст в вектор фиксированной размерности.

        Args:
            text: текст для кодирования
            level: уровень агрегации (1=avg, 2=positional, 3=attention)

        Returns:
            Вектор размерности EMBEDDING_DIM или None
        """
        tokens = self._engine.tokenize(text)
        words = [t for t in tokens if t.isalpha() or (len(t) > 1 and t.isalnum())]

        if not words:
            return None

        # Получаем Word2Vec эмбеддинги
        word_vecs = []
        valid_words = []
        for word in words:
            emb = self._engine._embeddings_cache.get(word.lower())
            if emb:
                word_vecs.append(emb)
                valid_words.append(word.lower())

        if not word_vecs:
            return None

        if level == 1:
            return self._encode_weighted_avg(valid_words, word_vecs)
        elif level == 2:
            return self._encode_positional(valid_words, word_vecs)
        elif level == 3:
            return self._encode_attention(valid_words, word_vecs)
        else:
            return self._encode_weighted_avg(valid_words, word_vecs)

    def _encode_weighted_avg(
        self,
        words: List[str],
        word_vecs: List[List[float]],
    ) -> List[float]:
        """
        Level 1: IDF-взвешенное среднее.

        Редкие слова важнее (IDF = log(N / doc_freq)).
        "Создай файл Python" → "Python" весит больше чем "файл".
        """
        dim = len(word_vecs[0])
        result = [0.0] * dim

        total_weight = 0.0
        for word, vec in zip(words, word_vecs):
            weight = self._get_idf(word)
            result = _vec_add(result, _vec_scale(vec, weight))
            total_weight += weight

        if total_weight > 0:
            result = _vec_scale(result, 1.0 / total_weight)

        return _vec_normalize(result)

    def _encode_positional(
        self,
        words: List[str],
        word_vecs: List[List[float]],
    ) -> List[float]:
        """
        Level 2: С позиционным кодированием (как в трансформерах).

        Позиция слова влияет на вектор:
        "не работает" ≠ "работает не" (хотя слова одинаковые)
        """
        dim = len(word_vecs[0])
        n = len(word_vecs)
        result = [0.0] * dim

        total_weight = 0.0
        for i, (word, vec) in enumerate(zip(words, word_vecs)):
            # IDF weight
            idf_weight = self._get_idf(word)

            # Positional encoding (синусоидальное, как в Vaswani et al.)
            pos_enc = self._positional_encoding(i, dim)

            # word_vec + pos_encoding
            enriched = _vec_add(vec, _vec_scale(pos_enc, 0.1))  # Небольшой вес позиции

            result = _vec_add(result, _vec_scale(enriched, idf_weight))
            total_weight += idf_weight

        if total_weight > 0:
            result = _vec_scale(result, 1.0 / total_weight)

        return _vec_normalize(result)

    def _encode_attention(
        self,
        words: List[str],
        word_vecs: List[List[float]],
    ) -> List[float]:
        """
        Level 3: Обучаемый attention pooling.

        attention_score(word) = W · word_vec
        weights = softmax(attention_scores)
        sentence_vec = Σ(weight_i * word_vec_i)

        W обучается на парах (вопрос, ответ).
        """
        dim = len(word_vecs[0])

        # Если attention weights ещё не обучены — fallback на Level 2
        if self._attention_w is None or len(self._attention_w) != dim:
            return self._encode_positional(words, word_vecs)

        # Вычисляем attention scores
        scores = []
        for vec in word_vecs:
            score = sum(a * b for a, b in zip(self._attention_w, vec))
            scores.append(score)

        # Softmax
        weights = _softmax(scores)

        # Взвешенная сумма
        result = [0.0] * dim
        for weight, vec in zip(weights, word_vecs):
            result = _vec_add(result, _vec_scale(vec, weight))

        return _vec_normalize(result)

    def _positional_encoding(self, pos: int, dim: int) -> List[float]:
        """
        Синусоидальное позиционное кодирование (Vaswani et al., 2017).

        PE(pos, 2i)   = sin(pos / 10000^(2i/dim))
        PE(pos, 2i+1) = cos(pos / 10000^(2i/dim))
        """
        pe = [0.0] * dim
        for i in range(0, dim, 2):
            div = math.pow(10000.0, (2 * i) / dim)
            pe[i] = math.sin(pos / div)
            if i + 1 < dim:
                pe[i + 1] = math.cos(pos / div)
        return pe

    def _get_idf(self, word: str) -> float:
        """IDF weight: log(total_docs / word_doc_count + 1)"""
        doc_freq = self._idf_cache.get(word, 0)
        if doc_freq == 0 or self._doc_count == 0:
            return 1.0
        return math.log(self._doc_count / (doc_freq + 1)) + 1.0

    # ═══════════════════════════════════════════════════════════════
    #               СРАВНЕНИЕ ФРАЗ
    # ═══════════════════════════════════════════════════════════════

    def similarity(self, text1: str, text2: str, level: int = 2) -> float:
        """
        Семантическое сходство двух фраз.

        similarity("Привет!", "Здравствуй!") → 0.87
        similarity("Создай файл", "Удали файл") → 0.45
        """
        vec1 = self.encode(text1, level=level)
        vec2 = self.encode(text2, level=level)
        if vec1 is None or vec2 is None:
            return 0.0
        return _cosine_similarity(vec1, vec2)

    def find_most_similar(
        self,
        query: str,
        candidates: List[str],
        level: int = 2,
        top_n: int = 5,
    ) -> List[Tuple[str, float]]:
        """
        Находит наиболее похожие фразы из списка кандидатов.

        Используется в IntentRouter для семантического поиска паттернов.
        """
        query_vec = self.encode(query, level=level)
        if query_vec is None:
            return []

        results = []
        for candidate in candidates:
            cand_vec = self.encode(candidate, level=level)
            if cand_vec is not None:
                sim = _cosine_similarity(query_vec, cand_vec)
                results.append((candidate, sim))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    # ═══════════════════════════════════════════════════════════════
    #               ОБУЧЕНИЕ
    # ═══════════════════════════════════════════════════════════════

    def learn_from_text(self, text: str):
        """
        Обновляет IDF статистику на новом тексте.
        Вызывается для каждого диалога.
        """
        tokens = self._engine.tokenize(text)
        words = set(t.lower() for t in tokens if t.isalpha() and len(t) > 2)

        if not words:
            return

        now = time.time()
        self._doc_count += 1

        for word in words:
            self._idf_cache[word] = self._idf_cache.get(word, 0) + 1
            self._conn.execute("""
                INSERT INTO word_doc_freq (word, doc_count, updated_at)
                VALUES (?, 1, ?)
                ON CONFLICT(word)
                DO UPDATE SET doc_count = doc_count + 1, updated_at = ?
            """, (word, now, now))

        self._conn.execute("""
            INSERT INTO meta (key, value) VALUES ('doc_count', ?)
            ON CONFLICT(key) DO UPDATE SET value = ?
        """, (str(self._doc_count), str(self._doc_count)))

        self._conn.commit()

    def train_attention(
        self,
        positive_pairs: List[Tuple[str, str]],
        negative_pairs: List[Tuple[str, str]] = None,
        learning_rate: float = 0.01,
        epochs: int = 10,
    ):
        """
        Обучает attention weights (Level 3) на парах фраз.

        positive_pairs: похожие фразы (вопрос-ответ, синонимы)
        negative_pairs: непохожие фразы (случайные пары)

        Цель: attention weights должны давать высокое сходство
        для positive_pairs и низкое для negative_pairs.
        """
        dim = self._embedding_dim

        # Инициализируем weights если нет
        if self._attention_w is None or len(self._attention_w) != dim:
            self._attention_w = [(random.random() - 0.5) * 0.1 for _ in range(dim)]

        # Генерируем negative pairs если не заданы
        if negative_pairs is None and len(positive_pairs) >= 2:
            negative_pairs = []
            texts = [t for pair in positive_pairs for t in pair]
            for _ in range(len(positive_pairs)):
                t1 = random.choice(texts)
                t2 = random.choice(texts)
                if t1 != t2:
                    negative_pairs.append((t1, t2))

        for epoch in range(epochs):
            total_loss = 0.0

            # Positive: увеличиваем сходство
            for text1, text2 in positive_pairs:
                vec1 = self._encode_attention_with_grad(text1)
                vec2 = self._encode_attention_with_grad(text2)
                if vec1 is None or vec2 is None:
                    continue

                sim = _cosine_similarity(vec1["vector"], vec2["vector"])
                loss = max(0, 1.0 - sim)  # Hinge loss: хотим sim → 1.0
                total_loss += loss

                if loss > 0:
                    self._update_attention_weights(
                        vec1, vec2, learning_rate, positive=True
                    )

            # Negative: уменьшаем сходство
            if negative_pairs:
                for text1, text2 in negative_pairs:
                    vec1 = self._encode_attention_with_grad(text1)
                    vec2 = self._encode_attention_with_grad(text2)
                    if vec1 is None or vec2 is None:
                        continue

                    sim = _cosine_similarity(vec1["vector"], vec2["vector"])
                    margin = 0.5
                    loss = max(0, sim - margin)  # Хотим sim < margin
                    total_loss += loss

                    if loss > 0:
                        self._update_attention_weights(
                            vec1, vec2, learning_rate, positive=False
                        )

        # Сохраняем weights
        self._save_attention_weights()
        logger.debug(f"📐 Attention trained: {epochs} epochs, final loss={total_loss:.4f}")

    def _encode_attention_with_grad(self, text: str) -> Optional[Dict]:
        """Кодирует с сохранением промежуточных данных для градиента"""
        tokens = self._engine.tokenize(text)
        words = [t.lower() for t in tokens if t.isalpha() and len(t) > 1]

        word_vecs = []
        valid_words = []
        for word in words:
            emb = self._engine._embeddings_cache.get(word)
            if emb:
                word_vecs.append(emb)
                valid_words.append(word)

        if not word_vecs:
            return None

        dim = len(word_vecs[0])
        if len(self._attention_w) != dim:
            return None

        scores = [sum(a * b for a, b in zip(self._attention_w, vec)) for vec in word_vecs]
        weights = _softmax(scores)

        result = [0.0] * dim
        for w, vec in zip(weights, word_vecs):
            result = _vec_add(result, _vec_scale(vec, w))

        return {
            "vector": _vec_normalize(result),
            "word_vecs": word_vecs,
            "weights": weights,
            "scores": scores,
        }

    def _update_attention_weights(
        self,
        data1: Dict,
        data2: Dict,
        lr: float,
        positive: bool,
    ):
        """Обновляет attention weights на одной паре"""
        dim = len(self._attention_w)
        direction = 1.0 if positive else -1.0

        # Простой gradient: двигаем W чтобы увеличить/уменьшить dot product
        # между encodings двух текстов
        for k in range(dim):
            grad = 0.0
            for vec, weight in zip(data1["word_vecs"], data1["weights"]):
                grad += vec[k] * data2["vector"][k]
            for vec, weight in zip(data2["word_vecs"], data2["weights"]):
                grad += vec[k] * data1["vector"][k]

            self._attention_w[k] += direction * lr * grad

    def _save_attention_weights(self):
        """Сохраняет attention weights в SQLite"""
        now = time.time()
        self._conn.execute("""
            INSERT INTO attention_weights (id, weights, training_steps, updated_at)
            VALUES (1, ?, 1, ?)
            ON CONFLICT(id)
            DO UPDATE SET weights = ?, training_steps = training_steps + 1, updated_at = ?
        """, (json.dumps(self._attention_w), now, json.dumps(self._attention_w), now))
        self._conn.commit()

    # ═══════════════════════════════════════════════════════════════
    #               УТИЛИТЫ
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        return {
            "embedding_dim": self._embedding_dim,
            "idf_words": len(self._idf_cache),
            "doc_count": self._doc_count,
            "attention_trained": self._attention_w is not None,
        }

    def close(self):
        self._conn.commit()
        self._conn.close()
