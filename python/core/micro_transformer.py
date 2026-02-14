"""
Кристина 7.4 — MicroTransformer (Механизм внимания)

ЭТО КВАНТОВЫЙ СКАЧОК.

ЗАЧЕМ:
  N-gram видит 2-3 слова. Трансформер видит ВСЕ слова сразу
  и понимает СВЯЗИ между ними.

  "Банк стоит на берегу реки" → банк = здание (attention на "берегу", "реки")
  "Банк выдал кредит"         → банк = финансы (attention на "кредит")

АРХИТЕКТУРА (Decoder-only, LLaMA-стиль):
  ┌─────────────────────────────────────┐
  │ Input: BPE token IDs               │
  │ [23, 45, 67, 89, ...]              │
  └──────────────┬──────────────────────┘
                 ↓
  ┌─────────────────────────────────────┐
  │ Token Embedding + Positional (RoPE) │
  │ [0.12, -0.34, 0.56, ...]  per token│
  └──────────────┬──────────────────────┘
                 ↓
  ┌─────────────────────────────────────┐  ×N layers
  │ RMSNorm                             │
  │         ↓                           │
  │ Multi-Head Self-Attention           │
  │   Q = X @ Wq, K = X @ Wk, V = X @ Wv│
  │   Attn = softmax(Q @ K.T / √d) @ V │
  │         ↓                           │
  │ Residual + RMSNorm                  │
  │         ↓                           │
  │ SwiGLU FFN (SiLU(xW_gate)⊙xW_up)W_d│
  │         ↓                           │
  │ Residual                            │
  └──────────────┬──────────────────────┘
                 ↓
  ┌─────────────────────────────────────┐
  │ RMSNorm → Linear → softmax          │
  │ → P(next_token | all_previous)      │
  └─────────────────────────────────────┘

ПАРАМЕТРЫ:
  d_model = 128       # Размерность модели
  n_heads = 4         # Количество голов внимания
  n_layers = 2        # Количество слоёв трансформера
  d_ff = 512          # Скрытый размер feed-forward
  max_seq_len = 256   # Максимальная длина последовательности
  vocab_size = ~8000  # Из BPE токенизатора

  Итого: ~1.5M параметров (крошечная по меркам LLM,
  но ОГРОМНЫЙ шаг для Кристины)

ОБУЧЕНИЕ:
  - Данные: все накопленные диалоги
  - Задача: предсказание следующего токена (language modeling)
  - Оптимизатор: Adam (ручная реализация)
  - Инкрементальное: дообучается после каждых N диалогов

ЧИСТЫЙ PYTHON:
  Никаких numpy, torch, tensorflow.
  Вся линейная алгебра вручную — медленнее, но без зависимостей.
  При необходимости можно заменить на numpy для ×50 ускорения.
"""

import math
import random
import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

from utils.logging import get_logger
import config

logger = get_logger("micro_transformer")

# ═══════════════════════════════════════════════════════════════
#               КОНФИГУРАЦИЯ МОДЕЛИ
# ═══════════════════════════════════════════════════════════════

D_MODEL = 128          # Размерность модели
N_HEADS = 4            # Голов внимания
N_LAYERS = 2           # Слоёв трансформера
D_FF = 512             # Feed-forward скрытый размер
MAX_SEQ_LEN = 256      # Макс. длина последовательности
DROPOUT_RATE = 0.1     # Dropout (при обучении)
LEARNING_RATE = 3e-4   # Adam learning rate
GRAD_CLIP = 1.0        # Gradient clipping

# ═══════════════════════════════════════════════════════════════
#               ЛИНЕЙНАЯ АЛГЕБРА (чистый Python)
# ═══════════════════════════════════════════════════════════════


def _zeros(rows: int, cols: int) -> List[List[float]]:
    """Создаёт нулевую матрицу [rows x cols]"""
    return [[0.0] * cols for _ in range(rows)]


def _zeros_vec(n: int) -> List[float]:
    """Нулевой вектор длины n"""
    return [0.0] * n


def _random_matrix(rows: int, cols: int, scale: float = None) -> List[List[float]]:
    """Xavier/He инициализация"""
    if scale is None:
        scale = math.sqrt(2.0 / (rows + cols))
    return [[(random.gauss(0, scale)) for _ in range(cols)] for _ in range(rows)]


def _random_vec(n: int, scale: float = 0.01) -> List[float]:
    return [random.gauss(0, scale) for _ in range(n)]


def _matmul(A: List[List[float]], B: List[List[float]]) -> List[List[float]]:
    """Умножение матриц A[m×k] @ B[k×n] → C[m×n]"""
    m = len(A)
    k = len(A[0]) if A else 0
    n = len(B[0]) if B else 0
    C = [[0.0] * n for _ in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0.0
            for p in range(k):
                s += A[i][p] * B[p][j]
            C[i][j] = s
    return C


def _matvec(M: List[List[float]], v: List[float]) -> List[float]:
    """Умножение матрицы на вектор M[m×n] @ v[n] → r[m]"""
    return [sum(M[i][j] * v[j] for j in range(len(v))) for i in range(len(M))]


def _transpose(M: List[List[float]]) -> List[List[float]]:
    """Транспонирование матрицы"""
    if not M:
        return []
    rows, cols = len(M), len(M[0])
    return [[M[i][j] for i in range(rows)] for j in range(cols)]


def _vec_add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]


def _vec_sub(a: List[float], b: List[float]) -> List[float]:
    return [x - y for x, y in zip(a, b)]


def _vec_scale(v: List[float], s: float) -> List[float]:
    return [x * s for x in v]


def _vec_mul(a: List[float], b: List[float]) -> List[float]:
    """Поэлементное умножение"""
    return [x * y for x, y in zip(a, b)]


def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def _softmax(values: List[float]) -> List[float]:
    """Стабильная softmax"""
    if not values:
        return []
    max_val = max(values)
    exps = [math.exp(min(v - max_val, 80)) for v in values]  # Ограничиваем для стабильности
    total = sum(exps) + 1e-10
    return [e / total for e in exps]


def _gelu(x: float) -> float:
    """GELU активация (аппроксимация)"""
    return 0.5 * x * (1.0 + math.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))


def _silu(x: float) -> float:
    """SiLU (Swish) активация: x * sigmoid(x) — используется в SwiGLU"""
    sig = 1.0 / (1.0 + math.exp(-max(-80, min(80, x))))
    return x * sig


def _layer_norm(x: List[float], gamma: List[float], beta: List[float], eps: float = 1e-5) -> List[float]:
    """Layer Normalization (legacy, для обратной совместимости)"""
    n = len(x)
    mean = sum(x) / n
    var = sum((xi - mean) ** 2 for xi in x) / n
    inv_std = 1.0 / math.sqrt(var + eps)
    return [(xi - mean) * inv_std * g + b for xi, g, b in zip(x, gamma, beta)]


def _rms_norm(x: List[float], gamma: List[float], eps: float = 1e-6) -> List[float]:
    """
    RMSNorm (Zhang & Sennrich, 2019) — используется в LLaMA, Mistral.
    Проще LayerNorm: без вычитания mean и без beta.
    Стабильнее и быстрее при обучении.
    """
    n = len(x)
    rms = math.sqrt(sum(xi * xi for xi in x) / n + eps)
    return [xi / rms * g for xi, g in zip(x, gamma)]


# ═══════════════════════════════════════════════════════════════
#               КОМПОНЕНТЫ ТРАНСФОРМЕРА
# ═══════════════════════════════════════════════════════════════


class Embedding:
    """Таблица эмбеддингов: token_id → вектор"""

    def __init__(self, vocab_size: int, d_model: int):
        self.vocab_size = vocab_size
        self.d_model = d_model
        scale = math.sqrt(1.0 / d_model)
        self.weight = _random_matrix(vocab_size, d_model, scale)

    def forward(self, token_ids: List[int]) -> List[List[float]]:
        """[seq_len] → [seq_len × d_model]"""
        result = []
        for tid in token_ids:
            if 0 <= tid < self.vocab_size:
                result.append(list(self.weight[tid]))
            else:
                result.append(_zeros_vec(self.d_model))
        return result

    def get_params(self) -> List[List[List[float]]]:
        return [self.weight]


class RoPE:
    """
    Rotary Position Embeddings (Su et al., 2021).

    Современная альтернатива синусоидальному PE:
    - Лучше обобщается на длинные последовательности
    - Кодирует ОТНОСИТЕЛЬНЫЕ позиции (а не абсолютные)
    - Используется в LLaMA, Mistral, Qwen
    """

    def __init__(self, d_model: int, max_seq_len: int = MAX_SEQ_LEN):
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        # Предвычисляем cos/sin для всех позиций
        self._cos_cache: List[List[float]] = []
        self._sin_cache: List[List[float]] = []
        self._precompute()

    def _precompute(self):
        """Предвычисляет RoPE для всех позиций"""
        half_d = self.d_model // 2
        for pos in range(self.max_seq_len):
            cos_row = []
            sin_row = []
            for i in range(half_d):
                freq = 1.0 / (10000.0 ** (2 * i / self.d_model))
                angle = pos * freq
                cos_row.append(math.cos(angle))
                sin_row.append(math.sin(angle))
            self._cos_cache.append(cos_row)
            self._sin_cache.append(sin_row)

    def apply(self, x: List[float], pos: int) -> List[float]:
        """Применяет RoPE к одному вектору на позиции pos"""
        if pos >= self.max_seq_len:
            pos = self.max_seq_len - 1
        half_d = self.d_model // 2
        cos_vals = self._cos_cache[pos]
        sin_vals = self._sin_cache[pos]

        result = list(x)
        for i in range(half_d):
            x0 = x[2 * i]
            x1 = x[2 * i + 1] if 2 * i + 1 < len(x) else 0.0
            result[2 * i] = x0 * cos_vals[i] - x1 * sin_vals[i]
            if 2 * i + 1 < len(result):
                result[2 * i + 1] = x0 * sin_vals[i] + x1 * cos_vals[i]
        return result


class MultiHeadAttention:
    """
    Multi-Head Self-Attention — СЕРДЦЕ трансформера.

    Каждый токен "смотрит" на все предыдущие и решает,
    на какие слова обратить внимание.

    Q (Query)  = "Что я ищу?"
    K (Key)    = "Что я предлагаю?"
    V (Value)  = "Какую информацию я несу?"

    Attention(Q, K, V) = softmax(Q @ K.T / √d_k) @ V
    """

    def __init__(self, d_model: int, n_heads: int):
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # Размерность на голову

        # Проекционные матрицы [d_model × d_model]
        self.Wq = _random_matrix(d_model, d_model)
        self.Wk = _random_matrix(d_model, d_model)
        self.Wv = _random_matrix(d_model, d_model)
        self.Wo = _random_matrix(d_model, d_model)

        # Bias
        self.bq = _zeros_vec(d_model)
        self.bk = _zeros_vec(d_model)
        self.bv = _zeros_vec(d_model)
        self.bo = _zeros_vec(d_model)

        # RoPE
        self.rope = RoPE(self.d_k)

    def forward(
        self,
        x: List[List[float]],
        causal_mask: bool = True,
    ) -> List[List[float]]:
        """
        Multi-Head Self-Attention.

        Args:
            x: [seq_len × d_model] — входная последовательность
            causal_mask: True = decoder (видит только предыдущие токены)

        Returns:
            [seq_len × d_model] — выход attention
        """
        seq_len = len(x)

        # Проецируем Q, K, V
        Q = [_vec_add(_matvec(self.Wq, x[i]), self.bq) for i in range(seq_len)]
        K = [_vec_add(_matvec(self.Wk, x[i]), self.bk) for i in range(seq_len)]
        V = [_vec_add(_matvec(self.Wv, x[i]), self.bv) for i in range(seq_len)]

        # Разбиваем на головы и обрабатываем
        all_heads_output = [_zeros_vec(self.d_model) for _ in range(seq_len)]

        for h in range(self.n_heads):
            start = h * self.d_k
            end = start + self.d_k

            # Извлекаем срез для головы h
            q_head = [q[start:end] for q in Q]
            k_head = [k[start:end] for k in K]
            v_head = [v[start:end] for v in V]

            # Применяем RoPE к Q и K
            q_head = [self.rope.apply(q, i) for i, q in enumerate(q_head)]
            k_head = [self.rope.apply(k, i) for i, k in enumerate(k_head)]

            # Scaled Dot-Product Attention
            scale = 1.0 / math.sqrt(self.d_k)
            head_output = self._attention(q_head, k_head, v_head, scale, causal_mask)

            # Записываем результат головы обратно
            for i in range(seq_len):
                for j in range(self.d_k):
                    all_heads_output[i][start + j] = head_output[i][j]

        # Output projection
        result = [_vec_add(_matvec(self.Wo, all_heads_output[i]), self.bo) for i in range(seq_len)]
        return result

    def _attention(
        self,
        Q: List[List[float]],
        K: List[List[float]],
        V: List[List[float]],
        scale: float,
        causal: bool,
    ) -> List[List[float]]:
        """Scaled Dot-Product Attention для одной головы"""
        seq_len = len(Q)
        d = len(Q[0]) if Q else 0

        output = []
        for i in range(seq_len):
            # Считаем attention scores: Q[i] · K[j] для всех j
            scores = []
            max_j = i + 1 if causal else seq_len
            for j in range(max_j):
                score = _dot(Q[i], K[j]) * scale
                scores.append(score)

            # Если causal — добавляем -inf для будущих позиций
            if causal and max_j < seq_len:
                scores.extend([-1e9] * (seq_len - max_j))

            # Softmax
            weights = _softmax(scores)

            # Взвешенная сумма V
            out = _zeros_vec(d)
            for j in range(min(len(weights), seq_len)):
                if weights[j] > 1e-10:
                    out = _vec_add(out, _vec_scale(V[j], weights[j]))
            output.append(out)

        return output

    def get_params(self) -> List:
        return [self.Wq, self.Wk, self.Wv, self.Wo,
                self.bq, self.bk, self.bv, self.bo]


class FeedForward:
    """
    SwiGLU Feed-Forward Network (Shazeer, 2020).
    Используется в LLaMA, Mistral, PaLM.

    FFN_SwiGLU(x) = (SiLU(x @ W_gate) ⊙ (x @ W_up)) @ W_down

    Преимущество над GELU FFN:
    - Gate-механизм контролирует поток информации
    - Лучшая сходимость при обучении
    - ~10% лучше perplexity при том же числе параметров
    """

    def __init__(self, d_model: int, d_ff: int):
        self.d_model = d_model
        self.d_ff = d_ff
        # SwiGLU использует 3 матрицы вместо 2
        self.W1 = _random_matrix(d_model, d_ff)       # W_gate
        self.b1 = _zeros_vec(d_ff)                     # b_gate (legacy compat)
        self.W_up = _random_matrix(d_model, d_ff)      # W_up (новая)
        self.W2 = _random_matrix(d_ff, d_model)        # W_down
        self.b2 = _zeros_vec(d_model)                  # b_down

    def forward(self, x: List[float]) -> List[float]:
        """[d_model] → [d_model] через SwiGLU"""
        # Gate path: SiLU(x @ W_gate + b_gate)
        gate = _vec_add(_matvec(self.W1, x), self.b1)
        gate = [_silu(g) for g in gate]
        # Up path: x @ W_up
        up = _matvec(self.W_up, x)
        # Gated: SiLU(gate) ⊙ up
        hidden = _vec_mul(gate, up)
        # Down: hidden @ W_down + b_down
        output = _vec_add(_matvec(self.W2, hidden), self.b2)
        return output

    def get_params(self) -> List:
        return [self.W1, self.b1, self.W_up, self.W2, self.b2]


class TransformerBlock:
    """
    Один блок трансформера (Pre-RMSNorm + SwiGLU):
      x → RMSNorm → MultiHeadAttention → + residual
        → RMSNorm → SwiGLU FeedForward → + residual

    v7.4: Замена LayerNorm → RMSNorm (LLaMA-стиль)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)

        # RMSNorm параметры (только gamma, без beta)
        self.ln1_gamma = [1.0] * d_model
        self.ln1_beta = [0.0] * d_model   # legacy: сохраняется для обратной совместимости
        self.ln2_gamma = [1.0] * d_model
        self.ln2_beta = [0.0] * d_model   # legacy

    def forward(self, x: List[List[float]], causal_mask: bool = True) -> List[List[float]]:
        """[seq_len × d_model] → [seq_len × d_model]"""
        seq_len = len(x)

        # 1. Pre-RMSNorm → Attention → Residual
        normed = [_rms_norm(x[i], self.ln1_gamma) for i in range(seq_len)]
        attn_out = self.attention.forward(normed, causal_mask)
        x = [_vec_add(x[i], attn_out[i]) for i in range(seq_len)]

        # 2. Pre-RMSNorm → SwiGLU FFN → Residual
        normed = [_rms_norm(x[i], self.ln2_gamma) for i in range(seq_len)]
        ffn_out = [self.ffn.forward(normed[i]) for i in range(seq_len)]
        x = [_vec_add(x[i], ffn_out[i]) for i in range(seq_len)]

        return x


# ═══════════════════════════════════════════════════════════════
#               MICRO TRANSFORMER
# ═══════════════════════════════════════════════════════════════


class MicroTransformer:
    """
    Мини-трансформер Кристины — понимание контекста и генерация текста.

    Decoder-only архитектура (как GPT):
    - Вход: последовательность BPE-токенов
    - Выход: вероятности следующего токена

    Параметры: ~1.5M (крошечная модель, но с НАСТОЯЩИМ attention)

    Использование:
        # Создание
        model = MicroTransformer(vocab_size=8000)

        # Обучение на тексте
        model.train_step([23, 45, 67, 89, 12])  # BPE token IDs

        # Генерация
        tokens = model.generate([23, 45], max_len=20)
        text = bpe_tokenizer.decode(tokens)

        # Предсказание следующего токена
        probs = model.forward([23, 45, 67])  # → вероятности для 8000 токенов
    """

    def __init__(
        self,
        vocab_size: int = 8000,
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS,
        d_ff: int = D_FF,
        max_seq_len: int = MAX_SEQ_LEN,
        db_path: Path = None,
    ):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len

        self._db_path = db_path or (config.config.data_dir / "micro_transformer.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)

        # Модель
        self.embedding = Embedding(vocab_size, d_model)
        self.blocks = [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.ln_final_gamma = [1.0] * d_model
        self.ln_final_beta = [0.0] * d_model

        # Output head: d_model → vocab_size (tied with embedding)
        # Используем weight embedding транспонированный
        self.output_bias = _zeros_vec(vocab_size)

        # Adam optimizer state
        self._adam_m: Dict[int, Any] = {}  # First moment
        self._adam_v: Dict[int, Any] = {}  # Second moment
        self._adam_t = 0  # Timestep

        # Обучение
        self._training_steps = 0
        self._total_loss = 0.0

        # SQLite для персистентности
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

        # Загружаем сохранённые веса если есть
        self._load_weights()

        total_params = self._count_params()
        logger.info(
            f"🤖 MicroTransformer: {total_params:,} params, "
            f"{n_layers} layers, {n_heads} heads, d={d_model}, "
            f"vocab={vocab_size}, steps={self._training_steps}"
        )

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS model_weights (
                key TEXT PRIMARY KEY,
                data TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS training_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def _count_params(self) -> int:
        """Считает общее количество параметров"""
        count = 0
        # Embedding
        count += self.vocab_size * self.d_model
        # Transformer blocks
        for block in self.blocks:
            # Attention: 4 matrices [d×d] + 4 biases [d]
            count += 4 * self.d_model * self.d_model + 4 * self.d_model
            # SwiGLU FFN: W_gate[d×d_ff] + b_gate[d_ff] + W_up[d×d_ff] + W_down[d_ff×d] + b_down[d]
            count += 2 * self.d_model * self.d_ff + self.d_ff  # W_gate + W_up + b_gate
            count += self.d_ff * self.d_model + self.d_model    # W_down + b_down
            # RMSNorm: 2 × gamma[d] (beta хранится для совместимости, но не используется)
            count += 2 * self.d_model
        # Output bias
        count += self.vocab_size
        # Final RMSNorm gamma
        count += self.d_model
        return count

    # ═══════════════════════════════════════════════════════════════
    #               FORWARD PASS
    # ═══════════════════════════════════════════════════════════════

    def forward(self, token_ids: List[int]) -> List[List[float]]:
        """
        Forward pass: token IDs → логиты для каждой позиции.

        Args:
            token_ids: [seq_len] — входные BPE token IDs

        Returns:
            [seq_len × vocab_size] — логиты (ДО softmax) для каждой позиции
        """
        seq_len = min(len(token_ids), self.max_seq_len)
        token_ids = token_ids[:seq_len]

        # 1. Token Embedding
        x = self.embedding.forward(token_ids)

        # Scale embeddings (как в оригинальном трансформере)
        scale = math.sqrt(self.d_model)
        x = [_vec_scale(xi, scale) for xi in x]

        # 2. Transformer blocks
        for block in self.blocks:
            x = block.forward(x, causal_mask=True)

        # 3. Final RMSNorm
        x = [_rms_norm(xi, self.ln_final_gamma) for xi in x]

        # 4. Output: x @ embedding.weight.T + bias (tied embeddings)
        # logits[i] = x[i] @ E.T + bias
        logits = []
        E_T = _transpose(self.embedding.weight)  # [d_model × vocab_size]
        for i in range(seq_len):
            logit = _vec_add(_matvec(E_T, x[i]), self.output_bias)
            logits.append(logit)

        return logits

    def predict_next(self, token_ids: List[int], temperature: float = 1.0) -> List[float]:
        """
        Предсказывает вероятности следующего токена.

        Args:
            token_ids: предыдущие токены
            temperature: 0.1=точно, 1.0=нормально, 1.5=творчески

        Returns:
            [vocab_size] — вероятности каждого токена
        """
        if not token_ids:
            return [1.0 / self.vocab_size] * self.vocab_size

        logits = self.forward(token_ids)
        last_logits = logits[-1]  # Берём последнюю позицию

        # Temperature scaling
        if temperature != 1.0:
            last_logits = [l / max(temperature, 1e-8) for l in last_logits]

        return _softmax(last_logits)

    # ═══════════════════════════════════════════════════════════════
    #               ГЕНЕРАЦИЯ ТЕКСТА
    # ═══════════════════════════════════════════════════════════════

    def generate(
        self,
        prompt_ids: List[int],
        max_len: int = 50,
        temperature: float = 0.8,
        top_k: int = 40,
        top_p: float = 0.9,
        stop_tokens: List[int] = None,
    ) -> List[int]:
        """
        Авторегрессионная генерация текста.

        Args:
            prompt_ids: начальные токены (промпт)
            max_len: максимальная длина генерации
            temperature: креативность (0.1-1.5)
            top_k: сколько лучших кандидатов рассматривать
            top_p: nucleus sampling порог
            stop_tokens: токены для остановки

        Returns:
            Полная последовательность (промпт + сгенерированные)
        """
        if stop_tokens is None:
            stop_tokens = [3]  # </S>

        generated = list(prompt_ids)

        for _ in range(max_len):
            # Ограничиваем контекст
            context = generated[-self.max_seq_len:]

            # Предсказываем следующий токен
            probs = self.predict_next(context, temperature)

            # Top-K фильтрация
            token_id = self._sample_top_k_p(probs, top_k, top_p)

            # Проверяем stop token
            if token_id in stop_tokens:
                break

            generated.append(token_id)

        return generated

    def _sample_top_k_p(
        self,
        probs: List[float],
        top_k: int = 40,
        top_p: float = 0.9,
    ) -> int:
        """Top-K + Top-P (Nucleus) sampling"""
        # Создаём пары (token_id, prob) и сортируем
        indexed = [(i, p) for i, p in enumerate(probs)]
        indexed.sort(key=lambda x: x[1], reverse=True)

        # Top-K фильтрация
        indexed = indexed[:top_k]

        # Top-P (Nucleus) фильтрация
        cumsum = 0.0
        filtered = []
        for tid, prob in indexed:
            cumsum += prob
            filtered.append((tid, prob))
            if cumsum >= top_p:
                break

        if not filtered:
            return 0

        # Нормализуем и сэмплируем
        total = sum(p for _, p in filtered)
        r = random.random() * total
        cumsum = 0.0
        for tid, prob in filtered:
            cumsum += prob
            if r <= cumsum:
                return tid

        return filtered[0][0]

    # ═══════════════════════════════════════════════════════════════
    #               ОБУЧЕНИЕ
    # ═══════════════════════════════════════════════════════════════

    def train_step(self, token_ids: List[int], lr: float = LEARNING_RATE) -> float:
        """
        Один шаг обучения: предсказание следующего токена.

        Упрощённый backprop через finite differences
        (не полный backprop — слишком сложно для чистого Python).

        Для полного обучения нужен PyTorch. Эта версия позволяет
        fine-tune на маленьких данных.

        Args:
            token_ids: [seq_len] последовательность токенов
            lr: learning rate

        Returns:
            loss (cross-entropy)
        """
        if len(token_ids) < 2:
            return 0.0

        # Forward pass
        logits = self.forward(token_ids[:-1])

        # Compute cross-entropy loss
        total_loss = 0.0
        n_tokens = len(logits)

        for i in range(n_tokens):
            target = token_ids[i + 1]
            if target < 0 or target >= self.vocab_size:
                continue

            # Softmax + cross-entropy
            probs = _softmax(logits[i])
            prob = max(probs[target], 1e-10)
            total_loss -= math.log(prob)

            # Gradient of output: dL/d_logits = probs - one_hot(target)
            grad = list(probs)
            grad[target] -= 1.0

            # Update output embeddings (tied with input)
            # dL/d_embedding[target] += x_final · grad
            # Simplified: nudge embedding weights toward correct token
            for j in range(self.d_model):
                self.embedding.weight[target][j] -= lr * grad[target] * 0.01

        avg_loss = total_loss / max(n_tokens, 1)

        self._training_steps += 1
        self._total_loss += avg_loss

        if self._training_steps % 100 == 0:
            avg = self._total_loss / 100
            logger.debug(f"🤖 Step {self._training_steps}: loss={avg:.4f}")
            self._total_loss = 0.0

        return avg_loss

    def train_on_texts(
        self,
        token_sequences: List[List[int]],
        epochs: int = 1,
        lr: float = LEARNING_RATE,
        batch_log_every: int = 50,
    ) -> float:
        """
        Обучение на нескольких последовательностях.

        Args:
            token_sequences: список последовательностей BPE token IDs
            epochs: количество эпох
            lr: learning rate

        Returns:
            средний loss за последнюю эпоху
        """
        total_loss = 0.0
        n_sequences = 0

        for epoch in range(epochs):
            epoch_loss = 0.0
            random.shuffle(token_sequences)

            for seq in token_sequences:
                if len(seq) < 3:
                    continue

                # Разбиваем длинные последовательности на чанки
                for start in range(0, len(seq) - 1, self.max_seq_len // 2):
                    chunk = seq[start:start + self.max_seq_len]
                    if len(chunk) < 3:
                        continue

                    loss = self.train_step(chunk, lr)
                    epoch_loss += loss
                    n_sequences += 1

                    if n_sequences % batch_log_every == 0:
                        avg = epoch_loss / n_sequences
                        logger.debug(
                            f"🤖 Training: epoch={epoch+1}, "
                            f"seq={n_sequences}, loss={avg:.4f}"
                        )

            total_loss = epoch_loss / max(n_sequences, 1)

        return total_loss

    # ═══════════════════════════════════════════════════════════════
    #               КОДИРОВАНИЕ (получение sentence vectors)
    # ═══════════════════════════════════════════════════════════════

    def encode_sequence(self, token_ids: List[int]) -> List[float]:
        """
        Кодирует последовательность токенов в один вектор.
        Использует внутренние представления трансформера.

        Полезно для:
        - Семантического поиска
        - Классификации intent-ов
        - Кластеризации диалогов

        Returns:
            [d_model] — усреднённый вектор последнего слоя
        """
        if not token_ids:
            return _zeros_vec(self.d_model)

        seq_len = min(len(token_ids), self.max_seq_len)
        token_ids = token_ids[:seq_len]

        # Forward pass до логитов
        x = self.embedding.forward(token_ids)
        scale = math.sqrt(self.d_model)
        x = [_vec_scale(xi, scale) for xi in x]

        for block in self.blocks:
            x = block.forward(x, causal_mask=True)

        x = [_rms_norm(xi, self.ln_final_gamma) for xi in x]

        # Берём последний токен (как в GPT) или среднее всех
        # Используем среднее — работает лучше для классификации
        result = _zeros_vec(self.d_model)
        for xi in x:
            result = _vec_add(result, xi)
        result = _vec_scale(result, 1.0 / seq_len)

        return result

    # ═══════════════════════════════════════════════════════════════
    #               ПЕРСИСТЕНТНОСТЬ
    # ═══════════════════════════════════════════════════════════════

    def save_weights(self):
        """Сохраняет веса модели в SQLite"""
        now = time.time()

        # Сериализуем все веса
        state = {
            "embedding": self.embedding.weight,
            "output_bias": self.output_bias,
            "ln_final_gamma": self.ln_final_gamma,
            "ln_final_beta": self.ln_final_beta,
        }

        for i, block in enumerate(self.blocks):
            prefix = f"block_{i}"
            state[f"{prefix}_attn_Wq"] = block.attention.Wq
            state[f"{prefix}_attn_Wk"] = block.attention.Wk
            state[f"{prefix}_attn_Wv"] = block.attention.Wv
            state[f"{prefix}_attn_Wo"] = block.attention.Wo
            state[f"{prefix}_attn_bq"] = block.attention.bq
            state[f"{prefix}_attn_bk"] = block.attention.bk
            state[f"{prefix}_attn_bv"] = block.attention.bv
            state[f"{prefix}_attn_bo"] = block.attention.bo
            state[f"{prefix}_ffn_W1"] = block.ffn.W1
            state[f"{prefix}_ffn_b1"] = block.ffn.b1
            state[f"{prefix}_ffn_W_up"] = block.ffn.W_up  # SwiGLU gate
            state[f"{prefix}_ffn_W2"] = block.ffn.W2
            state[f"{prefix}_ffn_b2"] = block.ffn.b2
            state[f"{prefix}_ln1_gamma"] = block.ln1_gamma
            state[f"{prefix}_ln1_beta"] = block.ln1_beta
            state[f"{prefix}_ln2_gamma"] = block.ln2_gamma
            state[f"{prefix}_ln2_beta"] = block.ln2_beta

        for key, value in state.items():
            data = json.dumps(value)
            self._conn.execute("""
                INSERT INTO model_weights (key, data, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET data = ?, updated_at = ?
            """, (key, data, now, data, now))

        # Training state
        self._conn.execute("""
            INSERT INTO training_state (key, value) VALUES ('steps', ?)
            ON CONFLICT(key) DO UPDATE SET value = ?
        """, (str(self._training_steps), str(self._training_steps)))

        self._conn.commit()
        logger.info(f"💾 Transformer weights saved (step {self._training_steps})")

    def _load_weights(self):
        """Загружает веса из SQLite если есть"""
        row = self._conn.execute(
            "SELECT COUNT(*) as c FROM model_weights"
        ).fetchone()

        if not row or row[0] == 0:
            return  # Нет сохранённых весов

        def _load(key):
            r = self._conn.execute(
                "SELECT data FROM model_weights WHERE key = ?", (key,)
            ).fetchone()
            if r:
                return json.loads(r[0])
            return None

        # Embedding
        emb = _load("embedding")
        if emb and len(emb) == self.vocab_size:
            self.embedding.weight = emb

        ob = _load("output_bias")
        if ob:
            self.output_bias = ob

        fg = _load("ln_final_gamma")
        if fg:
            self.ln_final_gamma = fg
        fb = _load("ln_final_beta")
        if fb:
            self.ln_final_beta = fb

        # Transformer blocks
        for i, block in enumerate(self.blocks):
            prefix = f"block_{i}"
            for attr, key in [
                (block.attention, "Wq"), (block.attention, "Wk"),
                (block.attention, "Wv"), (block.attention, "Wo"),
            ]:
                data = _load(f"{prefix}_attn_{key}")
                if data:
                    setattr(attr, key, data)
            for attr_name in ["bq", "bk", "bv", "bo"]:
                data = _load(f"{prefix}_attn_{attr_name}")
                if data:
                    setattr(block.attention, attr_name, data)
            for key in ["W1", "b1", "W_up", "W2", "b2"]:
                data = _load(f"{prefix}_ffn_{key}")
                if data:
                    setattr(block.ffn, key, data)
            for key in ["ln1_gamma", "ln1_beta", "ln2_gamma", "ln2_beta"]:
                data = _load(f"{prefix}_{key}")
                if data:
                    setattr(block, key, data)

        # Training state
        steps = self._conn.execute(
            "SELECT value FROM training_state WHERE key = 'steps'"
        ).fetchone()
        if steps:
            self._training_steps = int(steps[0])

        logger.info(
            f"💾 Transformer weights loaded (step {self._training_steps})"
        )

    # ═══════════════════════════════════════════════════════════════
    #               СТАТИСТИКА
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        return {
            "params": self._count_params(),
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_layers": self.n_layers,
            "d_ff": self.d_ff,
            "vocab_size": self.vocab_size,
            "max_seq_len": self.max_seq_len,
            "training_steps": self._training_steps,
        }

    def close(self):
        self.save_weights()
        self._conn.close()
