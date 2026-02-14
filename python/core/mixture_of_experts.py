"""
Кристина 7.3 — Mixture of Experts (Смесь экспертов)

ЗАЧЕМ:
  Один большой трансформер знает всё поверхностно.
  Несколько МАЛЕНЬКИХ трансформеров-специалистов знают свою область ГЛУБОКО.

  Router решает, какому эксперту отдать запрос:

    "Напиши функцию сортировки" → Expert: CODE (90%) + GENERAL (10%)
    "Как дела?"                 → Expert: CHAT (95%)
    "Проанализируй данные"     → Expert: ANALYSIS (80%) + CODE (20%)

АРХИТЕКТУРА:
  ┌─────────────────────────────────────────────┐
  │ Input: "Напиши функцию сортировки"          │
  │         ↓                                    │
  │ Sentence Embedding → [d_model]               │
  │         ↓                                    │
  │ ┌─────────────────┐                          │
  │ │   Router (MLP)  │ → [0.9, 0.02, 0.08, ...]│
  │ └────┬───────┬────┘                          │
  │      ↓       ↓                               │
  │  ┌────────┐ ┌────────┐                       │
  │  │Expert 1│ │Expert 2│  (top-K=2 активных)   │
  │  │ CODE   │ │GENERAL │                       │
  │  └───┬────┘ └───┬────┘                       │
  │      ↓          ↓                            │
  │  0.9 × out1 + 0.1 × out2  (weighted merge)  │
  │         ↓                                    │
  │  Final output                                │
  └─────────────────────────────────────────────┘

ЭКСПЕРТЫ:
  Каждый эксперт — это ЛЁГКИЙ FFN (Feed-Forward Network):
  - d_model → d_expert → d_model
  - Специализируется на своём типе данных
  - Обучается только когда Router его активирует

  НЕ полный трансформер! Эксперты работают как СПЕЦИАЛИЗИРОВАННЫЙ FFN-слой
  внутри MicroTransformer pipeline.

LOAD BALANCING:
  Без балансировки Router может схлопнуться к 1 эксперту.
  Используем auxiliary loss: штраф за неравномерную загрузку.
"""

import json
import math
import random
import sqlite3
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from utils.logging import get_logger
import config

logger = get_logger("mixture_of_experts")


# ═══════════════════════════════════════════════════════════════
#               КОНФИГУРАЦИЯ
# ═══════════════════════════════════════════════════════════════

NUM_EXPERTS = 6         # Количество экспертов
TOP_K_EXPERTS = 2       # Сколько экспертов активируется на запрос
D_EXPERT = 256          # Скрытый размер каждого эксперта
D_MODEL = 128           # Размерность входа/выхода (== MicroTransformer d_model)
BALANCE_COEFF = 0.01    # Коэффициент load balancing loss
ROUTER_LR = 1e-3        # Learning rate для Router

# Имена экспертов (семантические)
EXPERT_NAMES = [
    "chat",       # Общение, приветствия, small talk
    "code",       # Программирование, алгоритмы
    "analysis",   # Анализ, сравнение, данные
    "creative",   # Творчество, генерация текста
    "system",     # Системные задачи, команды
    "knowledge",  # Факты, объяснения, обучение
]

# Ключевые слова для начальной маршрутизации (до обучения Router)
EXPERT_KEYWORDS = {
    "chat": ["привет", "как дела", "здравствуй", "пока", "спасибо",
             "доброе утро", "добрый вечер", "как ты"],
    "code": ["код", "python", "функция", "класс", "программа", "скрипт",
             "баг", "ошибка", "api", "git", "алгоритм", "сортировка",
             "рекурсия", "массив", "переменная", "цикл"],
    "analysis": ["анализ", "сравни", "статистика", "данные", "отчёт",
                 "тренд", "метрика", "процент", "график"],
    "creative": ["напиши стих", "история", "сказка", "придумай",
                 "фантазия", "рассказ", "поэма", "песня"],
    "system": ["запусти", "установи", "настрой", "терминал", "сервер",
               "docker", "процесс", "файл", "папка", "команда"],
    "knowledge": ["объясни", "расскажи", "что такое", "почему", "зачем",
                  "как работает", "определение", "принцип"],
}


# ═══════════════════════════════════════════════════════════════
#               ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ═══════════════════════════════════════════════════════════════

def _zeros(n: int) -> List[float]:
    return [0.0] * n

def _randn(n: int, scale: float = 0.02) -> List[float]:
    return [random.gauss(0, scale) for _ in range(n)]

def _randn_matrix(rows: int, cols: int, scale: float = 0.02) -> List[List[float]]:
    return [_randn(cols, scale) for _ in range(rows)]

def _dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def _matvec(mat: List[List[float]], vec: List[float]) -> List[float]:
    return [_dot(row, vec) for row in mat]

def _relu(x: List[float]) -> List[float]:
    return [max(0.0, v) for v in x]

def _softmax(x: List[float]) -> List[float]:
    max_x = max(x) if x else 0.0
    exp_x = [math.exp(v - max_x) for v in x]
    s = sum(exp_x) + 1e-10
    return [v / s for v in exp_x]

def _vec_add(a: List[float], b: List[float]) -> List[float]:
    return [x + y for x, y in zip(a, b)]

def _vec_scale(a: List[float], s: float) -> List[float]:
    return [x * s for x in a]


# ═══════════════════════════════════════════════════════════════
#               EXPERT (FFN-специалист)
# ═══════════════════════════════════════════════════════════════


class Expert:
    """
    Один эксперт — двухслойный FFN:
      input [d_model] → W1 → ReLU → W2 → output [d_model]

    Каждый эксперт специализируется на своём типе задач.
    """

    def __init__(self, name: str, d_model: int = D_MODEL, d_expert: int = D_EXPERT):
        self.name = name
        self.d_model = d_model
        self.d_expert = d_expert

        # Weights
        scale = math.sqrt(2.0 / d_model)  # He init
        self.W1 = _randn_matrix(d_expert, d_model, scale)
        self.b1 = _zeros(d_expert)
        self.W2 = _randn_matrix(d_model, d_expert, math.sqrt(2.0 / d_expert))
        self.b2 = _zeros(d_model)

        # Stats
        self.activations = 0
        self.total_weight = 0.0

    def forward(self, x: List[float]) -> List[float]:
        """
        Forward pass: x [d_model] → output [d_model]
        """
        # Layer 1: x @ W1.T + b1 → ReLU
        hidden = _matvec(self.W1, x)
        hidden = _vec_add(hidden, self.b1)
        hidden = _relu(hidden)

        # Layer 2: hidden @ W2.T + b2
        output = _matvec(self.W2, hidden)
        output = _vec_add(output, self.b2)

        return output

    def get_params(self) -> Dict:
        return {
            "W1": self.W1, "b1": self.b1,
            "W2": self.W2, "b2": self.b2,
        }

    def load_params(self, data: Dict):
        if "W1" in data and len(data["W1"]) == self.d_expert:
            self.W1 = data["W1"]
            self.b1 = data["b1"]
            self.W2 = data["W2"]
            self.b2 = data["b2"]

    def param_count(self) -> int:
        return (self.d_model * self.d_expert + self.d_expert +
                self.d_expert * self.d_model + self.d_model)


# ═══════════════════════════════════════════════════════════════
#               ROUTER (Маршрутизатор)
# ═══════════════════════════════════════════════════════════════


class Router:
    """
    Router: решает, каких экспертов активировать.

    input [d_model] → W_router → softmax → gate weights [num_experts]

    Top-K gating: активируются только K экспертов с наибольшими весами.
    Остальные = 0 (sparse activation для эффективности).
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        num_experts: int = NUM_EXPERTS,
        top_k: int = TOP_K_EXPERTS,
    ):
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # Router weights: d_model → num_experts
        scale = math.sqrt(1.0 / d_model)
        self.W_gate = _randn_matrix(num_experts, d_model, scale)
        self.b_gate = _zeros(num_experts)

        # Noise for exploration (помогает не застрять на 1 эксперте)
        self._noise_scale = 0.1

        # Stats per expert
        self._routing_counts = [0] * num_experts

    def route(
        self,
        x: List[float],
        training: bool = False,
    ) -> List[Tuple[int, float]]:
        """
        Определяет top-K экспертов и их веса.

        Args:
            x: input vector [d_model]
            training: если True, добавляет noise для exploration

        Returns:
            [(expert_idx, gate_weight), ...] — top-K экспертов
        """
        # Logits: x @ W_gate.T + b
        logits = _matvec(self.W_gate, x)
        logits = _vec_add(logits, self.b_gate)

        # Add noise during training
        if training and self._noise_scale > 0:
            noise = _randn(self.num_experts, self._noise_scale)
            logits = _vec_add(logits, noise)

        # Softmax для вероятностей
        probs = _softmax(logits)

        # Top-K selection
        indexed = [(i, p) for i, p in enumerate(probs)]
        indexed.sort(key=lambda x: x[1], reverse=True)
        top_k = indexed[:self.top_k]

        # Re-normalize top-K weights
        total = sum(w for _, w in top_k) + 1e-10
        top_k = [(idx, w / total) for idx, w in top_k]

        # Track routing counts
        for idx, _ in top_k:
            self._routing_counts[idx] += 1

        return top_k

    def compute_balance_loss(self) -> float:
        """
        Auxiliary loss: штрафует за неравномерную загрузку экспертов.

        Идеальная загрузка: каждый эксперт = 1/num_experts запросов.
        """
        total = sum(self._routing_counts) + 1e-10
        fractions = [c / total for c in self._routing_counts]
        ideal = 1.0 / self.num_experts

        # CV (coefficient of variation) как мера дисбаланса
        variance = sum((f - ideal) ** 2 for f in fractions) / self.num_experts
        balance_loss = variance * self.num_experts * BALANCE_COEFF

        return balance_loss

    def get_params(self) -> Dict:
        return {"W_gate": self.W_gate, "b_gate": self.b_gate}

    def load_params(self, data: Dict):
        if "W_gate" in data and len(data["W_gate"]) == self.num_experts:
            self.W_gate = data["W_gate"]
            self.b_gate = data["b_gate"]


# ═══════════════════════════════════════════════════════════════
#               MIXTURE OF EXPERTS
# ═══════════════════════════════════════════════════════════════


class MixtureOfExperts:
    """
    Mixture of Experts: Router + набор специализированных FFN-экспертов.

    Использование:
        moe = MixtureOfExperts()

        # Маршрутизация + обработка
        output, routing = moe.forward(input_vec)
        # output: [d_model], routing: [(expert_idx, weight), ...]

        # Маршрутизация по тексту (с keyword fallback)
        output, routing = moe.process_text(
            text="Напиши функцию сортировки",
            input_vec=embedding,
        )

        # Обучение (с gradient from output)
        moe.train_step(input_vec, target_vec)
    """

    def __init__(
        self,
        d_model: int = D_MODEL,
        d_expert: int = D_EXPERT,
        num_experts: int = NUM_EXPERTS,
        top_k: int = TOP_K_EXPERTS,
        db_path: Path = None,
    ):
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k

        # Router
        self.router = Router(d_model, num_experts, top_k)

        # Experts
        self.experts: List[Expert] = []
        for i in range(num_experts):
            name = EXPERT_NAMES[i] if i < len(EXPERT_NAMES) else f"expert_{i}"
            self.experts.append(Expert(name, d_model, d_expert))

        # Persistence
        self._db_path = db_path or (config.config.data_dir / "mixture_of_experts.db")
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_tables()

        # Stats
        self._total_forwards = 0
        self._total_trains = 0
        self._load_state()

        total_params = self._count_params()
        logger.info(
            f"🧠 MoE: {num_experts} experts × {d_expert}d, "
            f"top-{top_k}, {total_params:,} params, "
            f"{self._total_forwards} forwards"
        )

    def _create_tables(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS moe_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)
        self._conn.commit()

    def _count_params(self) -> int:
        # Router
        count = self.d_model * self.num_experts + self.num_experts
        # Experts
        for expert in self.experts:
            count += expert.param_count()
        return count

    def _load_state(self):
        row = self._conn.execute(
            "SELECT value FROM moe_state WHERE key = 'model_data'"
        ).fetchone()
        if row:
            try:
                data = json.loads(row[0])
                self._total_forwards = data.get("total_forwards", 0)
                self._total_trains = data.get("total_trains", 0)
                if "router" in data:
                    self.router.load_params(data["router"])
                if "experts" in data:
                    for i, expert_data in enumerate(data["experts"]):
                        if i < len(self.experts):
                            self.experts[i].load_params(expert_data)
            except (json.JSONDecodeError, TypeError):
                pass

    def _save_state(self):
        data = {
            "total_forwards": self._total_forwards,
            "total_trains": self._total_trains,
            "router": self.router.get_params(),
            "experts": [e.get_params() for e in self.experts],
        }
        json_str = json.dumps(data)
        self._conn.execute("""
            INSERT INTO moe_state (key, value) VALUES ('model_data', ?)
            ON CONFLICT(key) DO UPDATE SET value = ?
        """, (json_str, json_str))
        self._conn.commit()

    # ═══════════════════════════════════════════════════════════════
    #           FORWARD PASS
    # ═══════════════════════════════════════════════════════════════

    def forward(
        self,
        x: List[float],
        training: bool = False,
    ) -> Tuple[List[float], List[Tuple[int, float]]]:
        """
        MoE forward pass.

        1. Router определяет top-K экспертов
        2. Каждый активный эксперт обрабатывает вход
        3. Результаты взвешенно суммируются

        Args:
            x: input [d_model]
            training: add noise to router

        Returns:
            (output [d_model], routing [(expert_idx, weight)])
        """
        # 1. Route
        routing = self.router.route(x, training=training)

        # 2. Forward through active experts
        output = _zeros(self.d_model)

        for expert_idx, gate_weight in routing:
            expert = self.experts[expert_idx]
            expert_output = expert.forward(x)

            # Weighted accumulation
            for i in range(self.d_model):
                output[i] += expert_output[i] * gate_weight

            # Track
            expert.activations += 1
            expert.total_weight += gate_weight

        self._total_forwards += 1

        # 3. Residual connection
        output = _vec_add(output, x)

        return output, routing

    # ═══════════════════════════════════════════════════════════════
    #           TEXT-LEVEL INTERFACE
    # ═══════════════════════════════════════════════════════════════

    def process_text(
        self,
        text: str,
        input_vec: List[float],
        training: bool = False,
    ) -> Tuple[List[float], List[Tuple[int, float]]]:
        """
        Обрабатывает текст через MoE с keyword-based routing hint.

        Для первых N запросов (пока Router не обучен) добавляет
        keyword bias к Router logits.
        """
        # Keyword-based bias (помогает Router на старте)
        keyword_bias = self._compute_keyword_bias(text)

        if keyword_bias and self._total_trains < 200:
            # Добавляем bias к Router gate logits
            orig_b = list(self.router.b_gate)
            for i, bias in enumerate(keyword_bias):
                if i < len(self.router.b_gate):
                    self.router.b_gate[i] += bias * max(0, 1.0 - self._total_trains / 200)

            output, routing = self.forward(input_vec, training=training)

            # Restore
            self.router.b_gate = orig_b
        else:
            output, routing = self.forward(input_vec, training=training)

        return output, routing

    def _compute_keyword_bias(self, text: str) -> Optional[List[float]]:
        """Вычисляет bias для Router на основе ключевых слов"""
        text_lower = text.lower()
        bias = _zeros(self.num_experts)
        has_match = False

        for i, expert in enumerate(self.experts):
            keywords = EXPERT_KEYWORDS.get(expert.name, [])
            score = sum(1 for kw in keywords if kw in text_lower)
            if score > 0:
                bias[i] = score * 0.5  # Мягкий bias
                has_match = True

        return bias if has_match else None

    def get_expert_for_text(self, text: str, input_vec: List[float]) -> str:
        """Возвращает имя основного эксперта для текста"""
        _, routing = self.process_text(text, input_vec)
        if routing:
            return self.experts[routing[0][0]].name
        return "unknown"

    # ═══════════════════════════════════════════════════════════════
    #           ОБУЧЕНИЕ
    # ═══════════════════════════════════════════════════════════════

    def train_step(
        self,
        x: List[float],
        target: List[float],
        lr: float = ROUTER_LR,
    ) -> float:
        """
        Обучение MoE: экспертов + Router.

        Gradient-free обучение (эволюционное):
        1. Forward → output
        2. Ошибка = output - target
        3. Обновляем веса активных экспертов пропорционально ошибке
        4. Router обновляется через reward: лучше маршрутил → reward

        Args:
            x: input [d_model]
            target: target output [d_model]
            lr: learning rate

        Returns:
            loss (MSE)
        """
        # Forward
        output, routing = self.forward(x, training=True)

        # Loss
        error = [output[i] - target[i] for i in range(self.d_model)]
        loss = sum(e * e for e in error) / self.d_model

        # Update active experts (gradient-free)
        for expert_idx, gate_weight in routing:
            expert = self.experts[expert_idx]
            self._update_expert(expert, x, error, lr * gate_weight)

        # Update router (reward-based)
        self._update_router(routing, loss, lr)

        self._total_trains += 1

        # Periodic save
        if self._total_trains % 50 == 0:
            self._save_state()

        return loss

    def _update_expert(
        self,
        expert: Expert,
        x: List[float],
        error: List[float],
        lr: float,
    ):
        """
        Обновляет веса эксперта (упрощённый gradient descent).

        Для FFN: output = W2 @ relu(W1 @ x + b1) + b2
        Gradient: dW2 ≈ error @ hidden.T, dW1 ≈ (W2.T @ error) ⊙ relu'(h) @ x.T
        """
        # Forward to get hidden
        hidden_raw = _matvec(expert.W1, x)
        hidden_raw = _vec_add(hidden_raw, expert.b1)
        hidden = _relu(hidden_raw)

        # Update W2: -= lr * error @ hidden.T
        for i in range(expert.d_model):
            for j in range(expert.d_expert):
                expert.W2[i][j] -= lr * error[i] * hidden[j]
            expert.b2[i] -= lr * error[i]

        # Backprop through W2 → hidden gradient
        hidden_grad = _zeros(expert.d_expert)
        for j in range(expert.d_expert):
            for i in range(expert.d_model):
                hidden_grad[j] += expert.W2[i][j] * error[i]

        # ReLU gradient
        for j in range(expert.d_expert):
            if hidden_raw[j] <= 0:
                hidden_grad[j] = 0.0

        # Update W1: -= lr * hidden_grad @ x.T
        for j in range(expert.d_expert):
            for k in range(expert.d_model):
                expert.W1[j][k] -= lr * hidden_grad[j] * x[k]
            expert.b1[j] -= lr * hidden_grad[j]

    def _update_router(
        self,
        routing: List[Tuple[int, float]],
        loss: float,
        lr: float,
    ):
        """
        Обновляет Router на основе loss.

        Reward signal: маленький loss → усиливаем текущую маршрутизацию,
                       большой loss → ослабляем.
        """
        # Reward = -loss (чем меньше loss, тем лучше)
        reward = math.exp(-loss) - 0.5  # Центрируем вокруг 0

        for expert_idx, gate_weight in routing:
            # Усиливаем/ослабляем маршрутизацию
            adjustment = lr * reward * 0.1
            self.router.b_gate[expert_idx] += adjustment

        # Balance loss: штрафуем неравномерность
        balance_loss = self.router.compute_balance_loss()
        if balance_loss > 0.01:
            total = sum(self.router._routing_counts) + 1e-10
            ideal = total / self.num_experts
            for i in range(self.num_experts):
                excess = (self.router._routing_counts[i] - ideal) / total
                self.router.b_gate[i] -= lr * excess * BALANCE_COEFF

    # ═══════════════════════════════════════════════════════════════
    #           STATISTICS
    # ═══════════════════════════════════════════════════════════════

    def get_stats(self) -> Dict:
        expert_stats = []
        for i, expert in enumerate(self.experts):
            expert_stats.append({
                "name": expert.name,
                "activations": expert.activations,
                "avg_weight": expert.total_weight / max(1, expert.activations),
            })

        # Sort by activations (most active first)
        expert_stats.sort(key=lambda x: x["activations"], reverse=True)

        total_routes = sum(self.router._routing_counts)
        routing_distribution = {}
        for i, count in enumerate(self.router._routing_counts):
            name = self.experts[i].name if i < len(self.experts) else f"expert_{i}"
            routing_distribution[name] = round(count / max(1, total_routes), 3)

        return {
            "total_forwards": self._total_forwards,
            "total_trains": self._total_trains,
            "num_experts": self.num_experts,
            "top_k": self.top_k,
            "experts": expert_stats,
            "routing_distribution": routing_distribution,
            "balance_loss": round(self.router.compute_balance_loss(), 6),
        }

    def close(self):
        self._save_state()
        self._conn.close()
