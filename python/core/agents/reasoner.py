"""
Reasoner Agent — логика и математические рассуждения
"""

from typing import Dict, Any
import re

from core.agents.base_agent import BaseAgent
import config

class ReasonerAgent(BaseAgent):
    """
    Логик — математика и сложные рассуждения
    
    Специализация:
    - Математические задачи
    - Логические рассуждения
    - Доказательства
    - Отладка кода
    - Step-by-step решения
    """
    
    def __init__(self):
        # Получаем конфиг из config.py
        model_config = config.AGENT_MODELS["reasoner"]
        
        super().__init__(
            name="reasoner",
            model_config=model_config,
            capabilities=[
                "math_problem",
                "logical_reasoning",
                "proof",
                "code_debugging",
                "step_by_step"
            ],
            description="Логик и математик"
        )
    
    async def execute(self, task: Dict[str, Any]) -> str:
        """
        Выполняет задачу, требующую рассуждений
        
        Args:
            task: {
                "type": "math" | "logic" | "debug",
                "problem": "описание задачи",
                "context": "дополнительный контекст"
            }
        """
        
        task_type = task.get("type")
        problem = task.get("problem", "")
        
        if not problem:
            return "ERROR: Не указана задача"
        
        if task_type == "math":
            return await self._solve_math(problem, task.get("context", ""))
        
        elif task_type == "logic":
            return await self._logical_reasoning(problem, task.get("context", ""))
        
        elif task_type == "debug":
            return await self._debug_code(problem, task.get("code", ""))
        
        else:
            # Общее рассуждение
            return await self._general_reasoning(problem)
    
    async def _solve_math(self, problem: str, context: str = "") -> str:
        """Решает математическую задачу"""
        
        self.logger.info(f"🧮 Решение математической задачи")
        
        prompt = f"""Реши следующую математическую задачу пошагово.

Задача: {problem}

{f'Контекст: {context}' if context else ''}

ТРЕБОВАНИЯ:
1. Распиши решение по шагам
2. Покажи все вычисления
3. Дай финальный ответ
4. Используй русский язык

Формат:
Шаг 1: [описание]
Шаг 2: [описание]
...
Ответ: [результат]"""
        
        messages = [
            {
                "role": "system",
                "content": "Ты математик. Решай задачи пошагово с подробными объяснениями."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            # DeepSeek-R1 автоматически использует Chain-of-Thought
            solution = await self._call_model(messages, temperature=0.1, max_tokens=800)
            
            return solution.strip()
        
        except Exception as e:
            self.logger.error(f"Ошибка решения задачи: {e}")
            return f"ERROR: {str(e)}"
    
    async def _logical_reasoning(self, problem: str, context: str = "") -> str:
        """Логическое рассуждение"""
        
        self.logger.info("🧠 Логическое рассуждение")
        
        prompt = f"""Реши следующую логическую задачу.

Задача: {problem}

{f'Контекст: {context}' if context else ''}

ТРЕБОВАНИЯ:
1. Проанализируй условия
2. Построй логическую цепочку
3. Сделай вывод
4. Объясни своё рассуждение
5. Используй русский язык"""
        
        messages = [
            {
                "role": "system",
                "content": "Ты логик. Рассуждай последовательно и обоснованно."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            reasoning = await self._call_model(messages, temperature=0.2, max_tokens=600)
            
            return reasoning.strip()
        
        except Exception as e:
            self.logger.error(f"Ошибка рассуждения: {e}")
            return f"ERROR: {str(e)}"
    
    async def _debug_code(self, problem: str, code: str) -> str:
        """Отладка кода"""
        
        self.logger.info("🐛 Отладка кода")
        
        prompt = f"""Найди ошибки в коде и предложи исправления.

Проблема: {problem}

Код:
```
{code[:1000]}
```

ТРЕБОВАНИЯ:
1. Найди все ошибки
2. Объясни каждую ошибку
3. Предложи исправленный код
4. Используй русский язык"""
        
        messages = [
            {
                "role": "system",
                "content": "Ты эксперт по отладке кода. Находи ошибки и предлагай решения."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            debug_result = await self._call_model(messages, temperature=0.2, max_tokens=700)
            
            return debug_result.strip()
        
        except Exception as e:
            self.logger.error(f"Ошибка отладки: {e}")
            return f"ERROR: {str(e)}"
    
    async def _general_reasoning(self, problem: str) -> str:
        """Общее рассуждение"""
        
        self.logger.info("💭 Общее рассуждение")
        
        prompt = f"""Проанализируй следующую задачу и дай обоснованный ответ.

Задача: {problem}

Рассуждай пошагово, обосновывай каждый вывод."""
        
        messages = [
            {
                "role": "system",
                "content": "Ты эксперт по аналитическому мышлению."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
        
        try:
            reasoning = await self._call_model(messages, temperature=0.3, max_tokens=600)
            
            return reasoning.strip()
        
        except Exception as e:
            self.logger.error(f"Ошибка рассуждения: {e}")
            return f"ERROR: {str(e)}"