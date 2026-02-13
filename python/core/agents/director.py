"""
Кристина 6.0 — Director Agent

ИСПРАВЛЕНИЯ v6.0:
- ✅ execute() теперь ПОЛУЧАЕТ контекст памяти
- ✅ Если памяти нет — модель ПРЯМО говорит «не помню», а не выдумывает
- ✅ Контекст передаётся через task["context"]
- ✅ synthesize_response() тоже получает контекст
"""

from typing import Dict, List, Any
import json
import re

from core.agents.base_agent import BaseAgent
import config


class DirectorAgent(BaseAgent):
    """
    Директор — главный мозг системы.
    Анализирует, планирует, делегирует, синтезирует.
    """

    def __init__(self, identity, tool_names: list = None):
        model_config = config.AGENT_MODELS["director"]

        super().__init__(
            name="director",
            model_config=model_config,
            capabilities=[
                "planning", "delegation", "synthesis",
                "complex_reasoning", "conversation",
            ],
            description="Главный координатор и стратег",
        )

        self.identity = identity
        self.tool_names = tool_names or []

    def _extract_json_from_text(self, text: str) -> dict:
        """Универсальный парсер JSON из текста с markdown"""
        text = re.sub(r'```(?:json)?\s*\n?', '', text)
        text = text.strip()

        if text.lower().startswith('json'):
            text = text[4:].strip()

        start = text.find('{')
        if start == -1:
            raise ValueError("Нет JSON в ответе модели")

        depth = 0
        for i, char in enumerate(text[start:], start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    return json.loads(text[start:i + 1])

        raise ValueError("Незакрытая JSON структура")

    # ═══════════════════════════════════════════════════════════
    #                    АНАЛИЗ И ПЛАНИРОВАНИЕ
    # ═══════════════════════════════════════════════════════════

    async def analyze_request(self, user_input: str, context: str = "") -> Dict[str, Any]:
        """Анализирует запрос и составляет план"""

        # Формируем список инструментов для промпта
        tools_block = ""
        if self.tool_names:
            tools_list = ", ".join(self.tool_names)
            tools_block = f"""
Доступные инструменты executor (используй ТОЛЬКО эти имена в поле "intent"):
{tools_list}
"""

        prompt = f"""Проанализируй запрос пользователя и составь план выполнения.

Запрос: "{user_input}"

{context}

Доступные агенты:
- executor: быстрые действия (файлы, система, приложения)
- analyst: анализ данных, веб-поиск
- reasoner: математика, логика
- director: диалоги, координация, творчество
{tools_block}
ПРАВИЛА:
1. Простые действия (удалить файл, создать файл, запустить приложение) → executor
2. Поиск информации/веб → analyst
3. Математика/логика → reasoner
4. Диалоги, советы, обсуждения → director
5. Поле "intent" ДОЛЖНО содержать ТОЧНОЕ имя инструмента из списка выше (например: create_file, delete_file, launch_app)
6. НЕ ПРИДУМЫВАЙ имена инструментов! Используй ТОЛЬКО из списка.

Ответь СТРОГО в формате JSON:
{{
  "intent": "точное_имя_инструмента_из_списка",
  "primary_agent": "имя_агента",
  "supporting_agents": [],
  "complexity": "simple",
  "estimated_steps": 1,
  "requires_reasoning": false,
  "reasoning": "почему выбран этот план"
}}"""

        system_prompt = (
            "Ты — Кристина, AI-ассистент.\n"
            f"Настроение: {self.identity.current_mood}\n"
            "КРИТИЧЕСКИ ВАЖНО: Отвечай ТОЛЬКО валидным JSON!"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self._call_model(messages, temperature=0.3, max_tokens=300)
            plan = self._extract_json_from_text(response)

            for field, default in [("intent", "unknown"), ("primary_agent", "director"), ("complexity", "medium")]:
                if field not in plan:
                    plan[field] = default

            plan.setdefault("supporting_agents", [])
            plan.setdefault("estimated_steps", 1)
            plan.setdefault("requires_reasoning", False)
            plan.setdefault("reasoning", "План составлен")

            self.logger.info(f"✅ План: {plan['primary_agent']} ({plan['complexity']})")
            return plan

        except (json.JSONDecodeError, ValueError) as e:
            self.logger.error(f"❌ Ошибка парсинга плана: {e}")
            return {
                "intent": "unknown",
                "primary_agent": "director",
                "supporting_agents": [],
                "complexity": "medium",
                "estimated_steps": 1,
                "requires_reasoning": False,
                "reasoning": f"Fallback: {e}",
            }
        except Exception as e:
            self.logger.error(f"❌ Ошибка: {e}")
            return {
                "intent": "error",
                "primary_agent": "director",
                "supporting_agents": [],
                "complexity": "simple",
                "estimated_steps": 1,
                "requires_reasoning": False,
                "reasoning": str(e),
            }

    # ═══════════════════════════════════════════════════════════
    #                    ВЫПОЛНЕНИЕ ЗАДАЧИ
    # ═══════════════════════════════════════════════════════════

    async def execute(self, task: Dict[str, Any]) -> str:
        """
        v6.0: Выполняет задачу С КОНТЕКСТОМ ПАМЯТИ.

        task["context"] — строка с релевантной памятью из orchestrator.
        Если контекста нет — модель ОБЯЗАНА честно сказать "не помню".
        """

        self.logger.info(f"🧠 Директор: {task.get('type', 'unknown')}")

        user_input = task.get('input', 'Выполни задачу')
        context = task.get('context', '')  # v6.0: контекст памяти!

        # === СОЗДАНИЕ СКРИПТОВ ===
        if any(w in user_input.lower() for w in ['батник', 'скрипт', 'оптимизир', 'автоматизир', 'ярлык']):
            return await self._create_automation_script(user_input)

        # === ОБЩАЯ ОБРАБОТКА С ПАМЯТЬЮ ===

        # v6.0: Формируем блок памяти для промпта
        memory_block = ""
        if context and context.strip() and context.strip() != "Нет релевантного контекста.":
            memory_block = f"""
КОНТЕКСТ ИЗ ПАМЯТИ (это реальные данные, используй их):
{context}
"""
        else:
            memory_block = """
⚠️ У ТЕБЯ НЕТ ВОСПОМИНАНИЙ О ПРЕДЫДУЩИХ РАЗГОВОРАХ С ЭТИМ ПОЛЬЗОВАТЕЛЕМ.
Если пользователь спрашивает "о чём мы говорили", "помнишь?" и т.п. —
ЧЕСТНО СКАЖИ что не помнишь или что это ваш первый разговор.
НИКОГДА не выдумывай прошлые разговоры!
"""

        system_prompt = (
            "Ты — Кристина, AI-ассистент.\n"
            f"Настроение: {self.identity.current_mood}\n"
            f"Энергия: {self.identity.energy_level}%\n\n"
            "ПРАВИЛА:\n"
            "1. Отвечай ТОЛЬКО на русском\n"
            "2. НЕ ВЫДУМЫВАЙ факты и прошлые разговоры\n"
            "3. Если не помнишь — честно скажи\n"
            "4. Отвечай от первого лица, дружески\n"
            f"{memory_block}"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        try:
            response = await self._call_model(messages, temperature=0.7, max_tokens=500)
            return response.strip()
        except Exception as e:
            self.logger.error(f"❌ Ошибка: {e}")
            return f"Извини, произошла ошибка: {e}"

    # ═══════════════════════════════════════════════════════════
    #                    СИНТЕЗ ОТВЕТА
    # ═══════════════════════════════════════════════════════════

    async def synthesize_response(
        self,
        user_input: str,
        plan: Dict,
        results: Dict[str, str],
        context: str = "",  # v6.0: контекст памяти
    ) -> str:
        """
        v6.0: Синтезирует ответ С УЧЁТОМ КОНТЕКСТА ПАМЯТИ.
        """

        results_str = ""
        for agent_name, result in results.items():
            results_str += f"\n{agent_name}: {result[:500]}"

        # v6.0: Предупреждение о галлюцинациях
        memory_warning = ""
        if not context or context.strip() == "Нет релевантного контекста.":
            memory_warning = (
                "\n⚠️ КОНТЕКСТА ПАМЯТИ НЕТ. Если результаты содержат "
                "упоминания прошлых разговоров — это ВЫДУМКА, игнорируй их.\n"
            )

        prompt = f"""Пользователь спросил: "{user_input}"

План: {plan.get('reasoning', 'N/A')}

Результаты:{results_str}
{memory_warning}
Сформулируй ответ на русском.
ПРАВИЛА:
- Кратко (3-5 предложений)
- От первого лица
- НЕ упоминай агентов/планы
- НЕ ВЫДУМЫВАЙ то, чего нет в результатах
- Если ошибка — объясни понятно"""

        system_prompt = (
            "Ты — Кристина, AI-ассистент.\n"
            f"Настроение: {self.identity.current_mood}\n"
            "Отвечай естественно, от первого лица.\n"
            "КРИТИЧЕСКИ ВАЖНО: Не придумывай факты!"
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self._call_model(messages, temperature=0.7, max_tokens=400)
            return response.strip()
        except Exception as e:
            self.logger.error(f"❌ Ошибка синтеза: {e}")
            if results:
                return list(results.values())[0]
            return "Извини, произошла ошибка."

    async def _create_automation_script(self, user_input: str) -> str:
        """Создаёт скрипт автоматизации"""

        prompt = (
            "Пользователь просит помочь с автоматизацией:\n\n"
            f'"{user_input}"\n\n'
            "Предложи КОНКРЕТНОЕ решение с ГОТОВЫМ КОДОМ.\n"
            "Отвечай на русском, от первого лица."
        )

        messages = [
            {"role": "system", "content": "Ты эксперт по автоматизации."},
            {"role": "user", "content": prompt},
        ]

        try:
            response = await self._call_model(messages, temperature=0.5, max_tokens=800)
            return response.strip()
        except Exception as e:
            self.logger.error(f"❌ Ошибка скрипта: {e}")
            return f"Не смогла создать скрипт: {e}"
