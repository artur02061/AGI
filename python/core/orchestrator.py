"""
Orchestrator — координатор Multi-Agent системы

v7.0 САМООБУЧЕНИЕ:
- IntentRouter (Tier 1+2) вместо LLM для роутинга
- ResponseGenerator вместо LLM для синтеза ответов
- LearnedPatterns — каждый LLM-вызов обучает Кристину
- LLM = учитель, вызывается только когда алгоритмы не справляются
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.agents.director import DirectorAgent
from core.agents.executor import ExecutorAgent
from core.agents.analyst import AnalystAgent
from core.agents.reasoner import ReasonerAgent
from core.vram_manager import VRAMManager
from core.learned_patterns import LearnedPatterns
from core.intent_router import IntentRouter
from core.response_generator import ResponseGenerator
from core.dialogue_engine import DialogueEngine

from utils.logging import get_logger
import config

logger = get_logger("orchestrator")

class Orchestrator:
    """
    Оркестратор — управляет всей Multi-Agent системой

    v7.0: Трёхуровневая архитектура самообучения:
      Tier 1: LearnedPatterns  — выученные у LLM паттерны (<10мс)
      Tier 2: RuleEngine       — regex правила (<5мс)
      Tier 3: LLM fallback     — director.analyze_request() (~25с)

    Каждый LLM-вызов (Tier 3) ОБУЧАЕТ Tier 1.
    Со временем Tier 3 вызывается всё реже.
    """

    def __init__(self, tools: Dict, memory, identity, vector_memory, thread_memory):
        logger.info("🧠 Инициализация Multi-Agent системы...")

        # Компоненты
        self.tools = tools
        self.memory = memory
        self.identity = identity
        self.vector_memory = vector_memory
        self.thread_memory = thread_memory

        # VRAM Manager
        self.vram_manager = VRAMManager()

        # ── v7.0: Самообучающийся мозг ──
        self.learned_patterns = LearnedPatterns()
        self.intent_router = IntentRouter(
            self.learned_patterns,
            tool_names=list(tools.keys()),
        )
        self.response_generator = ResponseGenerator(self.learned_patterns)
        self.dialogue_engine = DialogueEngine()

        # Агенты
        self.director = DirectorAgent(identity, tool_names=list(tools.keys()))
        self.executor = ExecutorAgent(tools)
        self.analyst = AnalystAgent(tools)
        self.reasoner = ReasonerAgent()

        self.agents = {
            "director": self.director,
            "executor": self.executor,
            "analyst": self.analyst,
            "reasoner": self.reasoner
        }

        # Отмечаем hot-loaded агентов как загруженных
        for agent_name in config.config.hot_loaded_agents:
            if agent_name in self.agents:
                self.agents[agent_name].is_loaded = True

        # Consciousness-модули (устанавливаются из main.py)
        self.vad_emotions = None
        self.self_awareness = None
        self.metacognition = None

        # Статистика
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "total_time": 0.0,
            "avg_time": 0.0,
            # v7.0: статистика самообучения
            "tier1_hits": 0,   # обработано выученными паттернами
            "tier2_hits": 0,   # обработано правилами
            "tier3_hits": 0,   # обработано LLM (и записано для обучения)
        }

        patterns_stats = self.learned_patterns.get_stats()
        dialogue_stats = self.dialogue_engine.get_stats()
        logger.info(f"✅ Multi-Agent система готова (агентов: {len(self.agents)})")
        logger.info(
            f"🧠 LearnedPatterns: {patterns_stats['routing']} routing, "
            f"{patterns_stats['response']} response, {patterns_stats['slots']} slots"
        )
        logger.info(
            f"💬 DialogueEngine: {dialogue_stats['phrases']} фраз "
            f"({dialogue_stats['phrases_from_llm']} от LLM), "
            f"{dialogue_stats['dialogues']} диалогов"
        )
        logger.info(f"📊 VRAM: {self.vram_manager.get_stats()['vram']}")

    async def process(self, user_input: str) -> str:
        """
        Главный цикл обработки запроса.

        v7.0: Трёхуровневый роутинг с самообучением.
        Порядок: LearnedPatterns → Rules → LLM (fallback + обучение)
        """

        start_time = datetime.now()
        self.stats["total_requests"] += 1

        logger.info(f"🎯 Обработка запроса: {user_input[:50]}...")

        try:
            # === ШАГ 1: СТРОИМ КОНТЕКСТ ===
            context = await self._build_context(user_input)

            # === ШАГ 2: ТРЁХУРОВНЕВЫЙ РОУТИНГ (v7.0) ===
            route = self.intent_router.route(user_input)

            if route:
                # ── Tier 1 или Tier 2 сработал: БЕЗ LLM ──
                tier = "Tier 1 (learned)" if route["source"] == "learned" else "Tier 2 (rule)"
                logger.info(f"⚡ {tier}: {route['intent']} → {route['agent']}")

                if route["source"] == "learned":
                    self.stats["tier1_hits"] += 1
                else:
                    self.stats["tier2_hits"] += 1

                plan = {
                    "intent": route["intent"],
                    "primary_agent": route["agent"],
                    "supporting_agents": [],
                    "complexity": "simple",
                    "reasoning": f"{tier} routing",
                }

                final_response = await self._process_with_plan(
                    plan, user_input, context, route,
                )
            else:
                # ── Tier 3: LLM fallback ──
                logger.info("🧠 Tier 3 (LLM): Директор анализирует запрос...")
                self.stats["tier3_hits"] += 1

                plan = await self.director.analyze_request(user_input, context)
                logger.info(f"📋 План: {plan['primary_agent']} + {plan['supporting_agents']}")

                # ОБУЧЕНИЕ: записываем LLM-решение в LearnedPatterns
                intent = plan.get("intent", "unknown")
                if intent != "unknown" and intent != "error":
                    self.learned_patterns.learn_routing(
                        user_input=user_input,
                        intent=intent,
                        agent=plan["primary_agent"],
                        source="llm",
                    )
                    logger.info(f"📝 Learned: '{user_input[:40]}' → {intent}")

                final_response = await self._process_with_plan(
                    plan, user_input, context, route=None,
                )

            # === СОХРАНЕНИЕ В ПАМЯТЬ ===
            await self._save_to_memory(user_input, final_response, plan)

            elapsed = (datetime.now() - start_time).total_seconds()
            self.stats["successful_requests"] += 1
            self.stats["total_time"] += elapsed
            self.stats["avg_time"] = self.stats["total_time"] / self.stats["successful_requests"]

            # MetaCognition: записываем результат стратегии
            if hasattr(self, 'metacognition') and self.metacognition:
                _agent_to_strategy = {
                    "director": "direct",
                    "executor": "tool_use",
                    "analyst": "web_search",
                    "reasoner": "delegate",
                }
                strategy = _agent_to_strategy.get(
                    plan.get("primary_agent", "director"), "direct"
                )
                self.metacognition.record_strategy_outcome(strategy, 1.0)
                confidence = self.metacognition.estimate_confidence(topic=user_input[:100])
                self.metacognition.record_outcome(confidence, True, topic=user_input[:100])

            logger.info(f"✅ Запрос обработан за {elapsed:.2f}s")

            return final_response

        except Exception as e:
            self.stats["failed_requests"] += 1
            if hasattr(self, 'metacognition') and self.metacognition:
                self.metacognition.record_outcome(0.5, False, topic=user_input[:100])
            logger.error(f"❌ Ошибка обработки: {e}", exc_info=True)
            return f"Произошла ошибка при обработке запроса: {str(e)}"

    async def _process_with_plan(
        self,
        plan: Dict,
        user_input: str,
        context: str,
        route: Optional[Dict],
    ) -> str:
        """
        Обрабатывает запрос по готовому плану.

        v7.0: Пытается ответить без LLM (ResponseGenerator).
        Если LLM всё же нужен — ОБУЧАЕТ ResponseGenerator.
        """
        primary_agent = plan["primary_agent"]
        intent = plan.get("intent", "unknown")

        # === FAST PATH: director диалог (simple) ===
        if (primary_agent == "director"
                and plan.get("complexity") == "simple"
                and not plan.get("supporting_agents")):

            # v7.0: Пробуем DialogueEngine (без LLM)
            mood = self.identity.current_mood
            energy = self.identity.energy_level
            dialogue_response = self.dialogue_engine.generate_response(
                user_input, mood=mood, energy=energy,
            )

            if dialogue_response:
                logger.info("⚡ DialogueEngine: ответ без LLM")
                return dialogue_response

            # Fallback: LLM
            logger.info("🧠 Director (LLM): диалоговый ответ")
            llm_response = await self.director.execute(
                {"type": "general", "input": user_input, "context": context},
            )

            # ОБУЧЕНИЕ: записываем LLM-ответ для DialogueEngine
            self.dialogue_engine.learn_from_dialogue(
                user_input=user_input,
                response=llm_response,
                mood=mood,
                source="llm",
            )
            logger.info("📝 DialogueEngine: learned from LLM response")

            return llm_response

        # === EXECUTOR PATH: инструментальные задачи ===
        if primary_agent == "executor":
            return await self._executor_path(plan, user_input, context, route)

        # === FULL PATH: сложные задачи с несколькими агентами ===
        required_agents = [primary_agent] + plan.get("supporting_agents", [])
        await self.vram_manager.ensure_loaded(required_agents)

        results = await self._execute_plan(plan, user_input, context)

        logger.info("🎨 Директор синтезирует ответ...")
        final_response = await self.director.synthesize_response(
            user_input, plan, results, context=context
        )

        return final_response

    async def _executor_path(
        self,
        plan: Dict,
        user_input: str,
        context: str,
        route: Optional[Dict],
    ) -> str:
        """
        Быстрый путь для инструментальных задач.

        v7.0: Пытается выполнить и ответить БЕЗ LLM.
        """
        intent = plan.get("intent")

        # Строим задачу для executor
        task = {
            "tool": intent,
            "args": route.get("slots", {}) if route else [],
            "user_input": user_input,
        }

        # Валидация intent
        if intent and intent not in self.tools:
            logger.warning(f"⚠️ Несуществующий инструмент '{intent}', fallback на NLU")
            task["tool"] = None

        # Выполняем инструмент
        try:
            tool_result = await self.executor.execute(task)
        except Exception as e:
            logger.error(f"❌ Ошибка executor: {e}")
            tool_result = f"ERROR: {str(e)}"

        logger.info(f"✅ executor: {tool_result[:100]}")

        # === ОТВЕТ БЕЗ LLM: ResponseGenerator ===
        response = self.response_generator.generate(intent, tool_result)

        if response:
            logger.info("⚡ Ответ сгенерирован без LLM (ResponseGenerator)")
            # ОБУЧЕНИЕ: записываем slot-паттерны если route сработал
            if route and route.get("slots") and intent:
                self.learned_patterns.learn_slots(
                    intent, user_input, route["slots"]
                )
            return response

        # === FALLBACK: LLM синтезирует ответ ===
        logger.info("🎨 LLM синтезирует ответ (ResponseGenerator не справился)...")
        results = {"executor": tool_result}
        final_response = await self.director.synthesize_response(
            user_input, plan, results, context=context
        )

        # ОБУЧЕНИЕ: записываем ответ LLM как шаблон
        if intent and not tool_result.startswith("ERROR"):
            self.learned_patterns.learn_response(
                intent=intent,
                tool_result=tool_result,
                final_response=final_response,
            )
            logger.info(f"📝 Learned response: {intent}")

        return final_response

    async def _execute_plan(self, plan: Dict, user_input: str, context: str = "") -> Dict[str, str]:
        """Выполняет план, передавая контекст памяти агентам."""

        primary_agent = plan["primary_agent"]
        supporting_agents = plan.get("supporting_agents", [])

        results = {}

        # === ОСНОВНОЙ АГЕНТ ===
        logger.info(f"⚡ Основной агент: {primary_agent}")

        primary_task = self._build_task(primary_agent, plan, user_input, context)

        try:
            primary_result = await self.agents[primary_agent].execute(primary_task)
            results[primary_agent] = primary_result
            logger.info(f"✅ {primary_agent}: {primary_result[:100]}")
        except Exception as e:
            logger.error(f"❌ Ошибка {primary_agent}: {e}")
            results[primary_agent] = f"ERROR: {str(e)}"

        # === ВСПОМОГАТЕЛЬНЫЕ АГЕНТЫ (параллельно) ===
        if supporting_agents:
            logger.info(f"🔄 Вспомогательные: {supporting_agents}")

            tasks = []
            valid_agents = []
            for agent_name in supporting_agents:
                if agent_name in self.agents:
                    task = self._build_task(agent_name, plan, user_input, context)
                    tasks.append(self._execute_agent(agent_name, task))
                    valid_agents.append(agent_name)

            if tasks:
                supporting_results = await asyncio.gather(*tasks, return_exceptions=True)

                for i, agent_name in enumerate(valid_agents):
                    result = supporting_results[i]
                    if isinstance(result, Exception):
                        results[agent_name] = f"ERROR: {str(result)}"
                    else:
                        results[agent_name] = result

        return results

    async def _execute_agent(self, agent_name: str, task: Dict) -> str:
        """Выполняет задачу через агента"""

        agent = self.agents.get(agent_name)

        if not agent:
            return f"ERROR: Агент {agent_name} не найден"

        try:
            result = await agent.execute(task)
            return result

        except Exception as e:
            logger.error(f"Ошибка выполнения {agent_name}: {e}")
            return f"ERROR: {str(e)}"

    def _build_task(self, agent_name: str, plan: Dict, user_input: str, context: str = "") -> Dict[str, Any]:
        """Строит задачу для агента, включая контекст памяти."""

        # Executor
        if agent_name == "executor":
            intent = plan.get("intent")
            if intent and intent not in self.tools:
                logger.warning(
                    f"⚠️ Директор предложил несуществующий инструмент '{intent}', "
                    f"executor определит инструмент из текста запроса"
                )
                intent = None
            return {
                "tool": intent,
                "args": [],
                "user_input": user_input,
            }

        # Analyst
        elif agent_name == "analyst":
            task_type = "web_search"
            if "анализ" in user_input.lower():
                task_type = "data_analysis"

            return {
                "type": task_type,
                "query": user_input,
                "max_results": 3,
            }

        # Reasoner
        elif agent_name == "reasoner":
            task_type = "general"
            if any(w in user_input.lower() for w in ["реши", "вычисли", "посчитай"]):
                task_type = "math"
            elif "логика" in user_input.lower() or "докажи" in user_input.lower():
                task_type = "logic"

            return {
                "type": task_type,
                "problem": user_input,
            }

        # Director
        elif agent_name == "director":
            return {
                "type": "general",
                "input": user_input,
                "context": context,
            }

        return {
            "type": "general",
            "input": user_input,
            "context": context,
        }

    async def _build_context(self, user_input: str) -> str:
        """Строит контекст из всех источников памяти."""

        # 1. Релевантная память
        relevant_memory = self.memory.get_relevant_context(user_input, max_items=3)

        # 2. Thread контекст (последние 3 сообщения)
        thread_context = ""
        if self.thread_memory.current_thread:
            thread = self.thread_memory.current_thread
            messages = thread.get('messages', [])[-3:]

            if messages:
                thread_context = f"\nТекущая тема: {thread['topic']}\n"
                thread_context += "Последние сообщения:\n"

                for msg in messages:
                    thread_context += f"  Пользователь: {msg['user'][:80]}\n"
                    thread_context += f"  Кристина: {msg['assistant'][:80]}\n"

        # 3. Векторная память (async)
        vector_results = await self.vector_memory.search_async(user_input, n_results=2)
        vector_context = ""

        if vector_results:
            vector_context = "\nИз долговременной памяти:\n"
            for r in vector_results[:2]:
                date = r['metadata'].get('date', '')
                text = r['text'][:100]
                vector_context += f"  [{date}] {text}...\n"

        context = f"""Контекст:
{relevant_memory}
{thread_context}
{vector_context}"""

        return context

    async def _save_to_memory(self, user_input: str, response: str, plan: Dict):
        """Сохраняет диалог в память"""

        try:
            self.memory.add_to_working("user", user_input)
            self.memory.add_to_working("assistant", response)

            importance = 2 if plan.get("complexity") == "complex" else 1
            self.memory.add_episode(
                user_input,
                response,
                self.identity.current_mood,
                importance
            )

            await self.vector_memory.add_dialogue_async(
                user_input,
                response,
                importance=importance
            )

            self.thread_memory.update(user_input, response)

        except Exception as e:
            logger.error(f"Ошибка сохранения в память: {e}")

    def get_stats(self) -> Dict[str, Any]:
        """Статистика оркестратора"""

        agent_stats = {
            name: agent.get_stats()
            for name, agent in self.agents.items()
        }

        total_routed = (
            self.stats["tier1_hits"]
            + self.stats["tier2_hits"]
            + self.stats["tier3_hits"]
        )
        llm_free_pct = 0.0
        if total_routed > 0:
            llm_free_pct = (
                (self.stats["tier1_hits"] + self.stats["tier2_hits"])
                / total_routed * 100
            )

        return {
            "orchestrator": self.stats,
            "agents": agent_stats,
            "vram": self.vram_manager.get_stats(),
            "learning": {
                "patterns": self.learned_patterns.get_stats(),
                "dialogue": self.dialogue_engine.get_stats(),
                "llm_free_percent": round(llm_free_pct, 1),
                "tier1_hits": self.stats["tier1_hits"],
                "tier2_hits": self.stats["tier2_hits"],
                "tier3_hits": self.stats["tier3_hits"],
            },
        }
