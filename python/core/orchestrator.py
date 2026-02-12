"""
Orchestrator — координатор Multi-Agent системы
"""

import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime

from core.agents.director import DirectorAgent
from core.agents.executor import ExecutorAgent
from core.agents.analyst import AnalystAgent
from core.agents.reasoner import ReasonerAgent
from core.vram_manager import VRAMManager

from utils.logging import get_logger
import config

logger = get_logger("orchestrator")

class Orchestrator:
    """
    Оркестратор — управляет всей Multi-Agent системой
    
    Функции:
    - Получает запрос пользователя
    - Делегирует директору для планирования
    - Управляет выполнением через агентов
    - Координирует VRAM менеджер
    - Синтезирует финальный ответ
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
        
        # Агенты
        self.director = DirectorAgent(identity)
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
            "avg_time": 0.0
        }

        logger.info(f"✅ Multi-Agent система готова (агентов: {len(self.agents)})")
        logger.info(f"📊 VRAM: {self.vram_manager.get_stats()['vram']}")
    
    async def process(self, user_input: str) -> str:
        """
        Главный цикл обработки запроса
        
        v6.0: Контекст памяти передаётся в execute() и synthesize_response()
        """
        
        start_time = datetime.now()
        self.stats["total_requests"] += 1
        
        logger.info(f"🎯 Обработка запроса: {user_input[:50]}...")
        
        try:
            # === ШАГ 1: СТРОИМ КОНТЕКСТ ===
            context = self._build_context(user_input)
            
            # === ШАГ 2: ДИРЕКТОР АНАЛИЗИРУЕТ ===
            logger.info("🧠 Директор анализирует запрос...")
            plan = await self.director.analyze_request(user_input, context)

            logger.info(f"📋 План: {plan['primary_agent']} + {plan['supporting_agents']}")

            # === FAST PATH: простые диалоговые запросы (1 LLM вызов вместо 3) ===
            if (plan["primary_agent"] == "director"
                    and plan.get("complexity") == "simple"
                    and not plan.get("supporting_agents")):
                logger.info("⚡ Fast path: простой запрос → director напрямую")
                final_response = await self.director.execute(
                    {"type": "general", "input": user_input, "context": context},
                )
            else:
                # === ШАГ 3: ЗАГРУЖАЕМ АГЕНТОВ ===
                required_agents = [plan["primary_agent"]] + plan.get("supporting_agents", [])
                await self.vram_manager.ensure_loaded(required_agents)

                # === ШАГ 4: ВЫПОЛНЕНИЕ (v6.0: передаём context!) ===
                results = await self._execute_plan(plan, user_input, context)

                # === ШАГ 5: СИНТЕЗ (v6.0: передаём context!) ===
                logger.info("🎨 Директор синтезирует ответ...")

                final_response = await self.director.synthesize_response(
                    user_input, plan, results, context=context
                )
            
            # === ШАГ 6: СОХРАНЕНИЕ ===
            self._save_to_memory(user_input, final_response, plan)

            elapsed = (datetime.now() - start_time).total_seconds()
            self.stats["successful_requests"] += 1
            self.stats["total_time"] += elapsed
            self.stats["avg_time"] = self.stats["total_time"] / self.stats["successful_requests"]

            # MetaCognition: записываем результат стратегии
            if hasattr(self, 'metacognition') and self.metacognition:
                # Маппинг имён агентов → стратегий MetaCognition
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
            # MetaCognition: записываем ошибку
            if hasattr(self, 'metacognition') and self.metacognition:
                self.metacognition.record_outcome(0.5, False, topic=user_input[:100])
            logger.error(f"❌ Ошибка обработки: {e}", exc_info=True)
            return f"Произошла ошибка при обработке запроса: {str(e)}"
    
    async def _execute_plan(self, plan: Dict, user_input: str, context: str = "") -> Dict[str, str]:
        """
        v6.0: Выполняет план, передавая контекст памяти агентам.
        """
        
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
        """
        v6.0: Строит задачу для агента, включая контекст памяти.
        """
        
        # Executor — передаём intent из плана директора как подсказку
        if agent_name == "executor":
            return {
                "tool": plan.get("intent"),
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
        
        # Director — v6.0: ПЕРЕДАЁМ КОНТЕКСТ!
        elif agent_name == "director":
            return {
                "type": "general",
                "input": user_input,
                "context": context,  # ← ЭТО КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ
            }
        
        return {
            "type": "general",
            "input": user_input,
            "context": context,
        }
    
    def _build_context(self, user_input: str) -> str:
        """
        v6.0: Строит контекст из всех источников памяти.
        
        Источники (по приоритету):
        1. Thread context (текущий разговор)
        2. Episodic memory + summaries (многоуровневый поиск)
        3. Knowledge Graph (факты о пользователе)
        4. Vector memory (семантический поиск)
        """
        
        # 1. Релевантная память (v6.0: ищет по ВСЕМ эпизодам + summaries + KG)
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
        
        # 3. Векторная память
        vector_results = self.vector_memory.search(user_input, n_results=2)
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
    
    def _save_to_memory(self, user_input: str, response: str, plan: Dict):
        """Сохраняет диалог в память"""
        
        try:
            # Рабочая память
            self.memory.add_to_working("user", user_input)
            self.memory.add_to_working("assistant", response)
            
            # Эпизодическая
            importance = 2 if plan.get("complexity") == "complex" else 1
            self.memory.add_episode(
                user_input,
                response,
                self.identity.current_mood,
                importance
            )
            
            # Векторная
            self.vector_memory.add_dialogue(
                user_input,
                response,
                importance=importance
            )
            
            # Thread
            self.thread_memory.update(user_input, response)
        
        except Exception as e:
            logger.error(f"Ошибка сохранения в память: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика оркестратора"""
        
        agent_stats = {
            name: agent.get_stats()
            for name, agent in self.agents.items()
        }
        
        return {
            "orchestrator": self.stats,
            "agents": agent_stats,
            "vram": self.vram_manager.get_stats()
        }