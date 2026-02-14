"""
Orchestrator — координатор Multi-Agent системы

v7.2 ЭВОЛЮЦИЯ К ПОНИМАНИЮ:
- IntentRouter (Tier 1+2) вместо LLM для роутинга
- ResponseGenerator вместо LLM для синтеза ответов
- LearnedPatterns — каждый LLM-вызов обучает Кристину
- NeuralEngine — Word2Vec + N-gram: Кристина строит СВОИ предложения
- BPE Tokenizer — подсловная токенизация (морфология русского языка)
- SentenceEmbeddings — понимание ФРАЗ, а не только слов
- ActiveLearning — умная неуверенность (лучше спросить, чем ошибиться)
- KnowledgeDistillation — сохранение ПРОЦЕССА рассуждений LLM
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
from core.bpe_tokenizer import BPETokenizer
from core.sentence_embeddings import SentenceEmbeddings
from core.active_learning import ActiveLearning
from core.knowledge_distillation import KnowledgeDistillation
from core.micro_transformer import MicroTransformer
from core.chain_of_thought import ChainOfThought
from core.self_play import SelfPlay
from core.cross_attention import MemoryAugmentedContext
from core.dialogue_memory import DialogueMemory
from core.task_planner import TaskPlanner
from core.conditional_gen import ConditionalGeneration
from core.mixture_of_experts import MixtureOfExperts
from core.code_understanding import CodeUnderstanding
from core.meta_learning import MetaLearner

from utils.logging import get_logger
import config

logger = get_logger("orchestrator")

class Orchestrator:
    """
    Оркестратор — управляет всей Multi-Agent системой

    v7.2: Четырёхуровневая архитектура:
      Tier 1: LearnedPatterns  — выученные у LLM паттерны (<10мс)
      Tier 2: RuleEngine       — regex правила (<5мс)
      Tier 3: KnowledgeDistillation — цепочки рассуждений (<50мс)
      Tier 4: LLM fallback     — director.analyze_request() (~25с)

    Новое в v7.2:
      + BPE Tokenizer — подсловная токенизация для русской морфологии
      + SentenceEmbeddings — понимание фраз целиком (не по словам)
      + ActiveLearning — Кристина спрашивает, когда не уверена
      + KnowledgeDistillation — запоминает КАК думает LLM, а не только ЧТО

    Каждый LLM-вызов обучает ВСЕ компоненты.
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
            sentence_embeddings=None,  # Подключим после инициализации SentenceEmbeddings
        )
        self.response_generator = ResponseGenerator(self.learned_patterns)
        self.dialogue_engine = DialogueEngine()

        # ── v7.2: Эволюционные компоненты ──
        self.bpe_tokenizer = BPETokenizer()
        self.sentence_embeddings = SentenceEmbeddings(
            self.dialogue_engine.neural
        )
        self.active_learning = ActiveLearning(
            neural_engine=self.dialogue_engine.neural,
            sentence_embeddings=self.sentence_embeddings,
        )

        # v7.4: Подключаем sentence embeddings к IntentRouter для Tier 2.5
        self.intent_router._sentence_embeddings = self.sentence_embeddings
        self.knowledge_distillation = KnowledgeDistillation(
            sentence_embeddings=self.sentence_embeddings,
        )

        # ── v7.2: MicroTransformer (Self-Attention) ──
        self.micro_transformer = MicroTransformer(
            vocab_size=max(self.bpe_tokenizer.get_vocab_size(), 8000),
        )

        # ── v7.3: Chain-of-Thought (рассуждения без LLM) ──
        self.chain_of_thought = ChainOfThought(
            knowledge_distillation=self.knowledge_distillation,
            sentence_embeddings=self.sentence_embeddings,
            tools=tools,
        )

        # ── v7.3: Self-Play (самооценка через LLM) ──
        # Инициализируется после директора (нужен director для LLM-вызовов)
        self._self_play_pending = True  # Ленивая инициализация

        # Агенты
        self.director = DirectorAgent(identity, tool_names=list(tools.keys()))
        self.executor = ExecutorAgent(tools)
        self.analyst = AnalystAgent(tools)
        self.reasoner = ReasonerAgent()

        # Self-Play (инициализируем после director)
        self.self_play = SelfPlay(
            director=self.director,
            learned_patterns=self.learned_patterns,
            neural_engine=self.dialogue_engine.neural if hasattr(self.dialogue_engine, 'neural') else None,
            knowledge_distillation=self.knowledge_distillation,
            chain_of_thought=self.chain_of_thought,
        )

        # ── v7.3: Cross-Attention с памятью (RAG внутри модели) ──
        self.memory_attention = MemoryAugmentedContext(
            vector_memory=vector_memory,
            sentence_embeddings=self.sentence_embeddings,
        )

        # ── v7.5: DialogueMemory (безлимитная память диалога) ──
        self.dialogue_memory = DialogueMemory(
            sentence_encoder=self.sentence_embeddings.encode,
            llm_summarizer=self._llm_summarize,
        )

        # ── v7.3: Task Planner (декомпозиция задач) ──
        self.task_planner = TaskPlanner(
            knowledge_distillation=self.knowledge_distillation,
            sentence_embeddings=self.sentence_embeddings,
        )

        # ── v7.3: Conditional Generation (условная генерация) ──
        self.conditional_gen = ConditionalGeneration(
            micro_transformer=self.micro_transformer,
            bpe_tokenizer=self.bpe_tokenizer,
        )

        # ── v7.3: Mixture of Experts (специализированные эксперты) ──
        self.moe = MixtureOfExperts()

        # ── v7.3: Code Understanding (понимание кода) ──
        self.code_understanding = CodeUnderstanding()

        # ── v7.3: Meta-Learning (обучение обучению) ──
        self.meta_learner = MetaLearner()

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
        neural_stats = dialogue_stats.get("neural", {})
        if neural_stats:
            logger.info(
                f"🧠 NeuralEngine: {neural_stats.get('vocabulary', 0)} слов, "
                f"{neural_stats.get('bigrams', 0)} биграмм, "
                f"{neural_stats.get('training_steps', 0)} обучений"
            )
        transformer_stats = self.micro_transformer.get_stats()
        logger.info(
            f"🤖 MicroTransformer: {transformer_stats['params']:,} params, "
            f"{transformer_stats['training_steps']} steps"
        )
        cot_stats = self.chain_of_thought.get_stats()
        logger.info(
            f"🧠 ChainOfThought: {cot_stats['total_reasonings']} рассуждений, "
            f"{cot_stats['success_rate']}% успех"
        )
        sp_stats = self.self_play.get_stats()
        logger.info(
            f"🎮 SelfPlay: {sp_stats['total_evaluations']} оценок, "
            f"avg={sp_stats['avg_score']}/10, "
            f"reinforce={sp_stats['reinforce_rate']}%"
        )
        dm_stats = self.dialogue_memory.get_stats()
        logger.info(
            f"💬 DialogueMemory: window={config.config.sliding_summary_window}, "
            f"max_summary={config.config.sliding_summary_max_tokens}tok"
        )
        ca_stats = self.memory_attention.get_stats()
        logger.info(
            f"🔗 CrossAttention: {ca_stats['total_enrichments']} обогащений, "
            f"gate={ca_stats['avg_gate']}"
        )
        tp_stats = self.task_planner.get_stats()
        logger.info(
            f"📋 TaskPlanner: {tp_stats['total_plans']} планов, "
            f"{tp_stats['total_tasks_completed']} задач"
        )
        cg_stats = self.conditional_gen.get_stats()
        logger.info(
            f"🎭 ConditionalGen: {cg_stats['total_generations']} генераций, "
            f"{cg_stats['condition_values']} условий"
        )
        moe_stats = self.moe.get_stats()
        logger.info(
            f"🧠 MoE: {moe_stats['num_experts']} experts, "
            f"{moe_stats['total_forwards']} forwards, "
            f"balance={moe_stats['balance_loss']:.4f}"
        )
        cu_stats = self.code_understanding.get_stats()
        logger.info(
            f"💻 CodeUnderstanding: {cu_stats['total_analyses']} analyses, "
            f"{cu_stats['indexed_snippets']} indexed"
        )
        ml_stats = self.meta_learner.get_stats()
        improving = sum(1 for c in ml_stats['components'].values() if c['trend'] == 'improving')
        plateau = sum(1 for c in ml_stats['components'].values() if c['trend'] == 'plateau')
        logger.info(
            f"🧬 MetaLearner: {ml_stats['total_meta_steps']} steps, "
            f"{improving}↑ {plateau}→, "
            f"quality={ml_stats['performance']['avg_quality']:.3f}"
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

            # === v7.3: Обогащение контекста памятью (Cross-Attention) ===
            try:
                enrichment = self.memory_attention.enrich(user_input)
                if enrichment and enrichment["gate"] > 0.3:
                    # Память релевантна — добавляем в контекст
                    mem_snippets = [
                        m["text"][:100] for m in enrichment["memories"][:3]
                        if m["weight"] > 0.1
                    ]
                    if mem_snippets:
                        context += "\n[Релевантная память]: " + "; ".join(mem_snippets)
                        logger.debug(
                            f"🔗 CrossAttn: gate={enrichment['gate']:.2f}, "
                            f"добавлено {len(mem_snippets)} воспоминаний"
                        )
            except Exception as e:
                logger.debug(f"CrossAttention enrichment skipped: {e}")

            # === ШАГ 2: ЧЕТЫРЁХУРОВНЕВЫЙ РОУТИНГ (v7.2) ===
            route = self.intent_router.route(user_input)

            # v7.2: Оценка уверенности (ActiveLearning)
            assessment = self.active_learning.assess_confidence(
                user_input, route_result=route,
            )

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

                # v7.2: ActiveLearning может изменить поведение
                if assessment["action"] == "clarify":
                    # Кристина не уверена — спрашивает уточнение
                    logger.info(f"❓ ActiveLearning: уточняю (conf={assessment['confidence']:.2f})")
                    final_response = assessment["clarification"]
                elif assessment["action"] == "uncertain":
                    logger.info(f"❓ ActiveLearning: не уверена (conf={assessment['confidence']:.2f})")
                    final_response = assessment["uncertainty_phrase"]
                else:
                    final_response = await self._process_with_plan(
                        plan, user_input, context, route,
                    )
                    # Добавляем оговорку если нужно
                    if assessment["action"] == "hedge":
                        final_response += f"\n\n{assessment['hedge_phrase']}"
            else:
                # ── Tier 3: Chain-of-Thought (рассуждения без LLM) ──
                cot_result = self.chain_of_thought.reason(
                    user_input, context=context,
                )

                if cot_result and cot_result.overall_confidence >= 0.6:
                    # CoT справился — отвечаем без LLM!
                    logger.info(
                        f"🧠 Tier 3 (CoT/{cot_result.strategy}): "
                        f"{len(cot_result.steps)} шагов, "
                        f"conf={cot_result.overall_confidence:.2f}, "
                        f"{cot_result.reasoning_time_ms:.0f}ms"
                    )
                    self.stats["tier3_hits"] += 1
                    final_response = cot_result.final_answer

                    # Сохраняем в память и возвращаем
                    plan = {
                        "intent": "cot_reasoning",
                        "primary_agent": "reasoner",
                        "supporting_agents": [],
                        "complexity": "simple",
                        "reasoning": f"Tier 3 (CoT/{cot_result.strategy})",
                    }
                    await self._save_to_memory(user_input, final_response, plan)

                    elapsed = (datetime.now() - start_time).total_seconds()
                    self.stats["successful_requests"] += 1
                    self.stats["total_time"] += elapsed
                    self.stats["avg_time"] = self.stats["total_time"] / self.stats["successful_requests"]
                    logger.info(f"✅ Запрос обработан за {elapsed:.2f}s (CoT, без LLM)")
                    return final_response

                # ── Tier 4: LLM fallback ──
                logger.info("🧠 Tier 4 (LLM): Директор анализирует запрос...")
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

            # v7.3: Conditional Generation (с учётом стиля/настроения)
            if self.micro_transformer._training_steps >= 50:
                try:
                    conditions = self.conditional_gen.detect_conditions(user_input, mood=mood)
                    cond_response = self.conditional_gen.generate(
                        prompt=user_input, conditions=conditions,
                    )
                    if cond_response and len(cond_response) >= 5:
                        logger.info(f"🎭 ConditionalGen: {conditions} → ответ без LLM")
                        return cond_response
                except Exception as e:
                    logger.debug(f"ConditionalGen failed: {e}")

                # Fallback: raw MicroTransformer (без условий)
                try:
                    prompt_ids = self.bpe_tokenizer.encode(user_input)
                    if prompt_ids and len(prompt_ids) >= 2:
                        generated_ids = self.micro_transformer.generate(
                            prompt_ids, max_len=40, temperature=0.8,
                            top_k=30, top_p=0.9,
                        )
                        new_ids = generated_ids[len(prompt_ids):]
                        if new_ids:
                            transformer_response = self.bpe_tokenizer.decode(new_ids).strip()
                            if len(transformer_response) >= 5:
                                logger.info("🤖 MicroTransformer: ответ без LLM")
                                return transformer_response
                except Exception as e:
                    logger.debug(f"MicroTransformer generation failed: {e}")

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
        # v7.3: TaskPlanner для декомпозиции сложных задач
        if plan.get("complexity") == "complex":
            try:
                task_plan = self.task_planner.plan(user_input)
                plan_text = self.task_planner.format_plan(task_plan)
                logger.info(f"📋 TaskPlanner: {task_plan.total_tasks} подзадач")
                # Добавляем план в контекст для LLM
                context += f"\n[План выполнения]:\n{plan_text}"
            except Exception as e:
                logger.debug(f"TaskPlanner skipped: {e}")

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
        # Извлекаем args из route (Tier 1/2) или пустой dict для fallback на _detect_tool_from_input
        task = {
            "tool": intent,
            "args": route.get("slots", {}) if route else {},
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

    async def _llm_summarize(self, prompt: str) -> str:
        """Суммаризация через LLM (для DialogueMemory)"""
        try:
            from ollama import AsyncClient
            client = AsyncClient(host=config.config.ollama_hosts.cpu)
            response = await client.generate(
                model=config.config.memory_summarizer_model,
                prompt=prompt,
                options={"temperature": 0.1, "num_predict": 300},
            )
            return response.get("response", "")
        except Exception as e:
            logger.debug(f"LLM summarize failed: {e}")
            return ""

    async def _build_context(self, user_input: str) -> str:
        """
        Строит контекст из всех источников памяти.

        v7.5: Использует DialogueMemory для безлимитного контекста сессии.

        Бюджет ~2000 токенов:
          - DialogueMemory (резюме + поиск + recent): ~1800 токенов
          - Долгосрочная память (ChromaDB): ~300 токенов
          - Code / MoE: ~200 токенов
        """

        # 1. DialogueMemory: резюме сессии + поиск + последние сообщения
        dialogue_context = await self.dialogue_memory.build_context(user_input)

        # 2. Релевантная эпизодическая память (short-term)
        relevant_memory = self.memory.get_relevant_context(user_input, max_items=3)

        # 3. Векторная долгосрочная память (async)
        vector_context = ""
        try:
            vector_results = await self.vector_memory.search_async(user_input, n_results=3)
            if vector_results:
                vector_parts = []
                for r in vector_results[:3]:
                    date = r['metadata'].get('date', '')
                    text = r['text'][:120]
                    vector_parts.append(f"  [{date}] {text}")
                vector_context = "\n[Долговременная память]:\n" + "\n".join(vector_parts)
        except Exception:
            pass

        # 4. Code Understanding: если пользователь прислал код
        code_context = ""
        try:
            import re as _re
            code_match = _re.search(r'```(?:python)?\s*\n(.+?)```', user_input, _re.DOTALL)
            if code_match:
                code_snippet = code_match.group(1)
                analysis = self.code_understanding.analyze_code(code_snippet)
                if analysis and analysis.summary:
                    code_context = f"\n[Анализ кода]: {analysis.summary}"
                    if analysis.patterns:
                        warnings = [p.message for p in analysis.patterns[:3]]
                        code_context += "\n  Замечания: " + "; ".join(warnings)
        except Exception:
            pass

        # 5. MoE routing: определяем доминирующего эксперта
        moe_context = ""
        try:
            input_emb = self.sentence_embeddings.encode(user_input)
            if input_emb:
                from core.mixture_of_experts import D_MODEL as MOE_D
                in_vec = (input_emb[:MOE_D] + [0.0] * MOE_D)[:MOE_D]
                expert_name = self.moe.get_expert_for_text(user_input, in_vec)
                moe_context = f"\nДоминирующий эксперт: {expert_name}"
        except Exception:
            pass

        context = f"""Контекст:
{dialogue_context}
{relevant_memory}
{vector_context}
{code_context}
{moe_context}"""

        return context

    async def _save_to_memory(self, user_input: str, response: str, plan: Dict):
        """Сохраняет диалог в память + обучает v7.2 компоненты"""

        try:
            # v7.5: Сохраняем в DialogueMemory (безлимитная память сессии)
            self.dialogue_memory.add('user', user_input)
            self.dialogue_memory.add('assistant', response)
            await self.dialogue_memory.maybe_compress()

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

            # ── v7.2: Обучение новых компонентов ──

            # BPE Tokenizer: учится на каждом тексте
            self.bpe_tokenizer.train_on_text(user_input)
            self.bpe_tokenizer.train_on_text(response)

            # SentenceEmbeddings: обновляет IDF статистику
            self.sentence_embeddings.learn_from_text(user_input)
            self.sentence_embeddings.learn_from_text(response)

            # MicroTransformer: дообучение (мета-управляемое)
            if self.meta_learner.should_train("micro_transformer"):
                try:
                    user_tokens = self.bpe_tokenizer.encode(user_input)
                    resp_tokens = self.bpe_tokenizer.encode(response)
                    if len(user_tokens) >= 3 and len(resp_tokens) >= 3:
                        combined = user_tokens + [4] + resp_tokens + [3]
                        loss = self.micro_transformer.train_step(combined)
                        if isinstance(loss, (int, float)):
                            self.meta_learner.report_loss("micro_transformer", loss)
                except Exception as e:
                    logger.debug(f"MicroTransformer training error: {e}")

            # ConditionalGen: обучаем с условиями (мета-управляемое)
            if self.meta_learner.should_train("conditional_gen"):
                try:
                    conditions = self.conditional_gen.detect_conditions(user_input)
                    self.conditional_gen.train(response, conditions)
                except Exception as e:
                    logger.debug(f"ConditionalGen training error: {e}")

            # MoE: обучаем экспертов (мета-управляемое)
            if self.meta_learner.should_train("moe"):
                try:
                    input_emb = self.sentence_embeddings.encode(user_input)
                    resp_emb = self.sentence_embeddings.encode(response[:200])
                    if input_emb and resp_emb:
                        from core.mixture_of_experts import D_MODEL as MOE_D
                        in_vec = (input_emb[:MOE_D] + [0.0] * MOE_D)[:MOE_D]
                        tgt_vec = (resp_emb[:MOE_D] + [0.0] * MOE_D)[:MOE_D]
                        loss = self.moe.train_step(in_vec, tgt_vec)
                        self.meta_learner.report_loss("moe", loss)
                except Exception as e:
                    logger.debug(f"MoE training error: {e}")

            # v7.4: Обучаем EmbeddingClassifier (Tier 2.5) на каждом роутинге
            intent = plan.get("intent", "unknown")
            primary_agent = plan.get("primary_agent", "director")
            if intent != "unknown" and intent != "error":
                self.intent_router.learn_from_route(user_input, intent, primary_agent)

            # KnowledgeDistillation: дистиллирует LLM-ответы
            intent = plan.get("intent", "unknown")
            reasoning = plan.get("reasoning", "")
            is_llm_response = reasoning.startswith("Tier 3") or \
                              reasoning.startswith("Tier 4") or \
                              "LLM" in reasoning
            if is_llm_response and intent != "unknown":
                self.knowledge_distillation.distill(
                    user_input=user_input,
                    llm_response=response,
                    intent=intent,
                    result_success=True,
                )

            # Self-Play: батчевая оценка ответов Tier 1-3 (без LLM)
            is_own_response = reasoning.startswith("Tier 1") or \
                              reasoning.startswith("Tier 2") or \
                              reasoning.startswith("Tier 3 (CoT")
            if is_own_response:
                tier = "tier1" if "Tier 1" in reasoning else \
                       "tier2" if "Tier 2" in reasoning else "tier3"
                self.self_play.add_to_batch(user_input, response, source_tier=tier)

                # Если буфер заполнился — запускаем батчевую оценку
                if self.self_play.batch_ready:
                    try:
                        await self.self_play.evaluate_batch()
                    except Exception as sp_err:
                        logger.debug(f"SelfPlay batch eval deferred: {sp_err}")

            # Meta-Learning: сообщаем о качестве и оптимизируем
            try:
                tier = "tier1" if "Tier 1" in reasoning else \
                       "tier2" if "Tier 2" in reasoning else \
                       "tier3" if "Tier 3" in reasoning else "tier4"
                # Оценка качества: длина ответа + наличие смысла
                quality = min(1.0, len(response) / 200) * 0.5 + 0.5
                components = ["micro_transformer", "moe", "conditional_gen"]
                if is_llm_response:
                    components.append("knowledge_distillation")
                self.meta_learner.report_response(quality, tier, components)
                self.meta_learner.optimize_step()
            except Exception as e:
                logger.debug(f"MetaLearner step error: {e}")

        except Exception as e:
            logger.error(f"Ошибка сохранения в память: {e}")

    async def close(self):
        """Корректно закрывает все соединения (предотвращает ResourceWarning)"""
        logger.info("🔌 Закрытие соединений агентов...")
        # Закрываем всех агентов
        for name, agent in self.agents.items():
            try:
                await agent.close()
            except Exception as e:
                logger.debug(f"Ошибка закрытия {name}: {e}")

        # Закрываем vector_memory async client
        if hasattr(self.vector_memory, 'close'):
            try:
                await self.vector_memory.close()
            except Exception as e:
                logger.debug(f"Ошибка закрытия vector_memory: {e}")

        # Закрываем SQLite-компоненты
        for component_name in ('micro_transformer', 'self_play', 'knowledge_distillation'):
            component = getattr(self, component_name, None)
            if component and hasattr(component, 'close'):
                try:
                    component.close()
                except Exception as e:
                    logger.debug(f"Ошибка закрытия {component_name}: {e}")

        logger.info("✅ Все соединения закрыты")

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

        dialogue_stats = self.dialogue_engine.get_stats()

        return {
            "orchestrator": self.stats,
            "agents": agent_stats,
            "vram": self.vram_manager.get_stats(),
            "learning": {
                "patterns": self.learned_patterns.get_stats(),
                "dialogue": dialogue_stats,
                "neural": dialogue_stats.get("neural", {}),
                "llm_free_percent": round(llm_free_pct, 1),
                "tier1_hits": self.stats["tier1_hits"],
                "tier2_hits": self.stats["tier2_hits"],
                "tier3_hits": self.stats["tier3_hits"],
            },
            # v7.2: Статистика эволюционных компонентов
            "evolution": {
                "bpe_tokenizer": self.bpe_tokenizer.get_stats(),
                "sentence_embeddings": self.sentence_embeddings.get_stats(),
                "active_learning": self.active_learning.get_stats(),
                "knowledge_distillation": self.knowledge_distillation.get_stats(),
                "micro_transformer": self.micro_transformer.get_stats(),
                "chain_of_thought": self.chain_of_thought.get_stats(),
                "self_play": self.self_play.get_stats(),
                "cross_attention": self.memory_attention.get_stats(),
                "dialogue_memory": self.dialogue_memory.get_stats(),
                "task_planner": self.task_planner.get_stats(),
                "conditional_gen": self.conditional_gen.get_stats(),
                "mixture_of_experts": self.moe.get_stats(),
                "code_understanding": self.code_understanding.get_stats(),
                "meta_learner": self.meta_learner.get_stats(),
            },
        }
