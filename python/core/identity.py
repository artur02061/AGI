"""
Движок личности (объединяет identity_engine и personality_evolution)
"""

import yaml
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from utils.logging import get_logger
import config

logger = get_logger("identity")

class IdentityEngine:
    """Управление личностью и эволюцией"""
    
    def __init__(self):
        self.identity_dir = config.BASE_DIR / "core" / "identity_data"
        
        # Загружаем личность
        self.core_identity = self._load_yaml('kristina_core.yaml')
        self.memories = self._load_json('kristina_memories.json')
        self.beliefs = self._load_yaml('kristina_beliefs.yaml')
        
        # Эволюция
        self.evolution_file = self.identity_dir / 'kristina_evolution.json'
        self.evolution_data = self._load_json('kristina_evolution.json')
        
        # Текущее состояние
        self.current_mood = config.DEFAULT_MOOD
        self.energy_level = config.INITIAL_ENERGY
        self.conversation_depth = 0
        
        # Счётчики паттернов (для эволюции)
        self.interaction_patterns = {
            'user_prefers_short': 0,
            'user_prefers_detailed': 0,
            'user_technical': 0
        }
        
        logger.info("🧠 Личность загружена")
        self._print_identity_summary()
    
    def _load_yaml(self, filename: str) -> Dict:
        """Загрузка YAML"""
        path = self.identity_dir / filename
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        return {}
    
    def _load_json(self, filename: str) -> Dict:
        """Загрузка JSON"""
        path = self.identity_dir / filename
        if path.exists():
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _print_identity_summary(self):
        """Краткая информация"""
        core = self.core_identity.get('core_identity', {})
        meta = self.core_identity.get('meta', {})
        
        print(f"\n{'='*60}")
        print(f"  👤 {core.get('full_name', 'Кристина')}")
        print(f"  📅 Версия: {meta.get('version', '4.0')}")
        print(f"  🎂 Создана: {meta.get('created', 'N/A')}")
        print(f"{'='*60}\n")
    
    def increment_conversation_depth(self):
        """Увеличивает счётчик глубины"""
        self.conversation_depth += 1
        
        # Падение энергии
        if self.conversation_depth % 10 == 0:
            self.energy_level = max(20, self.energy_level - config.ENERGY_DECAY_PER_MESSAGE)
            logger.debug(f"Энергия снижена до {self.energy_level}%")
    
    def update_mood(self, emotion: str):
        """Обновляет настроение"""
        valid_moods = ['happy', 'satisfied', 'neutral', 'curious', 'frustrated', 'tired']
        
        if emotion in valid_moods:
            old_mood = self.current_mood
            self.current_mood = emotion
            
            logger.debug(f"Настроение: {old_mood} → {emotion}")
    
    def analyze_interaction(self, user_input: str, response: str):
        """Анализирует взаимодействие для эволюции"""
        
        # Длина предпочтений
        if len(user_input.split()) < 10:
            self.interaction_patterns['user_prefers_short'] += 1
        else:
            self.interaction_patterns['user_prefers_detailed'] += 1
        
        # Технический уровень
        tech_words = ['api', 'функция', 'код', 'процесс', 'алгоритм', 'класс']
        if any(w in user_input.lower() for w in tech_words):
            self.interaction_patterns['user_technical'] += 1
        
        # Проверяем адаптации
        self._check_evolution()
    
    def _check_evolution(self):
        """Проверяет необходимость эволюции"""
        
        threshold = 15
        
        # Краткость
        if self.interaction_patterns['user_prefers_short'] > threshold:
            if 'prefers_brevity' not in self.evolution_data.get('adaptations', {}):
                self._apply_adaptation(
                    'prefers_brevity',
                    'Пользователь предпочитает краткие ответы'
                )
        
        # Технический стиль
        if self.interaction_patterns['user_technical'] > threshold:
            if 'technical_user' not in self.evolution_data.get('adaptations', {}):
                self._apply_adaptation(
                    'technical_user',
                    'Пользователь технически подкован'
                )
    
    def _apply_adaptation(self, adaptation_id: str, reason: str):
        """Применяет адаптацию"""
        
        logger.info(f"🔄 Эволюция: {reason}")
        
        if 'changes' not in self.evolution_data:
            self.evolution_data['changes'] = []
        if 'adaptations' not in self.evolution_data:
            self.evolution_data['adaptations'] = {}
        
        self.evolution_data['changes'].append({
            'timestamp': datetime.now().isoformat(),
            'adaptation_id': adaptation_id,
            'reason': reason
        })
        
        self.evolution_data['adaptations'][adaptation_id] = {
            'applied': datetime.now().isoformat()
        }
        
        self._save_evolution()
    
    def _save_evolution(self):
        """Сохраняет эволюцию"""
        try:
            with open(self.evolution_file, 'w', encoding='utf-8') as f:
                json.dump(self.evolution_data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Ошибка сохранения эволюции: {e}")
    
    def get_adaptations(self) -> Dict:
        """Возвращает адаптации"""
        return self.evolution_data.get('adaptations', {})
    
    def reset_session(self):
        """Сброс сессии"""
        self.conversation_depth = 0
        self.energy_level = config.INITIAL_ENERGY
        self.current_mood = config.DEFAULT_MOOD
        logger.info("Сессия сброшена")