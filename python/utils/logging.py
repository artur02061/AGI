"""
Система логирования
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import config

class ColoredFormatter(logging.Formatter):
    """Цветной formatter для консоли"""
    
    COLORS = {
        'DEBUG': '\033[36m',      # Cyan
        'INFO': '\033[32m',       # Green
        'WARNING': '\033[33m',    # Yellow
        'ERROR': '\033[31m',      # Red
        'CRITICAL': '\033[35m',   # Magenta
        'RESET': '\033[0m'
    }
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Форматируем имя логгера
        name = record.name.split('.')[-1]  # Только последняя часть
        
        # Укорачиваем длинные сообщения
        message = record.getMessage()
        if len(message) > 200:
            message = message[:197] + "..."
        
        formatted = f"{color}[{record.levelname:8}]{reset} {name:12} | {message}"
        
        if record.exc_info:
            formatted += "\n" + self.formatException(record.exc_info)
        
        return formatted

def setup_logging(
    name: str = "kristina",
    level: Optional[str] = None,
    log_to_file: bool = True
) -> logging.Logger:
    """
    Настраивает логирование
    
    Args:
        name: Имя логгера
        level: Уровень (DEBUG, INFO, WARNING, ERROR)
        log_to_file: Писать ли в файл
    
    Returns:
        Настроенный logger
    """
    
    logger = logging.getLogger(name)
    
    # Уровень
    level = level or config.LOG_LEVEL
    logger.setLevel(getattr(logging, level))
    
    # Убираем старые handlers
    logger.handlers.clear()
    
    # Console handler (цветной)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(ColoredFormatter())
    logger.addHandler(console_handler)
    
    # File handler
    if log_to_file:
        file_handler = logging.FileHandler(
            config.LOG_FILE,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(config.LOG_FORMAT)
        )
        logger.addHandler(file_handler)
    
    return logger

# Главный logger
logger = setup_logging()

def get_logger(name: str) -> logging.Logger:
    """Получить child logger"""
    return logging.getLogger(f"kristina.{name}")