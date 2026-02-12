"""
Контроллер системы с улучшенной безопасностью
"""

import subprocess
import platform
import psutil
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

from utils.logging import get_logger
from utils.validators import validate_file_path, validate_process_name
from modules.system_control.app_finder import AppFinder
import config

IS_WINDOWS = platform.system() == "Windows"

logger = get_logger("system_controller")

class SystemController:
    """Управление системой"""
    
    def __init__(self):
        # Инициализируем поисковик приложений
        logger.info("Инициализация System Controller...")
        self.app_finder = AppFinder()
        
        # Логирование операций
        self.operation_log = []
        
        logger.info("System Controller готов")
    
    # ═══════════════════════════════════════════════════════════
    #                    ЗАПУСК ПРИЛОЖЕНИЙ
    # ═══════════════════════════════════════════════════════════
    
    async def launch_app(self, app_name: str) -> Dict[str, Any]:
        """
        Запускает приложение
        
        Args:
            app_name: Название приложения
        
        Returns:
            {"success": bool, "message": str, ...}
        """
        
        logger.info(f"Попытка запуска: {app_name}")
        
        # Ищем приложение
        app = self.app_finder.find_app(app_name)
        
        if not app:
            logger.warning(f"Приложение не найдено: {app_name}")
            return {
                "success": False,
                "message": f"Приложение '{app_name}' не найдено. Попробуй: search_apps(\"{app_name}\")"
            }
        
        # Проверяем, не запущено ли уже
        app_name_clean = Path(app['path']).stem
        process_name = app_name_clean + '.exe' if IS_WINDOWS else app_name_clean
        if self._is_running(process_name):
            logger.info(f"Приложение уже запущено: {app['name']}")
            return {
                "success": True,
                "message": f"✅ {app['name']} уже запущен",
                "already_running": True
            }
        
        # Запускаем
        try:
            subprocess.Popen(app['path'], shell=False)

            logger.info(f"✅ Запущено: {app['name']}")
            self._log_operation("launch_app", app['path'], True, app['name'])

            return {
                "success": True,
                "message": f"✅ {app['name']} запущен",
                "path": app['path']
            }

        except Exception as e:
            logger.error(f"Ошибка запуска {app['name']}: {e}")
            self._log_operation("launch_app", app.get('path', ''), False, str(e))
            return {
                "success": False,
                "message": f"Ошибка запуска: {str(e)}"
            }
    
    async def search_apps(self, query: str) -> Dict[str, Any]:
        """
        Ищет приложения по запросу
        
        Args:
            query: Поисковый запрос
        
        Returns:
            {"success": bool, "message": str, "apps": List}
        """
        
        logger.info(f"Поиск приложений: {query}")
        
        query_lower = query.lower()
        
        found = []
        for key, app in self.app_finder.app_cache.items():
            if query_lower in key or query_lower in app['name'].lower():
                found.append(app)
                if len(found) >= 10:
                    break
        
        if not found:
            logger.warning(f"Приложения не найдены: {query}")
            return {
                "success": False,
                "message": f"Не нашла приложений по запросу '{query}'"
            }
        
        # Форматируем результат
        result = f"🔍 Найдено приложений: {len(found)}\n\n"
        
        for i, app in enumerate(found, 1):
            name = app['name']
            path = app['path'][:60] + "..." if len(app['path']) > 60 else app['path']
            result += f"{i}. {name}\n   📁 {path}\n"
        
        logger.info(f"✅ Найдено {len(found)} приложений")
        
        return {
            "success": True,
            "message": result,
            "apps": found
        }
    
    def _is_running(self, process_name: str) -> bool:
        """Проверяет, запущен ли процесс"""
        
        if not process_name:
            return False
        
        for proc in psutil.process_iter(['name']):
            try:
                if proc.info['name'].lower() == process_name.lower():
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return False
    
    # ═══════════════════════════════════════════════════════════
    #                    УПРАВЛЕНИЕ ПРОЦЕССАМИ
    # ═══════════════════════════════════════════════════════════
    
    async def list_processes(
        self,
        filter_keyword: Optional[str] = None,
        limit: int = 20
    ) -> List[Dict]:
        """
        Список процессов
        
        Args:
            filter_keyword: Фильтр по имени
            limit: Максимум процессов
        
        Returns:
            Список словарей с информацией о процессах
        """
        
        logger.debug(f"Получение процессов (фильтр: {filter_keyword}, лимит: {limit})")
        
        processes = []
        
        for proc in psutil.process_iter(['pid', 'name', 'cpu_percent', 'memory_percent', 'status']):
            try:
                info = proc.info
                
                # Фильтр
                if filter_keyword and filter_keyword.lower() not in info['name'].lower():
                    continue
                
                processes.append({
                    "pid": info['pid'],
                    "name": info['name'],
                    "cpu": info['cpu_percent'] or 0,
                    "memory": info['memory_percent'] or 0,
                    "status": info['status']
                })
            
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        # Сортируем по CPU
        processes.sort(key=lambda x: x['cpu'], reverse=True)
        
        logger.debug(f"Найдено {len(processes)} процессов")
        
        return processes[:limit]
    
    async def kill_process(self, process_name: str) -> Dict[str, Any]:
        """
        Завершает процесс
        
        Args:
            process_name: Имя процесса (например, chrome.exe)
        
        Returns:
            {"success": bool, "message": str}
        """
        
        logger.warning(f"Попытка завершения процесса: {process_name}")
        
        # Валидация
        is_valid, reason = validate_process_name(process_name)
        if not is_valid:
            logger.warning(f"Завершение запрещено: {reason}")
            return {
                "success": False,
                "message": f"❌ {reason}"
            }
        
        # Ищем процесс
        killed = []
        
        for proc in psutil.process_iter(['name', 'pid']):
            try:
                if proc.info['name'].lower() == process_name.lower():
                    proc.terminate()
                    killed.append(proc.info['pid'])
            
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        if not killed:
            logger.warning(f"Процесс не найден: {process_name}")
            return {
                "success": False,
                "message": f"Процесс '{process_name}' не найден"
            }
        
        logger.info(f"✅ Завершено процессов: {len(killed)}")
        
        return {
            "success": True,
            "message": f"✅ Завершено процессов '{process_name}': {len(killed)}",
            "killed_pids": killed
        }
    
    # ═══════════════════════════════════════════════════════════
    #                    МОНИТОРИНГ СИСТЕМЫ
    # ═══════════════════════════════════════════════════════════
    
    async def get_system_status(self) -> Dict[str, Any]:
        """
        Получает статус системы
        
        Returns:
            Словарь с информацией о CPU, RAM, GPU, дисках
        """
        
        logger.debug("Получение статуса системы")
        
        # CPU
        cpu_percent = psutil.cpu_percent(interval=0.5)
        
        # RAM
        memory = psutil.virtual_memory()
        
        # Диск
        disk = psutil.disk_usage('C:/' if IS_WINDOWS else '/')
        
        status = {
            "cpu": {
                "usage_percent": cpu_percent,
                "cores": psutil.cpu_count(logical=False),
                "threads": psutil.cpu_count(logical=True)
            },
            "ram": {
                "usage_percent": memory.percent,
                "used_gb": memory.used / (1024**3),
                "total_gb": memory.total / (1024**3),
                "available_gb": memory.available / (1024**3)
            },
            "disk": {
                "usage_percent": disk.percent,
                "used_gb": disk.used / (1024**3),
                "free_gb": disk.free / (1024**3),
                "total_gb": disk.total / (1024**3)
            }
        }
        
        # GPU (если доступен)
        try:
            import pynvml
            
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            gpu_name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(gpu_name, bytes):
                gpu_name = gpu_name.decode('utf-8')
            
            gpu_util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            gpu_temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            status["gpu"] = {
                "name": gpu_name,
                "usage_percent": gpu_util.gpu,
                "memory_used_mb": gpu_mem.used // (1024 * 1024),
                "memory_total_mb": gpu_mem.total // (1024 * 1024),
                "memory_percent": (gpu_mem.used / gpu_mem.total) * 100,
                "temperature_c": gpu_temp
            }
            
            pynvml.nvmlShutdown()
        
        except Exception as e:
            logger.debug(f"GPU недоступен: {e}")
            status["gpu"] = {"error": "Недоступен"}
        
        return status
    
    # ═══════════════════════════════════════════════════════════
    #                    ФАЙЛОВЫЕ ОПЕРАЦИИ
    # ═══════════════════════════════════════════════════════════
    
    async def search_file(self, filename: str, search_paths: List[str] = None) -> Dict[str, Any]:
        """
        Ищет файл на всех дисках
        
        Args:
            filename: Имя файла для поиска
            search_paths: Пути для поиска (опционально)
        
        Returns:
            {"success": bool, "message": str, "files": List}
        """
        
        logger.info(f"Поиск файла: {filename}")
        
        if not search_paths:
            # Популярные места
            search_paths = [
                str(Path.home() / "Desktop"),
                str(Path.home() / "Documents"),
                str(Path.home() / "Downloads"),
            ]
            
            # Все доступные диски / корневые каталоги
            if IS_WINDOWS:
                for drive in "CDEFGH":
                    drive_path = f"{drive}:/"
                    if Path(drive_path).exists():
                        search_paths.append(drive_path)
            else:
                search_paths.append(str(Path.home()))
        
        found_files = []
        
        for search_path in search_paths:
            if not Path(search_path).exists():
                continue
            
            try:
                for root, dirs, files in Path(search_path).walk():
                    # Проверяем безопасность
                    is_safe, _ = validate_file_path(Path(root))
                    if not is_safe:
                        continue
                    
                    # Ограничиваем глубину
                    depth = str(root)[len(search_path):].count(os.sep)
                    if depth > config.FILE_SEARCH_MAX_DEPTH:
                        continue
                    
                    for file in files:
                        if filename.lower() in file.lower():
                            full_path = root / file
                            
                            # Проверяем безопасность
                            is_safe, _ = validate_file_path(full_path)
                            if is_safe:
                                found_files.append(str(full_path))
                            
                            if len(found_files) >= config.FILE_SEARCH_MAX_RESULTS:
                                break
                    
                    if len(found_files) >= config.FILE_SEARCH_MAX_RESULTS:
                        break
            
            except (PermissionError, OSError):
                continue
        
        if not found_files:
            logger.warning(f"Файл не найден: {filename}")
            return {
                "success": False,
                "message": f"Файл '{filename}' не найден"
            }
        
        # Форматируем результат
        result = f"📁 Найдено файлов: {len(found_files)}\n\n"
        
        for i, file_path in enumerate(found_files, 1):
            try:
                file_size = Path(file_path).stat().st_size / 1024  # KB
                file_name = Path(file_path).name
                result += f"{i}. {file_name} ({file_size:.1f} KB)\n"
                result += f"   📂 {file_path}\n"
            except:
                result += f"{i}. {file_path}\n"
        
        logger.info(f"✅ Найдено {len(found_files)} файлов")
        
        return {
            "success": True,
            "message": result,
            "files": found_files
        }
    
    async def open_file(self, filepath: str) -> Dict[str, Any]:
        """
        Открывает файл системным приложением
        
        Args:
            filepath: Путь к файлу
        
        Returns:
            {"success": bool, "message": str}
        """
        
        logger.info(f"Открытие файла: {filepath}")
        
        path = Path(filepath)
        
        # Проверка безопасности
        is_safe, reason = validate_file_path(path)
        if not is_safe:
            logger.warning(f"Доступ запрещён: {reason}")
            return {
                "success": False,
                "message": f"❌ Доступ запрещён: {reason}"
            }
        
        if not path.exists():
            logger.warning(f"Файл не найден: {filepath}")
            return {
                "success": False,
                "message": f"Файл не найден: {filepath}"
            }
        
        try:
            if IS_WINDOWS:
                os.startfile(path)
            elif platform.system() == "Darwin":
                subprocess.Popen(["open", str(path)])
            else:
                subprocess.Popen(["xdg-open", str(path)])
            
            logger.info(f"✅ Файл открыт: {path.name}")
            self._log_operation("open_file", str(path), True)

            return {
                "success": True,
                "message": f"✅ Файл '{path.name}' открыт"
            }

        except Exception as e:
            logger.error(f"Ошибка открытия файла: {e}")
            self._log_operation("open_file", str(path), False, str(e))
            return {
                "success": False,
                "message": f"Ошибка открытия: {str(e)}"
            }
    
    # ═══════════════════════════════════════════════════════════
    #                    ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ
    # ═══════════════════════════════════════════════════════════
    
    def _log_operation(self, operation: str, path: str, success: bool, details: str = ""):
        """Логирует операцию"""
        
        if config.LOG_FILE_OPERATIONS:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "operation": operation,
                "path": path,
                "success": success,
                "details": details
            }
            
            self.operation_log.append(log_entry)
            
            # Ограничиваем размер
            if len(self.operation_log) > 100:
                self.operation_log = self.operation_log[-100:]
    
    def get_operation_log(self, limit: int = 20) -> List[Dict]:
        """Возвращает последние операции"""
        return self.operation_log[-limit:]