"""
Универсальный поиск приложений на ПК с поддержкой игр
"""

import os
import winreg
from pathlib import Path
from typing import List, Dict, Optional
import json

class AppFinder:
    """Находит любое приложение на компьютере, включая игры"""
    
    def __init__(self):
        self.app_cache_file = Path("data/app_cache.json")
        self.app_cache = self._load_cache()
        
        # Если кэш пустой — сканируем систему
        if not self.app_cache:
            print("🔍 Первый запуск: сканирую систему...")
            self.scan_system()
    
    def _load_cache(self) -> Dict:
        """Загружает кэш приложений"""
        if self.app_cache_file.exists():
            with open(self.app_cache_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _save_cache(self):
        """Сохраняет кэш"""
        self.app_cache_file.parent.mkdir(exist_ok=True)
        with open(self.app_cache_file, 'w', encoding='utf-8') as f:
            json.dump(self.app_cache, f, ensure_ascii=False, indent=2)
    
    def scan_system(self):
        """Сканирует систему на наличие приложений"""
        print("   Сканирую реестр Windows...")
        apps = {}
        
        # 1. Из реестра
        apps.update(self._scan_registry())
        
        # 2. Program Files
        print("   Сканирую Program Files...")
        apps.update(self._scan_program_files())
        
        # 3. Меню Пуск
        print("   Сканирую меню Пуск...")
        apps.update(self._scan_start_menu())
        
        # 4. Desktop
        print("   Сканирую Desktop...")
        apps.update(self._scan_desktop())
        
        # 5. Игры
        print("   Сканирую игры (Steam, Epic, etc.)...")
        apps.update(self._scan_game_launchers())
        
        self.app_cache = apps
        self._save_cache()
        
        print(f"✅ Найдено {len(apps)} приложений")
    
    def _scan_game_launchers(self) -> Dict:
        """Сканирует папки игровых лаунчеров"""
        apps = {}
        
        # Steam
        steam_paths = self._find_steam_games()
        for game_name, game_path in steam_paths.items():
            apps[game_name.lower()] = {
                'name': game_name,
                'path': game_path,
                'source': 'steam'
            }
        
        # Epic Games
        epic_paths = self._find_epic_games()
        for game_name, game_path in epic_paths.items():
            apps[game_name.lower()] = {
                'name': game_name,
                'path': game_path,
                'source': 'epic'
            }
        
        # Общие папки с играми
        game_folders = [
            r"C:\Games",
            r"D:\Games",
            r"E:\Games",
            r"F:\Games",
            r"F:\Program Files (x86)\Steam\steamapps\common",
        ]
        
        for folder in game_folders:
            if os.path.exists(folder):
                apps.update(self._scan_game_folder(folder))
        
        return apps
    
    def _find_steam_games(self) -> Dict:
        """Находит игры Steam"""
        games = {}
        steam_paths_to_check = []
        
        # Реестр Steam
        try:
            key = winreg.OpenKey(winreg.HKEY_CURRENT_USER, r"Software\Valve\Steam")
            steam_path, _ = winreg.QueryValueEx(key, "SteamPath")
            winreg.CloseKey(key)
            steam_paths_to_check.append(Path(steam_path))
        except:
            pass
        
        # Стандартные пути на всех дисках
        for drive in ['C', 'D', 'E', 'F', 'G']:
            steam_paths_to_check.extend([
                Path(f"{drive}:/Steam"),
                Path(f"{drive}:/Program Files (x86)/Steam"),
                Path(f"{drive}:/Program Files/Steam"),
            ])
        
        # Сканируем пути
        for steam_base_path in steam_paths_to_check:
            common_path = steam_base_path / "steamapps" / "common"
            
            if not common_path.exists():
                continue
            
            print(f"   🔍 Сканирую Steam: {common_path}")
            
            try:
                for game_folder in common_path.iterdir():
                    if not game_folder.is_dir():
                        continue
                    
                    game_name_lower = game_folder.name.lower()
                    
                    if game_name_lower in games:
                        continue
                    
                    # Ищем exe
                    for exe_file in game_folder.glob("*.exe"):
                        exe_name_lower = exe_file.name.lower()
                        
                        if any(skip in exe_name_lower for skip in ['unins', 'crash', 'report', 'launcher', 'updater']):
                            continue
                        
                        games[game_name_lower] = str(exe_file)
                        break
                    
                    # Если не нашли, ищем глубже
                    if game_name_lower not in games:
                        for exe_file in game_folder.rglob("*.exe"):
                            depth = str(exe_file.relative_to(game_folder)).count(os.sep)
                            if depth > 2:
                                continue
                            
                            exe_name_lower = exe_file.name.lower()
                            
                            # Расширенный список служебных файлов
                            skip_patterns = [
                                'unins', 'crash', 'report', 'launcher',
                                'updater', 'support', 'redist', 'redistributable',
                                'vcredist', 'directx', 'dotnet', 'physx',
                                'easyanticheat', 'battleye', 'setup', 'install'
                            ]
                            
                            if any(skip in exe_name_lower for skip in skip_patterns):
                                continue
                            
                            # Приоритет: exe с названием игры
                            game_name_clean = game_folder.name.lower().replace(' ', '').replace('-', '')
                            exe_name_clean = exe_file.stem.lower().replace(' ', '').replace('-', '')
                            
                            if game_name_clean in exe_name_clean or exe_name_clean in game_name_clean:
                                games[game_name_lower] = str(exe_file)
                                break
            
            except Exception as e:
                print(f"   ⚠️ Ошибка: {e}")
                continue
        
        return games
    
    def _find_epic_games(self) -> Dict:
        """Находит игры Epic Games"""
        games = {}
        
        try:
            manifests_path = Path(os.environ.get('PROGRAMDATA', '')) / "Epic" / "EpicGamesLauncher" / "Data" / "Manifests"
            
            if manifests_path.exists():
                for manifest_file in manifests_path.glob("*.item"):
                    try:
                        with open(manifest_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            
                            game_name = data.get('DisplayName', '')
                            install_location = data.get('InstallLocation', '')
                            
                            if game_name and install_location and os.path.exists(install_location):
                                install_path = Path(install_location)
                                for exe_file in install_path.rglob("*.exe"):
                                    if game_name.lower() in exe_file.name.lower():
                                        games[game_name] = str(exe_file)
                                        break
                    except:
                        continue
        except:
            pass
        
        return games
    
    def _scan_game_folder(self, folder_path: str) -> Dict:
        """Сканирует папку с играми"""
        apps = {}
        
        try:
            folder = Path(folder_path)
            
            for game_dir in folder.iterdir():
                if not game_dir.is_dir():
                    continue
                
                # Ищем exe
                for exe_file in game_dir.glob("*.exe"):
                    if any(skip in exe_file.name.lower() for skip in ['unins', 'setup', 'crash']):
                        continue
                    
                    game_name = game_dir.name
                    apps[game_name.lower()] = {
                        'name': game_name,
                        'path': str(exe_file),
                        'source': 'games_folder'
                    }
                    break
        except Exception as e:
            print(f"   ⚠️ Ошибка: {e}")
        
        return apps
    
    def _scan_registry(self) -> Dict:
        """Сканирует реестр Windows"""
        apps = {}
        
        registry_paths = [
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall"),
            (winreg.HKEY_CURRENT_USER, r"SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall"),
        ]
        
        for hkey, path in registry_paths:
            try:
                key = winreg.OpenKey(hkey, path)
                for i in range(winreg.QueryInfoKey(key)[0]):
                    try:
                        subkey_name = winreg.EnumKey(key, i)
                        subkey = winreg.OpenKey(key, subkey_name)
                        
                        try:
                            name = winreg.QueryValueEx(subkey, "DisplayName")[0]
                            icon = winreg.QueryValueEx(subkey, "DisplayIcon")[0]
                            
                            if icon and os.path.exists(icon.split(',')[0]):
                                exe_path = icon.split(',')[0].strip('"')
                                name_lower = name.lower()
                                apps[name_lower] = {
                                    'name': name,
                                    'path': exe_path,
                                    'source': 'registry'
                                }
                        except:
                            pass
                        
                        winreg.CloseKey(subkey)
                    except:
                        continue
                
                winreg.CloseKey(key)
            except:
                continue
        
        return apps
    
    def _scan_program_files(self) -> Dict:
        """Сканирует Program Files"""
        apps = {}
        
        locations = [
            r"C:\Program Files",
            r"C:\Program Files (x86)",
            r"D:\Program Files",
            r"D:\Program Files (x86)",
            r"E:\Program Files",
            r"F:\Program Files",
            r"F:\Program Files (x86)",
        ]
        
        for location in locations:
            if not os.path.exists(location):
                continue
            
            try:
                for root, dirs, files in os.walk(location):
                    depth = root[len(location):].count(os.sep)
                    if depth > 2:
                        continue
                    
                    for file in files:
                        if file.endswith('.exe'):
                            full_path = os.path.join(root, file)
                            name = file[:-4].lower()
                            
                            if any(skip in name for skip in ['unins', 'setup', 'update']):
                                continue
                            
                            apps[name] = {
                                'name': file[:-4],
                                'path': full_path,
                                'source': 'program_files'
                            }
            except:
                continue
        
        return apps
    
    def _scan_start_menu(self) -> Dict:
        """Сканирует меню Пуск"""
        apps = {}
        
        start_menu_paths = [
            Path(os.environ.get('APPDATA', '')) / r"Microsoft\Windows\Start Menu\Programs",
            Path(os.environ.get('PROGRAMDATA', '')) / r"Microsoft\Windows\Start Menu\Programs",
        ]
        
        for start_path in start_menu_paths:
            if not start_path.exists():
                continue
            
            for lnk_file in start_path.rglob("*.lnk"):
                try:
                    import win32com.client
                    shell = win32com.client.Dispatch("WScript.Shell")
                    shortcut = shell.CreateShortCut(str(lnk_file))
                    target = shortcut.Targetpath
                    
                    if target and os.path.exists(target) and target.endswith('.exe'):
                        name = lnk_file.stem.lower()
                        apps[name] = {
                            'name': lnk_file.stem,
                            'path': target,
                            'source': 'start_menu'
                        }
                except:
                    continue
        
        return apps
    
    def _scan_desktop(self) -> Dict:
        """Сканирует рабочий стол"""
        apps = {}
        desktop = Path.home() / "Desktop"
        
        if desktop.exists():
            for item in desktop.glob("*.lnk"):
                try:
                    import win32com.client
                    shell = win32com.client.Dispatch("WScript.Shell")
                    shortcut = shell.CreateShortCut(str(item))
                    target = shortcut.Targetpath
                    
                    if target and os.path.exists(target):
                        name = item.stem.lower()
                        apps[name] = {
                            'name': item.stem,
                            'path': target,
                            'source': 'desktop'
                        }
                except:
                    continue
        
        return apps
    
    def find_app(self, app_name: str) -> Optional[Dict]:
        """Ищет приложение по имени"""
        app_name_lower = app_name.lower().strip()
        
        # 1. Точное совпадение
        if app_name_lower in self.app_cache:
            return self.app_cache[app_name_lower]
        
        # 2. Частичное
        for key, app in self.app_cache.items():
            if app_name_lower in key or key in app_name_lower:
                return app
        
        # 3. Синонимы
        aliases = {
            'браузер': ['chrome', 'firefox', 'edge'],
            'блокнот': ['notepad'],
            'стим': ['steam'],
            'сталкрафт': ['stalcraft'],
            'дискорд': ['discord'],
        }
        
        for alias, targets in aliases.items():
            if alias in app_name_lower:
                for target in targets:
                    if target in self.app_cache:
                        return self.app_cache[target]
                    
                    for key, app in self.app_cache.items():
                        if target in key:
                            return app
        
        # 4. Нечёткий поиск
        for key, app in self.app_cache.items():
            if self._fuzzy_match(app_name_lower, key):
                return app
        
        return None
    
    def _fuzzy_match(self, s1: str, s2: str, threshold: int = 3) -> bool:
        """Нечёткое сравнение"""
        if abs(len(s1) - len(s2)) > threshold:
            return False
        
        differences = sum(c1 != c2 for c1, c2 in zip(s1, s2))
        return differences <= threshold
    
    def list_all_apps(self, limit: int = 50) -> List[Dict]:
        """Список приложений"""
        return list(self.app_cache.values())[:limit]
    
    def rescan(self):
        """Пересканировать систему"""
        print("🔄 Пересканирую систему...")
        self.app_cache = {}
        self.scan_system()