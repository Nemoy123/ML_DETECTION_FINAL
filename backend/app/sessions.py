import uuid
import time
from typing import Dict, Optional
from datetime import datetime, timedelta
from app.config import settings

class SessionManager:
    def __init__(self):
        self.sessions: Dict[str, dict] = {}
        self.cleanup_interval = 60  # очистка каждые 60 секунд
        self.last_cleanup = time.time()
    
    def create_session(self, user_data: dict = None) -> str:
        """Создание новой сессии"""
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "created_at": datetime.now(),
            "last_activity": datetime.now(),
            "data": user_data or {},
            "files_processed": 0
        }
        self._cleanup_old_sessions()
        return session_id
    
    def get_session(self, session_id: str) -> Optional[dict]:
        """Получение сессии по ID"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            # Проверка времени жизни
            if datetime.now() - session["last_activity"] > settings.SESSION_TIMEOUT:
                self.delete_session(session_id)
                return None
            
            # Обновление времени активности
            session["last_activity"] = datetime.now()
            return session
        return None
    
    def update_session(self, session_id: str, data: dict = None):
        """Обновление данных сессии"""
        session = self.get_session(session_id)
        if session:
            if data:
                session["data"].update(data)
            session["files_processed"] += 1
    
    def delete_session(self, session_id: str):
        """Удаление сессии"""
        if session_id in self.sessions:
            del self.sessions[session_id]
    
    def _cleanup_old_sessions(self):
        """Очистка устаревших сессий"""
        current_time = time.time()
        if current_time - self.last_cleanup > self.cleanup_interval:
            now = datetime.now()
            expired_sessions = [
                session_id for session_id, session in self.sessions.items()
                if now - session["last_activity"] > settings.SESSION_TIMEOUT
            ]
            for session_id in expired_sessions:
                del self.sessions[session_id]
            self.last_cleanup = current_time
    
    def get_active_sessions_count(self) -> int:
        """Количество активных сессий"""
        self._cleanup_old_sessions()
        return len(self.sessions)

# Глобальный менеджер сессий
session_manager = SessionManager()