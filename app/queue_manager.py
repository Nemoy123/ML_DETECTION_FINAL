import asyncio
import uuid
import time
from typing import Dict, List, Optional, Callable
from enum import Enum
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
from app.config import settings

class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class ProcessingTask:
    def __init__(self, task_id: str, session_id: str, file_path: str, filename: str):
        self.task_id = task_id
        self.session_id = session_id
        self.file_path = file_path
        self.filename = filename
        self.status = TaskStatus.PENDING
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.result = None
        self.error = None
        self.progress = 0

class TaskQueueManager:
    def __init__(self, max_workers: int = 6):
        self.tasks: Dict[str, ProcessingTask] = {}
        self.queue: List[str] = []  # Очередь task_id
        self.processing: List[str] = []  # Текущие задачи
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.lock = threading.Lock()
        self._stop_event = threading.Event()
        
        # Запуск обработчика очереди
        self.worker_thread = threading.Thread(target=self._process_queue, daemon=True)
        self.worker_thread.start()
    
    def submit_task(self, session_id: str, file_path: str, filename: str) -> str:
        """Добавление задачи в очередь"""
        task_id = str(uuid.uuid4())
        
        task = ProcessingTask(
            task_id=task_id,
            session_id=session_id,
            file_path=file_path,
            filename=filename
        )
        
        with self.lock:
            self.tasks[task_id] = task
            self.queue.append(task_id)
        
        return task_id
    
    def get_task_status(self, task_id: str) -> Optional[ProcessingTask]:
        """Получение статуса задачи"""
        with self.lock:
            return self.tasks.get(task_id)
    
    def cancel_task(self, task_id: str) -> bool:
        """Отмена задачи"""
        with self.lock:
            if task_id in self.tasks:
                task = self.tasks[task_id]
                if task.status == TaskStatus.PENDING and task_id in self.queue:
                    self.queue.remove(task_id)
                    task.status = TaskStatus.CANCELLED
                    return True
        return False
    
    def _process_queue(self):
        """Фоновая обработка очереди"""
        while not self._stop_event.is_set():
            with self.lock:
                # Проверка свободных слотов
                if len(self.processing) < self.max_workers and self.queue:
                    task_id = self.queue.pop(0)
                    task = self.tasks[task_id]
                    
                    # Обновление статуса
                    task.status = TaskStatus.PROCESSING
                    task.started_at = datetime.now()
                    self.processing.append(task_id)
                    
                    # Запуск обработки в отдельном потоке
                    self.executor.submit(self._process_task, task_id)
            
            # Пауза перед следующей проверкой
            time.sleep(0.1)
    
    def _process_task(self, task_id: str):
        """Обработка задачи"""
        from app.models import yolo_model
        
        try:
            with self.lock:
                task = self.tasks[task_id]
            
            # Обновление прогресса
            task.progress = 10
            
            # Загрузка модели (если еще не загружена)
            task.progress = 20
            
            # Обработка через YOLO
            result = yolo_model.predict_image(task.file_path)
            task.progress = 80
            
            with self.lock:
                if result["success"]:
                    task.status = TaskStatus.COMPLETED
                    task.result = result
                else:
                    task.status = TaskStatus.FAILED
                    task.error = result.get("error", "Unknown error")
                
                task.completed_at = datetime.now()
                task.progress = 100
                
                # Удаление из списка обрабатываемых
                if task_id in self.processing:
                    self.processing.remove(task_id)
                
        except Exception as e:
            with self.lock:
                task = self.tasks[task_id]
                task.status = TaskStatus.FAILED
                task.error = str(e)
                task.completed_at = datetime.now()
                
                if task_id in self.processing:
                    self.processing.remove(task_id)
    
    def get_queue_stats(self) -> Dict:
        """Статистика очереди"""
        with self.lock:
            pending = len([t for t in self.tasks.values() if t.status == TaskStatus.PENDING])
            processing = len(self.processing)
            completed = len([t for t in self.tasks.values() if t.status == TaskStatus.COMPLETED])
            failed = len([t for t in self.tasks.values() if t.status == TaskStatus.FAILED])
            
            return {
                "total_tasks": len(self.tasks),
                "pending": pending,
                "processing": processing,
                "completed": completed,
                "failed": failed,
                "queue_length": len(self.queue),
                "max_workers": self.max_workers,
                "active_workers": processing
            }
    
    def cleanup_old_tasks(self, max_age_hours: int = 24):
        """Очистка старых задач"""
        with self.lock:
            cutoff_time = datetime.now().timestamp() - (max_age_hours * 3600)
            
            tasks_to_remove = []
            for task_id, task in self.tasks.items():
                if task.completed_at and task.completed_at.timestamp() < cutoff_time:
                    tasks_to_remove.append(task_id)
            
            for task_id in tasks_to_remove:
                del self.tasks[task_id]
    
    def stop(self):
        """Остановка менеджера очереди"""
        self._stop_event.set()
        self.executor.shutdown(wait=True)

# Глобальный экземпляр менеджера очереди
task_queue = TaskQueueManager(max_workers=4)  # 3 параллельных обработки