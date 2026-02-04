# pip install fastapi uvicorn python-multipart

from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime
import asyncio
import os
import uvicorn

from app.config import settings
from app.sessions import session_manager
from app.models import yolo_model
from app.utils import read_upload_file, validate_image_bytes, cleanup_old_files, save_bytes
from app.queue_manager import task_queue, TaskStatus


# Инициализация FastAPI
app = FastAPI(
    title="YOLO Object Detection API",
    description="API для обнаружения объектов с использованием YOLO",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Настройте под свои нужды
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Создание директорий при запуске
settings.create_dirs()

# Функция для проверки сессии (добавлена в начало файла)
async def get_session(session_id: Optional[str] = None) -> Tuple[str, dict]:
    """Получение или создание сессии"""
    if session_id:
        session = session_manager.get_session(session_id)
        if not session:
            raise HTTPException(
                status_code=401,
                detail="Session expired or invalid. Please start a new session."
            )
        return session_id, session
    else:
        # Создание новой сессии
        new_session_id = session_manager.create_session()
        session = session_manager.get_session(new_session_id)
        if not session:
            raise HTTPException(
                status_code=500,
                detail="Failed to create session"
            )
        return new_session_id, session

# Эндпоинты
@app.get("/")
async def root():
    """Корневой эндпоинт"""
    return {
        "message": "Transport Object Detection API",
        "version": "1.0.0",
        "endpoints": {
            "create_session": "POST /session",
            "upload_image_async": "POST /detect/async",
            "get_task_status": "GET /task/{task_id}",
            "session_info": "GET /session/{session_id}",
            "health": "GET /health"
        }
    }

@app.post("/session")
async def create_session_endpoint():
    """Создание новой сессии"""
    session_id = session_manager.create_session()
    session = session_manager.get_session(session_id)
    
    return {
        "session_id": session_id,
        "created_at": session["created_at"].isoformat(),
        "expires_in": settings.SESSION_TIMEOUT.total_seconds(),
        "message": "Session created successfully"
    }


@app.post("/detect/async")
async def detect_objects_async(
    request: Request,
    file: UploadFile = File(...),
    session_data: tuple = Depends(get_session),
):
    """
    Асинхронная загрузка изображения в очередь обработки
    """
    session_id, session = session_data
    
    data = await read_upload_file(file)

    if not validate_image_bytes(data):
        raise HTTPException(status_code=400, detail="Invalid image")

    file_path = save_bytes(data, session_id, file.filename)

    
    if os.path.getsize(file_path) > 50 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large (max 50MB)")
    
    try:
        
        # Добавление задачи в очередь
        task_id = task_queue.submit_task(
            session_id=session_id,
            file_path=file_path,
            filename=file.filename
        )
        
        # Обновление сессии
        session_manager.update_session(session_id)
        
        return {
            "task_id": task_id,
            "session_id": session_id,
            "status": "queued",
            "message": "Task added to processing queue",
            "queue_position": len(task_queue.queue) + 1,
            "estimated_wait_time": (len(task_queue.queue) + 1) * 30,  # Примерная оценка
            "filename": file.filename,
            "created_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

@app.get("/task/{task_id}")
async def get_task_status_endpoint(task_id: str, session_data: tuple = Depends(get_session)):
    """
    Получение статуса задачи
    """
    session_id, session = session_data
    
    task = task_queue.get_task_status(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Проверка принадлежности задачи сессии
    if task.session_id != session_id:
        raise HTTPException(status_code=403, detail="Task does not belong to this session")
    
    response = {
        "task_id": task.task_id,
        "session_id": task.session_id,
        "status": task.status.value,
        "filename": task.filename,
        "progress": task.progress,
        "created_at": task.created_at.isoformat() if task.created_at else None,
        "started_at": task.started_at.isoformat() if task.started_at else None,
        "completed_at": task.completed_at.isoformat() if task.completed_at else None
    }
    
    if task.status == TaskStatus.COMPLETED and task.result:
        response["result"] = {
            "detections_count": task.result.get("detections_count", 0),
            "image_size": task.result.get("image_size", {}),
            "model": task.result.get("model", "unknown")
        }
        # Полные данные детекции (опционально)
        if "detections" in task.result:
            response["result"]["detections"] = task.result["detections"]
    
    elif task.status == TaskStatus.FAILED:
        response["error"] = task.error
    
    return response

@app.get("/task/{task_id}/result")
async def get_task_result_endpoint(task_id: str, session_data: tuple = Depends(get_session)):
    """
    Получение полных результатов задачи
    """
    session_id, session = session_data
    
    task = task_queue.get_task_status(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.session_id != session_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if task.status != TaskStatus.COMPLETED:
        raise HTTPException(
            status_code=400, 
            detail=f"Task not completed. Current status: {task.status.value}"
        )
    
    if not task.result:
        raise HTTPException(status_code=404, detail="Result not available")
    
    return {
        "task_id": task_id,
        "session_id": task.session_id,
        "status": "completed",
        "filename": task.filename,
        "processing_time": (
            (task.completed_at - task.started_at).total_seconds() 
            if task.started_at and task.completed_at else None
        ),
        **task.result
    }

@app.delete("/task/{task_id}")
async def cancel_task_endpoint(task_id: str, session_data: tuple = Depends(get_session)):
    """
    Отмена задачи (если еще в очереди)
    """
    session_id, session = session_data
    
    task = task_queue.get_task_status(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    if task.session_id != session_id:
        raise HTTPException(status_code=403, detail="Access denied")
    
    if task.status not in [TaskStatus.PENDING, TaskStatus.PROCESSING]:
        raise HTTPException(
            status_code=400, 
            detail=f"Cannot cancel task with status: {task.status.value}"
        )
    
    success = task_queue.cancel_task(task_id)
    
    if success:
        return {"message": "Task cancelled successfully", "task_id": task_id}
    else:
        raise HTTPException(status_code=400, detail="Failed to cancel task")

@app.get("/session/{session_id}")
async def get_session_info_endpoint(session_id: str):
    """Получение информации о сессии"""
    session = session_manager.get_session(session_id)
    
    if not session:
        raise HTTPException(
            status_code=404,
            detail="Session not found or expired"
        )
    
    time_remaining = settings.SESSION_TIMEOUT - (
        datetime.now() - session["last_activity"]
    )
    
    return {
        "session_id": session_id,
        "created_at": session["created_at"].isoformat(),
        "last_activity": session["last_activity"].isoformat(),
        "time_remaining_seconds": max(0, time_remaining.total_seconds()),
        "files_processed": session["files_processed"],
        "session_data": session["data"]
    }

@app.delete("/session/{session_id}")
async def delete_session_endpoint(session_id: str):
    """Удаление сессии"""
    session_manager.delete_session(session_id)
    return {"message": "Session deleted successfully"}

@app.get("/health")
async def health_check():
    """Проверка здоровья сервиса"""
    active_sessions = session_manager.get_active_sessions_count()
    
    return {
        "status": "healthy",
        "model_loaded": yolo_model.model is not None,
        "active_sessions": active_sessions,
        "upload_dir_exists": os.path.exists(settings.UPLOAD_DIR),
        "cache_dir_exists": os.path.exists(settings.CACHE_DIR)
    }

@app.get("/queue/stats")
async def get_queue_stats_endpoint():
    """
    Статистика очереди
    """
    stats = task_queue.get_queue_stats()
    
    # Очистка старых задач (раз в 24 часа)
    task_queue.cleanup_old_tasks(max_age_hours=24)
    
    return {
        **stats,
        "timestamp": datetime.now().isoformat(),
        "session_count": session_manager.get_active_sessions_count()
    }

@app.websocket("/ws/task/{task_id}")
async def websocket_task_progress(websocket: WebSocket, task_id: str):
    """
    WebSocket для отслеживания прогресса задачи в реальном времени
    """
    await websocket.accept()
    
    try:
        last_progress = -1
        while True:
            task = task_queue.get_task_status(task_id)
            
            if not task:
                await websocket.send_json({
                    "error": "Task not found",
                    "task_id": task_id
                })
                break
            
            if task.progress != last_progress:
                await websocket.send_json({
                    "task_id": task_id,
                    "status": task.status.value,
                    "progress": task.progress,
                    "filename": task.filename
                })
                last_progress = task.progress
            
            if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
                await websocket.send_json({
                    "task_id": task_id,
                    "status": task.status.value,
                    "final": True,
                    "result": task.result if task.status == TaskStatus.COMPLETED else None,
                    "error": task.error if task.status == TaskStatus.FAILED else None
                })
                break
            
            await asyncio.sleep(1)  # Обновление каждую секунду
            
    except WebSocketDisconnect:
        print(f"Client disconnected from task {task_id}")
    except Exception as e:
        await websocket.send_json({
            "error": str(e),
            "task_id": task_id
        })

@app.get("/stats")
async def get_stats_endpoint():
    """Статистика сервиса"""
    import glob
    
    total_files = len(glob.glob(os.path.join(settings.UPLOAD_DIR, "*")))
    cache_files = len(glob.glob(os.path.join(settings.CACHE_DIR, "*")))
    
    return {
        "active_sessions": session_manager.get_active_sessions_count(),
        "total_uploaded_files": total_files,
        "cache_files": cache_files,
        "session_timeout_minutes": settings.SESSION_TIMEOUT.total_seconds() / 60,
        "model": settings.MODEL_NAME,
        "queue_stats": task_queue.get_queue_stats()
    }

# Middleware для логирования
@app.middleware("http")
async def log_requests(request: Request, call_next):
    import time
    start_time = time.time()
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Обработчик на старте приложения
@app.on_event("startup")
async def startup_event():
    """Действия при запуске приложения"""
    print("Starting YOLO API with task queue...")
    print(f"Queue workers: {task_queue.max_workers}")
    
    # Очистка старых файлов
    cleanup_old_files(settings.UPLOAD_DIR, max_age_minutes=60)
    
    # Создание директории для результатов
    results_dir = os.path.join(settings.BASE_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    print(f"Upload directory: {settings.UPLOAD_DIR}")
    print(f"Cache directory: {settings.CACHE_DIR}")
    print(f"Model: {settings.MODEL_NAME}")

@app.on_event("shutdown")
async def shutdown_event():
    """Действия при остановке приложения"""
    print("Shutting down task queue...")
    task_queue.stop()

# if __name__ == "__main__":
#     uvicorn.run(
#         "app.main:app",
#         host="0.0.0.0",
#         port=9000,
#         reload=True,
#         log_level="info"
#     )