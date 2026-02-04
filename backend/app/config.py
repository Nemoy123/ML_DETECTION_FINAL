import os
from datetime import timedelta

class Settings:
    # Пути
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
    CACHE_DIR = os.path.join(BASE_DIR, "cache")
    
    # Настройки сессий
    SESSION_TIMEOUT = timedelta(minutes=15)
    SESSION_SECRET_KEY = os.getenv("SESSION_SECRET", "your-secret-key-change-in-production")
    
    # Настройки модели
    # MODEL_NAME = "D:\\develop\\python\\ML_Diplom_backend\\weights\\26m_640_50map.pt"  # или yolo11s.pt для лучшей точности
    MODEL_NAME = "weights/best9c_57.onnx" 

    # Создание директорий
    @staticmethod
    def create_dirs():
        os.makedirs(Settings.UPLOAD_DIR, exist_ok=True)
        os.makedirs(Settings.CACHE_DIR, exist_ok=True)

settings = Settings()