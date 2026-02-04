import os
import uuid
from fastapi import UploadFile
from PIL import Image
import io
from app.config import settings

# def save_upload_file(upload_file: UploadFile, session_id: str) -> str:
#     """Сохранение загруженного файла"""
#     # Создание уникального имени файла
#     file_ext = os.path.splitext(upload_file.filename)[1]
#     filename = f"{session_id}_{uuid.uuid4()}{file_ext}"
#     file_path = os.path.join(settings.UPLOAD_DIR, filename)
    
#     # Сохранение файла
#     with open(file_path, "wb") as buffer:
#         content = upload_file.file.read()
#         buffer.write(content)
    
#     return file_path
def save_upload_file(upload_file: UploadFile, session_id: str) -> str:
    file_ext = os.path.splitext(upload_file.filename)[1]
    filename = f"{session_id}_{uuid.uuid4()}{file_ext}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)

    upload_file.file.seek(0) 

    with open(file_path, "wb") as buffer:
        buffer.write(upload_file.file.read())

    upload_file.file.seek(0)  

    return file_path

async def read_upload_file(file: UploadFile) -> bytes:
    data = await file.read()

    if not data:
        raise ValueError("Empty upload")

    return data

def validate_image_bytes(data: bytes) -> bool:
    try:
        image = Image.open(io.BytesIO(data))
        image.verify()
        return True
    except Exception:
        return False

def save_bytes(data: bytes, session_id: str, filename: str) -> str:
    ext = os.path.splitext(filename)[1]
    path = os.path.join(
        settings.UPLOAD_DIR,
        f"{session_id}_{uuid.uuid4()}{ext}"
    )

    with open(path, "wb") as f:
        f.write(data)

    return path

# def validate_image_file(file: UploadFile) -> bool:
#     """Проверка, что файл является изображением"""
#     allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
#     file_ext = os.path.splitext(file.filename.lower())[1]
    
#     if file_ext not in allowed_extensions:
#         return False
    
#     # Дополнительная проверка через PIL
#     try:
#         content = file.file.read(1024)
#         file.file.seek(0)
#         image = Image.open(io.BytesIO(content))
#         image.verify()
#         file.file.seek(0)
#         return True
#     except:
#         return False

def cleanup_old_files(directory: str, max_age_minutes: int = 60):
    """Очистка старых файлов"""
    import time
    current_time = time.time()
    
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path):
            file_age = current_time - os.path.getmtime(file_path)
            if file_age > max_age_minutes * 60:
                os.remove(file_path)