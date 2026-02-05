FROM python:3.11-slim

WORKDIR /app

# системные зависимости
RUN apt-get update && apt-get install -y gcc   
RUN apt-get install -y libgl1 libglib2.0-0 
RUN rm -rf /var/lib/apt/lists/*
# зависимости Python
COPY requirements.txt .
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# код
COPY . .

# порт
EXPOSE 9000

# запуск
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]
