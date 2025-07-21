# 🌐 API - Документация

## Обзор

`api.py` - REST API сервер для предоставления доступа к модели предсказания конверсии через HTTP интерфейс.

## Архитектура

### Технологии
- **Фреймворк**: Flask
- **Модель**: SberAutoModel
- **Логирование**: Python logging
- **Сериализация**: JSON

### Структура
```
api.py
├── Глобальные переменные
│   ├── app (Flask)
│   └── model (SberAutoModel)
├── Вспомогательные функции
│   └── load_model()
└── Эндпоинты
    ├── GET /health
    ├── POST /predict
    ├── POST /predict_batch
    ├── GET /model_info
    ├── GET /example
    ├── GET /features
    └── GET /stats
```

## Запуск сервера

### Командная строка
```bash
cd code
python api.py
```

### Переменные окружения
```bash
export MODEL_PATH="../build/sber_auto_model.pkl"
export API_PORT=5001
export API_HOST="0.0.0.0"
```

### Конфигурация
- **Хост**: 0.0.0.0 (доступ со всех интерфейсов)
- **Порт**: 5001
- **Режим**: Production (debug=False)
- **Логирование**: INFO уровень

## Эндпоинты

### 1. `GET /health`

Проверка здоровья API и статуса модели.

#### Запрос
```bash
curl http://localhost:5001/health
```

#### Ответ
```json
{
    "status": "healthy",
    "model_loaded": true,
    "timestamp": 1234567890.123
}
```

#### Поля ответа
- `status` (string): Статус API ("healthy" или "unhealthy")
- `model_loaded` (boolean): Загружена ли модель
- `timestamp` (float): Время запроса в Unix timestamp

#### Коды ответов
- `200 OK`: API работает нормально
- `500 Internal Server Error`: Ошибка сервера

### 2. `POST /predict`

Предсказание конверсии для одной сессии.

#### Запрос
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "visit_number": 1,
    "total_hits": 5,
    "unique_pages": 3,
    "session_duration": 120,
    "visit_hour": 14,
    "visit_weekday": 2,
    "is_weekend": 0,
    "is_mobile": 1,
    "is_android": 0,
    "is_ios": 1,
    "is_desktop": 0,
    "is_tablet": 0,
    "is_moscow": 1,
    "is_paid": 1,
    "avg_time_per_page": 24.0,
    "bounce_rate": 0,
    "deep_engagement": 0,
    "long_session": 0
  }'
```

#### Обязательные поля
- `visit_number` (int): Номер посещения
- `total_hits` (int): Общее количество действий
- `unique_pages` (int): Количество уникальных страниц
- `session_duration` (int): Длительность сессии в секундах
- `visit_hour` (int): Час посещения (0-23)
- `visit_weekday` (int): День недели (0-6)
- `is_weekend` (int): Выходной день (0/1)
- `is_mobile` (int): Мобильное устройство (0/1)
- `is_android` (int): Android устройство (0/1)
- `is_ios` (int): iOS устройство (0/1)
- `is_desktop` (int): Десктоп (0/1)
- `is_tablet` (int): Планшет (0/1)
- `is_moscow` (int): Москва (0/1)
- `is_paid` (int): Платный трафик (0/1)
- `avg_time_per_page` (float): Среднее время на страницу
- `bounce_rate` (int): Показатель отказов (0/1)
- `deep_engagement` (int): Глубокое вовлечение (0/1)
- `long_session` (int): Длительная сессия (0/1)

#### Ответ
```json
{
    "prediction": 1,
    "probability": 0.75,
    "will_convert": true,
    "conversion_probability": "75.00%",
    "confidence_level": "высокая",
    "execution_time": 0.045,
    "status": "success"
}
```

#### Поля ответа
- `prediction` (int): Предсказание (0 или 1)
- `probability` (float): Вероятность конверсии (0.0-1.0)
- `will_convert` (boolean): Будет ли конверсия
- `conversion_probability` (string): Вероятность в процентах
- `confidence_level` (string): Уровень уверенности ("низкая", "средняя", "высокая")
- `execution_time` (float): Время выполнения в секундах
- `status` (string): Статус запроса ("success" или "error")

#### Коды ответов
- `200 OK`: Успешное предсказание
- `400 Bad Request`: Неверные данные
- `500 Internal Server Error`: Ошибка модели

### 3. `POST /predict_batch`

Пакетное предсказание для нескольких сессий.

#### Запрос
```bash
curl -X POST http://localhost:5001/predict_batch \
  -H "Content-Type: application/json" \
  -d '{
    "sessions": [
        {
            "visit_number": 1,
            "total_hits": 5,
            "unique_pages": 3,
            "session_duration": 120,
            "visit_hour": 14,
            "is_mobile": 1,
            "is_ios": 1
        },
        {
            "visit_number": 2,
            "total_hits": 10,
            "unique_pages": 7,
            "session_duration": 300,
            "visit_hour": 16,
            "is_desktop": 1
        }
    ]
}'
```

#### Структура запроса
- `sessions` (array): Массив сессий для предсказания
- Каждая сессия содержит те же поля, что и в `/predict`

#### Ограничения
- Максимум 1000 сессий за запрос
- Все сессии должны содержать одинаковый набор полей

#### Ответ
```json
{
    "predictions": [
        {
            "session_id": 0,
            "prediction": 1,
            "probability": 0.75,
            "will_convert": true,
            "conversion_probability": "75.00%",
            "confidence_level": "высокая"
        },
        {
            "session_id": 1,
            "prediction": 0,
            "probability": 0.25,
            "will_convert": false,
            "conversion_probability": "25.00%",
            "confidence_level": "средняя"
        }
    ],
    "total_sessions": 2,
    "execution_time": 0.089,
    "status": "success",
    "statistics": {
        "successful_predictions": 2,
        "failed_predictions": 0,
        "average_probability": 50.0,
        "high_confidence_predictions": 1,
        "high_confidence_percentage": 50.0
    }
}
```

#### Поля ответа
- `predictions` (array): Массив результатов предсказаний
- `total_sessions` (int): Общее количество сессий
- `execution_time` (float): Время выполнения в секундах
- `status` (string): Статус запроса
- `statistics` (object): Статистика результатов
  - `successful_predictions` (int): Успешные предсказания
  - `failed_predictions` (int): Неудачные предсказания
  - `average_probability` (float): Средняя вероятность
  - `high_confidence_predictions` (int): Предсказания с высокой уверенностью
  - `high_confidence_percentage` (float): Процент высокоуверенных предсказаний

#### Коды ответов
- `200 OK`: Успешное пакетное предсказание
- `400 Bad Request`: Неверные данные или превышен лимит
- `500 Internal Server Error`: Ошибка модели

### 4. `GET /model_info`

Информация о загруженной модели.

#### Запрос
```bash
curl http://localhost:5001/model_info
```

#### Ответ
```json
{
    "feature_count": 40,
    "target_actions_count": 22,
    "feature_names": [
        "visit_number",
        "total_hits",
        "unique_pages",
        "session_duration",
        "unique_events"
    ],
    "target_actions": [
        "sub_submit_success",
        "sub_call_number_click",
        "sub_callback_submit_click",
        "phone_auth_success",
        "start_chat"
    ],
    "status": "loaded"
}
```

#### Поля ответа
- `feature_count` (int): Количество признаков модели
- `target_actions_count` (int): Количество целевых действий
- `feature_names` (array): Первые 10 имен признаков
- `target_actions` (array): Первые 5 целевых действий
- `status` (string): Статус модели ("loaded" или "not_loaded")

#### Коды ответов
- `200 OK`: Информация получена
- `500 Internal Server Error`: Модель не загружена

### 5. `GET /example`

Пример данных для предсказания.

#### Запрос
```bash
curl http://localhost:5001/example
```

#### Ответ
```json
{
    "example_data": {
        "visit_number": 1,
        "total_hits": 5,
        "unique_pages": 3,
        "session_duration": 120,
        "visit_hour": 14,
        "visit_weekday": 2,
        "is_weekend": 0,
        "is_mobile": 1,
        "is_android": 0,
        "is_ios": 1,
        "is_desktop": 0,
        "is_tablet": 0,
        "is_moscow": 1,
        "is_paid": 1,
        "avg_time_per_page": 24.0,
        "bounce_rate": 0,
        "deep_engagement": 0,
        "long_session": 0
    },
    "description": "Пример данных для предсказания конверсии"
}
```

#### Поля ответа
- `example_data` (object): Пример данных сессии
- `description` (string): Описание примера

### 6. `GET /features`

Список всех признаков модели с категоризацией.

#### Запрос
```bash
curl http://localhost:5001/features
```

#### Ответ
```json
{
    "features": [
        "visit_number",
        "total_hits",
        "unique_pages",
        "session_duration",
        "unique_events"
    ],
    "feature_count": 40,
    "feature_categories": {
        "temporal": [
            "visit_hour",
            "visit_weekday",
            "is_weekend",
            "is_workday",
            "is_morning",
            "is_afternoon",
            "is_evening",
            "is_night"
        ],
        "device": [
            "is_mobile",
            "is_android",
            "is_ios",
            "is_desktop",
            "is_tablet",
            "is_windows",
            "is_macos"
        ],
        "geographic": [
            "is_moscow",
            "is_spb",
            "is_million_plus",
            "is_regional_center",
            "city_conversion_rate",
            "city_avg_duration",
            "city_avg_hits",
            "city_tier_low",
            "city_tier_medium",
            "city_tier_high",
            "city_tier_very_high"
        ],
        "behavioral": [
            "total_hits",
            "unique_pages",
            "session_duration",
            "unique_events",
            "avg_time_per_page",
            "bounce_rate",
            "deep_engagement",
            "long_session",
            "very_long_session",
            "high_activity",
            "very_high_activity",
            "events_per_page",
            "engagement_score"
        ],
        "traffic": [
            "is_paid",
            "is_organic",
            "is_referral",
            "is_direct"
        ]
    }
}
```

#### Поля ответа
- `features` (array): Полный список признаков
- `feature_count` (int): Количество признаков
- `feature_categories` (object): Признаки по категориям
  - `temporal`: Временные признаки
  - `device`: Признаки устройств
  - `geographic`: Географические признаки
  - `behavioral`: Поведенческие признаки
  - `traffic`: Признаки источников трафика

### 7. `GET /stats`

Статистика использования API.

#### Запрос
```bash
curl http://localhost:5001/stats
```

#### Ответ
```json
{
    "uptime": 1234567890.123,
    "endpoints": [
        "GET /health - проверка здоровья",
        "POST /predict - предсказание для одной сессии",
        "POST /predict_batch - пакетное предсказание",
        "GET /model_info - информация о модели",
        "GET /example - пример данных",
        "GET /features - список признаков",
        "GET /stats - статистика API"
    ]
}
```

#### Поля ответа
- `uptime` (float): Время работы сервера в секундах
- `endpoints` (array): Список доступных эндпоинтов

## Обработка ошибок

### Общие ошибки

#### 1. Модель не загружена
```json
{
    "error": "Модель не загружена",
    "status": "error"
}
```

#### 2. Неверные данные
```json
{
    "error": "Данные не предоставлены",
    "status": "error"
}
```

#### 3. Ошибка предсказания
```json
{
    "error": "Ошибка предсказания: ...",
    "execution_time": 0.045,
    "status": "error"
}
```

### Коды HTTP статусов

- `200 OK`: Успешный запрос
- `400 Bad Request`: Неверные данные запроса
- `500 Internal Server Error`: Ошибка сервера

## Логирование

### Уровни логирования
- **INFO**: Основные операции (загрузка модели, предсказания)
- **ERROR**: Ошибки и исключения
- **WARNING**: Предупреждения

### Примеры логов
```
INFO:root:✅ Модель успешно загружена
INFO:root:✅ Предсказание выполнено за 0.045с
INFO:root:📊 Результат: 75.00% (уверенность: высокая)
ERROR:root:❌ Ошибка предсказания: Модель не загружена
```

## Производительность

### Метрики
- **Время ответа**: <0.1 секунды для одиночных предсказаний
- **Пакетная обработка**: до 1000 сессий за запрос
- **Память**: ~100MB для модели
- **CPU**: Минимальное использование

### Оптимизации
- Кэширование модели в памяти
- Параллельная обработка пакетных запросов
- Эффективная сериализация JSON
- Ленивая загрузка модели

## Безопасность

### Рекомендации
1. Используйте HTTPS в продакшене
2. Ограничьте доступ по IP при необходимости
3. Добавьте аутентификацию для критических эндпоинтов
4. Валидируйте входные данные
5. Ограничьте размер запросов

### Примеры валидации
```python
# Проверка размера запроса
if len(sessions) > 1000:
    return jsonify({'error': 'Максимальное количество сессий: 1000'}), 400

# Проверка обязательных полей
required_fields = ['visit_number', 'total_hits', 'unique_pages']
for field in required_fields:
    if field not in data:
        return jsonify({'error': f'Отсутствует обязательное поле: {field}'}), 400
```

## Мониторинг

### Метрики для отслеживания
- Количество запросов в минуту
- Среднее время ответа
- Процент ошибок
- Использование памяти
- Загрузка CPU

### Примеры мониторинга
```python
# Prometheus метрики
from prometheus_client import Counter, Histogram

request_count = Counter('api_requests_total', 'Total API requests')
request_duration = Histogram('api_request_duration_seconds', 'API request duration')

@app.route('/predict', methods=['POST'])
def predict():
    request_count.inc()
    with request_duration.time():
        # ... логика предсказания
```

## Примеры клиентов

### Python
```python
import requests
import json

def predict_conversion(data):
    response = requests.post(
        'http://localhost:5001/predict',
        json=data,
        timeout=10
    )
    
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.text}")

# Использование
data = {
    "visit_number": 1,
    "total_hits": 5,
    "unique_pages": 3,
    "session_duration": 120,
    "visit_hour": 14,
    "is_mobile": 1,
    "is_ios": 1
}

result = predict_conversion(data)
print(f"Конверсия: {result['conversion_probability']}")
```

### JavaScript
```javascript
async function predictConversion(data) {
    const response = await fetch('http://localhost:5001/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    });
    
    if (!response.ok) {
        throw new Error(`API Error: ${response.statusText}`);
    }
    
    return await response.json();
}

// Использование
const data = {
    visit_number: 1,
    total_hits: 5,
    unique_pages: 3,
    session_duration: 120,
    visit_hour: 14,
    is_mobile: 1,
    is_ios: 1
};

predictConversion(data)
    .then(result => {
        console.log(`Конверсия: ${result.conversion_probability}`);
    })
    .catch(error => {
        console.error('Ошибка:', error);
    });
```

### cURL
```bash
# Проверка здоровья
curl http://localhost:5001/health

# Предсказание
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "visit_number": 1,
    "total_hits": 5,
    "unique_pages": 3,
    "session_duration": 120,
    "visit_hour": 14,
    "is_mobile": 1,
    "is_ios": 1
  }'

# Информация о модели
curl http://localhost:5001/model_info
```

## Развертывание

### Docker
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY code/ ./code/
COPY data/ ./data/
COPY build/ ./build/

EXPOSE 5001
CMD ["python", "code/api.py"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - "5001:5001"
    volumes:
      - ./data:/app/data
      - ./build:/app/build
    environment:
      - MODEL_PATH=/app/build/sber_auto_model.pkl
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sber-auto-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sber-auto-api
  template:
    metadata:
      labels:
        app: sber-auto-api
    spec:
      containers:
      - name: api
        image: sber-auto-api:latest
        ports:
        - containerPort: 5001
        env:
        - name: MODEL_PATH
          value: "/app/build/sber_auto_model.pkl"
---
apiVersion: v1
kind: Service
metadata:
  name: sber-auto-api-service
spec:
  selector:
    app: sber-auto-api
  ports:
  - port: 80
    targetPort: 5001
  type: LoadBalancer
``` 