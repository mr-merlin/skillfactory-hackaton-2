# 📊 SberAutoModel - Документация

## Обзор

`SberAutoModel` - основной класс для работы с моделью машинного обучения, предназначенной для предсказания конверсии пользователей на сайте "СберАвтоподписка".

## Класс SberAutoModel

### Инициализация

```python
from sber_auto_model import SberAutoModel

model = SberAutoModel()
```

### Атрибуты

- `model`: Обученная модель Random Forest Classifier
- `feature_names`: Список имен признаков
- `target_actions`: Список целевых действий
- `scaler`: Нормализатор данных (не используется в текущей версии)

## Методы

### `load_data()`

Загружает данные из файлов pickle.

```python
sessions, hits = model.load_data()
```

**Возвращает:**
- `sessions` (DataFrame): Данные сессий пользователей
- `hits` (DataFrame): Данные хитов пользователей

**Файлы данных:**
- `../data/ga_sessions.pkl` - сессии пользователей
- `../data/ga_hits.pkl` - хиты пользователей

### `define_target_actions(hits)`

Определяет целевые действия на основе анализа событий.

```python
target_actions = model.define_target_actions(hits)
```

**Параметры:**
- `hits` (DataFrame): Данные хитов пользователей

**Алгоритм:**
1. Анализирует все уникальные события
2. Ищет ключевые слова в названиях событий
3. Формирует список целевых действий

**Ключевые слова для поиска:**
- submit, success, call, contact, request
- callback, claim, chat, auth, phone
- sms, code, confirm, start_chat
- user_message, proactive, invitation

**Возвращает:**
- `list`: Список целевых действий

### `create_features(sessions, hits)`

Создает признаки для модели машинного обучения.

```python
df = model.create_features(sessions, hits)
```

**Параметры:**
- `sessions` (DataFrame): Данные сессий
- `hits` (DataFrame): Данные хитов

**Создаваемые признаки:**

#### Базовые признаки
- `visit_number`: Номер посещения
- `total_hits`: Общее количество действий
- `unique_pages`: Количество уникальных страниц
- `session_duration`: Длительность сессии (секунды)
- `unique_events`: Количество уникальных событий

#### Временные признаки
- `visit_hour`: Час посещения (0-23)
- `visit_weekday`: День недели (0-6, где 0=понедельник)
- `is_weekend`: Выходной день (0/1)
- `is_workday`: Рабочий день (0/1)
- `is_morning`: Утренние часы 6-11 (0/1)
- `is_afternoon`: Дневные часы 12-17 (0/1)
- `is_evening`: Вечерние часы 18-23 (0/1)
- `is_night`: Ночные часы 0-5 (0/1)

#### Признаки устройств
- `is_mobile`: Мобильное устройство (0/1)
- `is_android`: Android устройство (0/1)
- `is_ios`: iOS устройство (0/1)
- `is_desktop`: Десктоп (0/1)
- `is_tablet`: Планшет (0/1)
- `is_windows`: Windows ОС (0/1)
- `is_macos`: macOS (0/1)

#### Признаки источников трафика
- `is_paid`: Платный трафик (0/1)
- `is_organic`: Органический трафик (0/1)
- `is_referral`: Реферальный трафик (0/1)
- `is_direct`: Прямой трафик (0/1)

#### Поведенческие метрики
- `avg_time_per_page`: Среднее время на страницу
- `bounce_rate`: Показатель отказов (0/1)
- `deep_engagement`: Глубокое вовлечение (0/1)
- `long_session`: Длительная сессия >5 мин (0/1)
- `very_long_session`: Очень длительная сессия >10 мин (0/1)
- `high_activity`: Высокая активность >=10 действий (0/1)
- `very_high_activity`: Очень высокая активность >=15 действий (0/1)
- `events_per_page`: Событий на страницу
- `engagement_score`: Общий скор вовлечения

#### Признаки повторных посещений
- `is_returning`: Повторный посетитель (0/1)
- `is_frequent`: Частый посетитель >=3 посещений (0/1)

#### Географические признаки
- `is_moscow`: Москва (0/1)
- `is_spb`: Санкт-Петербург (0/1)
- `is_million_plus`: Город-миллионник (0/1)
- `is_regional_center`: Региональный центр (0/1)
- `city_conversion_rate`: Конверсия города (%)
- `city_avg_duration`: Средняя длительность сессии в городе
- `city_avg_hits`: Среднее количество действий в городе
- `city_tier_low`: Низкий tier города (0/1)
- `city_tier_medium`: Средний tier города (0/1)
- `city_tier_high`: Высокий tier города (0/1)
- `city_tier_very_high`: Очень высокий tier города (0/1)

**Возвращает:**
- `DataFrame`: Датасет с признаками

### `prepare_features(df)`

Подготавливает признаки для обучения модели.

```python
X, y = model.prepare_features(df)
```

**Параметры:**
- `df` (DataFrame): Датасет с признаками

**Операции:**
1. Выбирает нужные признаки
2. Заполняет пропуски нулями
3. Создает целевую переменную

**Возвращает:**
- `X` (DataFrame): Матрица признаков
- `y` (Series): Целевая переменная

### `optimize_hyperparameters(X, y)`

Оптимизирует гиперпараметры модели с помощью Grid Search.

```python
best_model = model.optimize_hyperparameters(X, y)
```

**Параметры:**
- `X` (DataFrame): Признаки
- `y` (Series): Целевая переменная

**Гиперпараметры для оптимизации:**
```python
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [8, 10, 12],
    'min_samples_split': [30, 50, 70],
    'min_samples_leaf': [15, 20, 25]
}
```

**Метод оптимизации:**
- Grid Search с кросс-валидацией (3-fold)
- Метрика: ROC-AUC
- Параллельное выполнение

**Возвращает:**
- `RandomForestClassifier`: Оптимизированная модель

### `train_model(X, y)`

Обучает модель и оценивает её качество.

```python
roc_auc = model.train_model(X, y)
```

**Параметры:**
- `X` (DataFrame): Признаки
- `y` (Series): Целевая переменная

**Процесс обучения:**
1. Разделение данных (80% обучение, 20% тест)
2. Стратификация по целевой переменной
3. Оптимизация гиперпараметров
4. Обучение модели
5. Оценка качества

**Метрики качества:**
- ROC-AUC
- Classification Report (precision, recall, f1-score)
- Важность признаков

**Возвращает:**
- `float`: ROC-AUC score

### `save_model(filename)`

Сохраняет модель в файл.

```python
model.save_model('../build/sber_auto_model.pkl')
```

**Параметры:**
- `filename` (str): Путь к файлу для сохранения

**Сохраненные данные:**
- Обученная модель
- Список признаков
- Список целевых действий

### `load_model(filename)`

Загружает модель из файла.

```python
model.load_model('sber_auto_model.pkl')
```

**Параметры:**
- `filename` (str): Путь к файлу модели

### `predict(data)`

Выполняет предсказание для одной сессии.

```python
result = model.predict({
    'visit_number': 1,
    'total_hits': 5,
    'unique_pages': 3,
    'session_duration': 120,
    'visit_hour': 14,
    'visit_weekday': 2,
    'is_weekend': 0,
    'is_mobile': 1,
    'is_android': 0,
    'is_ios': 1,
    'is_desktop': 0,
    'is_tablet': 0,
    'is_moscow': 1,
    'is_paid': 1,
    'avg_time_per_page': 24.0,
    'bounce_rate': 0,
    'deep_engagement': 0,
    'long_session': 0
})
```

**Параметры:**
- `data` (dict): Словарь с признаками сессии

**Обработка данных:**
- Автоматическое добавление недостающих признаков
- Заполнение пропусков нулями
- Проверка порядка признаков

**Возвращает:**
```python
{
    'prediction': 1,                    # 0 или 1
    'probability': 0.75,                # Вероятность конверсии
    'will_convert': True,               # Будет ли конверсия
    'conversion_probability': '75.00%', # Вероятность в процентах
    'confidence_level': 'высокая'       # Уровень уверенности
}
```

**Уровни уверенности:**
- `высокая`: probability > 0.7
- `средняя`: 0.3 < probability <= 0.7
- `низкая`: probability <= 0.3

### `predict_batch(data_list)`

Выполняет пакетное предсказание для нескольких сессий.

```python
results = model.predict_batch([
    {'visit_number': 1, 'total_hits': 5, ...},
    {'visit_number': 2, 'total_hits': 10, ...}
])
```

**Параметры:**
- `data_list` (list): Список словарей с признаками сессий

**Обработка ошибок:**
- Каждая сессия обрабатывается отдельно
- Ошибки не прерывают обработку
- Результаты содержат информацию об ошибках

**Возвращает:**
- `list`: Список результатов предсказаний

## Функция train_and_save_model()

Объединяет весь процесс обучения и сохранения модели.

```python
from sber_auto_model import train_and_save_model

model = train_and_save_model()
```

**Процесс:**
1. Создание экземпляра модели
2. Загрузка данных
3. Определение целевых действий
4. Создание признаков
5. Подготовка данных
6. Обучение модели
7. Сохранение модели
8. Тестирование

**Вывод:**
- Информация о процессе обучения
- Метрики качества модели
- Сравнение с целевыми показателями

## Примеры использования

### Полный цикл обучения

```python
from sber_auto_model import SberAutoModel

# Создание модели
model = SberAutoModel()

# Загрузка данных
sessions, hits = model.load_data()

# Определение целевых действий
model.define_target_actions(hits)

# Создание признаков
df = model.create_features(sessions, hits)

# Подготовка данных
X, y = model.prepare_features(df)

# Обучение модели
roc_auc = model.train_model(X, y)

# Сохранение модели
model.save_model('../build/sber_auto_model.pkl')

print(f"ROC-AUC: {roc_auc:.4f}")
```

### Использование обученной модели

```python
from sber_auto_model import SberAutoModel

# Создание модели
model = SberAutoModel()

# Загрузка обученной модели
model.load_model('sber_auto_model.pkl')

# Предсказание
data = {
    'visit_number': 1,
    'total_hits': 5,
    'unique_pages': 3,
    'session_duration': 120,
    'visit_hour': 14,
    'is_mobile': 1,
    'is_ios': 1
}

result = model.predict(data)
print(f"Конверсия: {result['conversion_probability']}")
print(f"Будет ли конверсия: {result['will_convert']}")
```

### Пакетное предсказание

```python
# Список сессий для предсказания
sessions_data = [
    {
        'visit_number': 1,
        'total_hits': 5,
        'unique_pages': 3,
        'session_duration': 120,
        'visit_hour': 14,
        'is_mobile': 1,
        'is_ios': 1
    },
    {
        'visit_number': 2,
        'total_hits': 10,
        'unique_pages': 7,
        'session_duration': 300,
        'visit_hour': 16,
        'is_desktop': 1
    }
]

# Пакетное предсказание
results = model.predict_batch(sessions_data)

# Анализ результатов
for i, result in enumerate(results):
    print(f"Сессия {i}: {result['conversion_probability']} конверсия")
```

## Обработка ошибок

### Типичные ошибки и решения

#### 1. Ошибка загрузки данных
```python
FileNotFoundError: [Errno 2] No such file or directory: '../data/ga_sessions.pkl'
```
**Решение:** Убедитесь, что файлы данных находятся в папке `data/`

#### 2. Ошибка загрузки модели
```python
FileNotFoundError: [Errno 2] No such file or directory: 'sber_auto_model.pkl'
```
**Решение:** Сначала обучите и сохраните модель

#### 3. Ошибка предсказания
```python
ValueError: Модель не загружена. Сначала загрузите или обучите модель.
```
**Решение:** Загрузите модель перед выполнением предсказаний

#### 4. Ошибка признаков
```python
KeyError: 'feature_name'
```
**Решение:** Убедитесь, что все необходимые признаки присутствуют в данных

## Производительность

### Время выполнения
- **Загрузка данных**: ~2-3 секунды
- **Создание признаков**: ~10-15 секунд
- **Обучение модели**: ~2-3 минуты
- **Предсказание**: <0.1 секунды

### Память
- **Обучение**: ~4GB RAM
- **Предсказание**: ~100MB RAM

### Масштабируемость
- **Пакетное предсказание**: до 1000 сессий за запрос
- **Параллельная обработка**: Grid Search использует все ядра CPU

## Лучшие практики

### 1. Обработка данных
- Всегда проверяйте наличие файлов данных
- Используйте try-except для обработки ошибок
- Логируйте важные этапы процесса

### 2. Обучение модели
- Используйте стратификацию для несбалансированных данных
- Оптимизируйте гиперпараметры
- Сохраняйте модель после обучения

### 3. Предсказания
- Проверяйте наличие всех необходимых признаков
- Обрабатывайте ошибки в пакетных предсказаниях
- Валидируйте входные данные

### 4. Мониторинг
- Отслеживайте качество предсказаний
- Логируйте время выполнения
- Мониторьте использование памяти 