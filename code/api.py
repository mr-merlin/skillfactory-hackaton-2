import logging
import os
import sys
import time

from flask import Flask, jsonify, request

# Добавляем путь к модулям и импортируем
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # noqa: E402
from sber_auto_model import SberAutoModel  # noqa: E402

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Глобальная переменная для модели
model = None


def load_model():
    """Загрузка модели при запуске"""
    global model
    try:
        model = SberAutoModel()
        model.load_model("../build/sber_auto_model.pkl")
        logger.info("✅ Модель успешно загружена")
        return True
    except Exception as e:
        logger.error(f"❌ Ошибка загрузки модели: {e}")
        return False


@app.route("/health", methods=["GET"])
def health_check():
    """Проверка здоровья API"""
    return jsonify(
        {
            "status": "healthy",
            "model_loaded": model is not None,
            "timestamp": time.time(),
        }
    )


@app.route("/predict", methods=["POST"])
def predict():
    """
    Предсказание для одной сессии

    Пример запроса:
    {
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
    }
    """
    start_time = time.time()

    if model is None:
        return jsonify({"error": "Модель не загружена"}), 500

    try:
        # Получаем данные из запроса
        data = request.get_json()

        if not data:
            return jsonify({"error": "Данные не предоставлены"}), 400

        # Выполняем предсказание
        result = model.predict(data)

        # Добавляем время выполнения
        result["execution_time"] = round(time.time() - start_time, 3)
        result["status"] = "success"

        logger.info(f"✅ Предсказание выполнено за {result['execution_time']}с")
        logger.info(
            f"📊 Результат: {result['conversion_probability']} "
            f"(уверенность: {result['confidence_level']})"
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"❌ Ошибка предсказания: {e}")
        return (
            jsonify(
                {
                    "error": str(e),
                    "execution_time": round(time.time() - start_time, 3),
                    "status": "error",
                }
            ),
            500,
        )


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    """
    Пакетное предсказание для нескольких сессий

    Пример запроса:
    {
        "sessions": [
            {
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
            {
                "visit_number": 2,
                "total_hits": 10,
                "unique_pages": 7,
                "session_duration": 300,
                "visit_hour": 16,
                "visit_weekday": 3,
                "is_weekend": 0,
                "is_mobile": 0,
                "is_android": 0,
                "is_ios": 0,
                "is_desktop": 1,
                "is_tablet": 0,
                "is_moscow": 1,
                "is_paid": 0,
                "avg_time_per_page": 30.0,
                "bounce_rate": 0,
                "deep_engagement": 1,
                "long_session": 1
            }
        ]
    }
    """
    start_time = time.time()

    if model is None:
        return jsonify({"error": "Модель не загружена"}), 500

    try:
        # Получаем данные из запроса
        request_data = request.get_json()

        if not request_data or "sessions" not in request_data:
            return jsonify({"error": "Данные сессий не предоставлены"}), 400

        sessions = request_data["sessions"]

        if not isinstance(sessions, list):
            return jsonify({"error": "sessions должен быть списком"}), 400

        if len(sessions) > 1000:
            return jsonify({"error": "Максимальное количество сессий: 1000"}), 400

        # Выполняем пакетное предсказание
        results = model.predict_batch(sessions)

        # Добавляем метаданные
        response = {
            "predictions": results,
            "total_sessions": len(sessions),
            "execution_time": round(time.time() - start_time, 3),
            "status": "success",
        }

        # Статистика результатов
        successful_predictions = [r for r in results if "error" not in r]
        if successful_predictions:
            avg_probability = sum(r["probability"] for r in successful_predictions) / len(
                successful_predictions
            )
            high_confidence_count = sum(
                1 for r in successful_predictions if r["confidence_level"] == "высокая"
            )

            response["statistics"] = {
                "successful_predictions": len(successful_predictions),
                "failed_predictions": len(results) - len(successful_predictions),
                "average_probability": round(avg_probability * 100, 2),
                "high_confidence_predictions": high_confidence_count,
                "high_confidence_percentage": (
                    round(high_confidence_count / len(successful_predictions) * 100, 2)
                    if successful_predictions
                    else 0
                ),
            }

        logger.info(
            f"✅ Пакетное предсказание для {len(sessions)} сессий "
            f"за {response['execution_time']}с"
        )

        return jsonify(response)

    except Exception as e:
        logger.error(f"❌ Ошибка пакетного предсказания: {e}")
        return (
            jsonify(
                {
                    "error": str(e),
                    "execution_time": round(time.time() - start_time, 3),
                    "status": "error",
                }
            ),
            500,
        )


@app.route("/model_info", methods=["GET"])
def model_info():
    """Информация о модели"""
    if model is None:
        return jsonify({"error": "Модель не загружена"}), 500

    return jsonify(
        {
            "feature_count": len(model.feature_names) if model.feature_names else 0,
            "target_actions_count": len(model.target_actions) if model.target_actions else 0,
            "feature_names": model.feature_names[:10] if model.feature_names else [],
            "target_actions": model.target_actions[:5] if model.target_actions else [],
            "status": "loaded",
        }
    )


@app.route("/example", methods=["GET"])
def get_example():
    """Пример данных для предсказания"""
    return jsonify(
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
                "long_session": 0,
            },
            "description": "Пример данных для предсказания конверсии",
        }
    )


@app.route("/features", methods=["GET"])
def get_features():
    """Список всех признаков модели"""
    if model is None:
        return jsonify({"error": "Модель не загружена"}), 500

    return jsonify(
        {
            "features": model.feature_names,
            "feature_count": len(model.feature_names),
            "feature_categories": {
                "temporal": [
                    f
                    for f in model.feature_names
                    if any(
                        x in f for x in ["hour", "week", "morning", "afternoon", "evening", "night"]
                    )
                ],
                "device": [
                    f
                    for f in model.feature_names
                    if any(
                        x in f
                        for x in [
                            "mobile",
                            "android",
                            "ios",
                            "desktop",
                            "tablet",
                            "windows",
                            "macos",
                        ]
                    )
                ],
                "geographic": [
                    f
                    for f in model.feature_names
                    if any(x in f for x in ["moscow", "spb", "city", "regional"])
                ],
                "behavioral": [
                    f
                    for f in model.feature_names
                    if any(x in f for x in ["hits", "pages", "duration", "engagement", "activity"])
                ],
                "traffic": [
                    f
                    for f in model.feature_names
                    if any(x in f for x in ["paid", "organic", "referral", "direct"])
                ],
            },
        }
    )


@app.route("/stats", methods=["GET"])
def get_stats():
    """Статистика использования API"""
    return jsonify(
        {
            "uptime": time.time(),
            "endpoints": [
                "GET /health - проверка здоровья",
                "POST /predict - предсказание для одной сессии",
                "POST /predict_batch - пакетное предсказание",
                "GET /model_info - информация о модели",
                "GET /example - пример данных",
                "GET /features - список признаков",
                "GET /stats - статистика API",
            ],
        }
    )


if __name__ == "__main__":
    print("🚀 API сервер запускается...")

    # Загружаем модель
    if load_model():
        print("📊 Доступные эндпоинты:")
        print("   GET  /health - проверка здоровья")
        print("   POST /predict - предсказание для одной сессии")
        print("   POST /predict_batch - пакетное предсказание")
        print("   GET  /model_info - информация о модели")
        print("   GET  /example - пример данных")
        print("   GET  /features - список признаков")
        print("   GET  /stats - статистика API")

        print("🌐 Сервер доступен по адресу: http://localhost:5001")
        print(f"🔧 Количество признаков: {len(model.feature_names)}")

        # Запускаем сервер
        app.run(host="0.0.0.0", port=5001, debug=False)
    else:
        print("❌ Не удалось загрузить модель. " "Проверьте наличие файла sber_auto_model.pkl")
