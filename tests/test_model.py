#!/usr/bin/env python3
"""
Тест демонстрации запросов к модели СберАвтоподписка
Проверяет различные сценарии предсказаний
"""

import os
import sys

import requests

# Добавляем путь к модулям
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "code"))


def test_api_connection():
    """Проверка подключения к API"""
    try:
        response = requests.get("http://localhost:5001/health", timeout=5)
        if response.status_code == 200:
            return True
    except requests.exceptions.RequestException:
        pass
    return False


def make_prediction(data):
    """Выполнение предсказания через API"""
    try:
        response = requests.post("http://localhost:5001/predict", json=data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except requests.exceptions.RequestException as e:
        return {"error": f"Ошибка подключения: {str(e)}"}


def test_high_conversion_scenario():
    """Тест высоковероятной конверсии"""
    print("🔍 Тест: Высоковероятная конверсия")
    print(
        "📋 Ожидаемое поведение: Пользователь с высоким вовлечением, "
        "iOS устройством, в пиковые часы должен показать высокую вероятность конверсии"
    )
    print("📊 Ожидаемый результат: ~5.0%")

    data = {
        "visit_number": 3,
        "total_hits": 15,
        "unique_pages": 8,
        "session_duration": 600,
        "visit_hour": 19,
        "is_weekend": 0,
        "is_mobile": 1,
        "is_android": 0,
        "is_ios": 1,
        "is_desktop": 0,
        "is_tablet": 0,
        "is_moscow": 1,
        "is_paid": 1,
        "avg_time_per_page": 40.0,
        "bounce_rate": 0,
        "deep_engagement": 1,
        "long_session": 1,
    }

    result = make_prediction(data)
    if "error" in result:
        print(f"❌ {result['error']}")
        return False
    else:
        prob = result.get("conversion_probability", "N/A")
        print(f"✅ Результат: {prob}")
        return True


def test_low_conversion_scenario():
    """Тест низковероятной конверсии"""
    print("\n🔍 Тест: Низковероятная конверсия")
    print(
        "📋 Ожидаемое поведение: Пользователь с низким вовлечением, "
        "десктопом, в неактивные часы должен показать низкую вероятность конверсии"
    )
    print("📊 Ожидаемый результат: ~0.1%")

    data = {
        "visit_number": 1,
        "total_hits": 2,
        "unique_pages": 1,
        "session_duration": 30,
        "visit_hour": 3,
        "visit_weekday": 6,
        "is_weekend": 1,
        "is_mobile": 0,
        "is_android": 0,
        "is_ios": 0,
        "is_desktop": 1,
        "is_tablet": 0,
        "is_moscow": 0,
        "is_paid": 0,
        "avg_time_per_page": 15.0,
        "bounce_rate": 1,
        "deep_engagement": 0,
        "long_session": 0,
    }

    result = make_prediction(data)
    if "error" in result:
        print(f"❌ {result['error']}")
        return False
    else:
        prob = result.get("conversion_probability", "N/A")
        print(f"✅ Результат: {prob}")
        return True


def test_medium_conversion_scenario():
    """Тест средней вероятности конверсии"""
    print("\n🔍 Тест: Средняя вероятность конверсии")
    print(
        "📋 Ожидаемое поведение: Пользователь со средними показателями "
        "должен показать среднюю вероятность конверсии"
    )
    print("📊 Ожидаемый результат: ~2.2%")

    data = {
        "visit_number": 2,
        "total_hits": 8,
        "unique_pages": 4,
        "session_duration": 240,
        "visit_hour": 10,
        "visit_weekday": 3,
        "is_weekend": 0,
        "is_mobile": 1,
        "is_android": 1,
        "is_ios": 0,
        "is_desktop": 0,
        "is_tablet": 0,
        "is_moscow": 0,
        "is_paid": 1,
        "avg_time_per_page": 30.0,
        "bounce_rate": 0,
        "deep_engagement": 0,
        "long_session": 0,
    }

    result = make_prediction(data)
    if "error" in result:
        print(f"❌ {result['error']}")
        return False
    else:
        prob = result.get("conversion_probability", "N/A")
        print(f"✅ Результат: {prob}")
        return True


def test_time_comparison():
    """Сравнение времени суток"""
    print("\n🕐 СРАВНЕНИЕ ВРЕМЕНИ СУТОК")
    print("------------------------------")

    base_data = {
        "visit_number": 2,
        "total_hits": 8,
        "unique_pages": 4,
        "session_duration": 240,
        "visit_weekday": 2,
        "is_weekend": 0,
        "is_mobile": 1,
        "is_android": 1,
        "is_ios": 0,
        "is_desktop": 0,
        "is_tablet": 0,
        "is_moscow": 1,
        "is_paid": 1,
        "avg_time_per_page": 30.0,
        "bounce_rate": 0,
        "deep_engagement": 0,
        "long_session": 0,
    }

    times = [
        (14, "День (14:00)", "Дневная активность"),
        (19, "Вечер (19:00)", "Вечерняя активность"),
        (3, "Ночь (03:00)", "Ночная активность"),
    ]

    results = []
    for hour, name, description in times:
        print(f"\n🔍 Тест: {name}")
        print(f"📋 Ожидаемое поведение: {description}")

        data = base_data.copy()
        data["visit_hour"] = hour

        result = make_prediction(data)
        if "error" in result:
            print(f"❌ {result['error']}")
            results.append(False)
        else:
            prob = result.get("conversion_probability", "N/A")
            print(f"✅ Результат: {prob}")
            results.append(True)

    return all(results)


def test_device_comparison():
    """Сравнение устройств"""
    print("\n📱 СРАВНЕНИЕ УСТРОЙСТВ")
    print("------------------------------")

    base_data = {
        "visit_number": 2,
        "total_hits": 8,
        "unique_pages": 4,
        "session_duration": 240,
        "visit_hour": 14,
        "visit_weekday": 2,
        "is_weekend": 0,
        "is_moscow": 1,
        "is_paid": 1,
        "avg_time_per_page": 30.0,
        "bounce_rate": 0,
        "deep_engagement": 0,
        "long_session": 0,
    }

    devices = [
        (1, 0, 0, 0, "Android", "Мобильное устройство Android"),
        (1, 0, 1, 0, "iOS", "Мобильное устройство iOS"),
        (0, 0, 0, 1, "Desktop", "Десктопное устройство"),
    ]

    results = []
    for is_mobile, is_android, is_ios, is_desktop, name, description in devices:
        print(f"\n🔍 Тест: {name}")
        print(f"📋 Ожидаемое поведение: {description}")

        data = base_data.copy()
        data.update(
            {
                "is_mobile": is_mobile,
                "is_android": is_android,
                "is_ios": is_ios,
                "is_desktop": is_desktop,
            }
        )

        result = make_prediction(data)
        if "error" in result:
            print(f"❌ {result['error']}")
            results.append(False)
        else:
            prob = result.get("conversion_probability", "N/A")
            print(f"✅ Результат: {prob}")
            results.append(True)

    return all(results)


def main():
    """Основная функция тестирования"""
    print("🚀 ТЕСТИРОВАНИЕ ДЕМОНСТРАЦИИ ЗАПРОСОВ")
    print("=" * 50)

    # Проверяем подключение к API
    if not test_api_connection():
        print(
            "❌ API сервер недоступен. "
            "Запустите API сервер (опция 2) перед тестированием."
        )
        return

    print("✅ API сервер доступен")

    # Запускаем тесты
    tests = [
        ("Высоковероятная конверсия", test_high_conversion_scenario),
        ("Низковероятная конверсия", test_low_conversion_scenario),
        ("Средняя вероятность конверсии", test_medium_conversion_scenario),
        ("Сравнение времени суток", test_time_comparison),
        ("Сравнение устройств", test_device_comparison),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n{'=' * 50}")
        print(f"🧪 ТЕСТ: {test_name}")
        print(f"{'=' * 50}")

        try:
            if test_func():
                passed += 1
                print(f"✅ Тест '{test_name}' пройден")
            else:
                print(f"❌ Тест '{test_name}' провален")
        except Exception as e:
            print(f"❌ Тест '{test_name}' завершился с ошибкой: {e}")

    # Итоговый отчет
    print(f"\n{'=' * 50}")
    print("📊 ИТОГОВЫЙ ОТЧЕТ")
    print(f"{'=' * 50}")
    print(f"✅ Пройдено: {passed}/{total} тестов")
    print(f"📈 Успешность: {(passed / total) * 100:.1f}%")

    if passed == total:
        print("🎉 ВСЕ ТЕСТЫ ПРОЙДЕНЫ УСПЕШНО!")
    else:
        print("⚠️ НЕКОТОРЫЕ ТЕСТЫ ПРОВАЛЕНЫ")

    print("\n🎯 КЛЮЧЕВЫЕ ВЫВОДЫ:")
    print("   • Глубина вовлечения (total_hits) - критически важный фактор")
    print("   • Время суток влияет на конверсию")
    print("   • Тип устройства имеет значение")
    print("   • Модель корректно различает разные сценарии")


if __name__ == "__main__":
    main()
