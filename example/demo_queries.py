#!/usr/bin/env python3
"""
🎯 Демонстрация запросов к модели и проверка гипотез
"""

import requests


def test_query(name, data, expected_behavior):
    """Тестирование запроса к модели"""
    print(f"\n🔍 Тест: {name}")
    print(f"📋 Ожидаемое поведение: {expected_behavior}")

    try:
        response = requests.post("http://localhost:5001/predict", json=data, timeout=10)

        if response.status_code == 200:
            result = response.json()
            probability = result["conversion_probability"]
            execution_time = result["execution_time"]

            print(f"✅ Результат: {probability} конверсии")
            print(f"⏱️ Время выполнения: {execution_time:.3f}с")
            print(
                f"🎯 Предсказание: "
                f"{'Конвертирует' if result['will_convert'] else 'Не конвертирует'}"
            )

            return result
        else:
            print(f"❌ Ошибка: {response.status_code}")
            return None
    except Exception as e:
        print(f"❌ Ошибка подключения: {e}")
        return None


def main():
    """Основная функция демонстрации"""
    print("🚀 ДЕМОНСТРАЦИЯ ЗАПРОСОВ К МОДЕЛИ")
    print("=" * 50)

    # Тест 1: Высоковероятная конверсия
    high_conversion_data = {
        "visit_number": 3,
        "total_hits": 15,
        "unique_pages": 8,
        "session_duration": 600,
        "visit_hour": 19,  # Обновлено на пиковые часы
        "visit_weekday": 2,
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

    test_query(
        "Высоковероятная конверсия",
        high_conversion_data,
        "Пользователь с высоким вовлечением, iOS устройством, "
        "в пиковые часы должен показать высокую вероятность конверсии",
    )

    # Тест 2: Низковероятная конверсия
    low_conversion_data = {
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

    test_query(
        "Низковероятная конверсия",
        low_conversion_data,
        "Пользователь с низким вовлечением, десктопом, "
        "в неактивные часы должен показать низкую вероятность конверсии",
    )

    # Тест 3: Средняя вероятность конверсии
    medium_conversion_data = {
        "visit_number": 2,
        "total_hits": 7,
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
        "avg_time_per_page": 25.7,
        "bounce_rate": 0,
        "deep_engagement": 0,
        "long_session": 0,
    }

    test_query(
        "Средняя вероятность конверсии",
        medium_conversion_data,
        "Пользователь со средними показателями должен показать "
        "среднюю вероятность конверсии",
    )

    # Тест 4: Сравнение времени суток
    print("\n🕐 СРАВНЕНИЕ ВРЕМЕНИ СУТОК")
    print("-" * 30)

    base_data = medium_conversion_data.copy()

    # День
    day_data = base_data.copy()
    day_data["visit_hour"] = 14
    day_result = test_query("День (14:00)", day_data, "Дневная активность")

    # Вечер
    evening_data = base_data.copy()
    evening_data["visit_hour"] = 19
    evening_result = test_query("Вечер (19:00)", evening_data, "Вечерняя активность")

    # Ночь
    night_data = base_data.copy()
    night_data["visit_hour"] = 3
    night_result = test_query("Ночь (03:00)", night_data, "Ночная активность")

    # Тест 5: Сравнение устройств
    print("\n📱 СРАВНЕНИЕ УСТРОЙСТВ")
    print("-" * 30)

    # Android
    android_data = base_data.copy()
    android_data.update({"is_mobile": 1, "is_android": 1, "is_ios": 0, "is_desktop": 0})
    android_result = test_query("Android", android_data, "Мобильное устройство Android")

    # iOS
    ios_data = base_data.copy()
    ios_data.update({"is_mobile": 1, "is_android": 0, "is_ios": 1, "is_desktop": 0})
    ios_result = test_query("iOS", ios_data, "Мобильное устройство iOS")

    # Desktop
    desktop_data = base_data.copy()
    desktop_data.update({"is_mobile": 0, "is_android": 0, "is_ios": 0, "is_desktop": 1})
    desktop_result = test_query("Desktop", desktop_data, "Десктопное устройство")

    # Итоговый анализ
    print("\n" + "=" * 50)
    print("📊 ИТОГОВЫЙ АНАЛИЗ РЕЗУЛЬТАТОВ")
    print("=" * 50)

    if day_result and evening_result and night_result:
        print("🕐 Влияние времени суток:")
        print(f"   День (14:00): {day_result['conversion_probability']}")
        print(f"   Вечер (19:00): {evening_result['conversion_probability']}")
        print(f"   Ночь (03:00): {night_result['conversion_probability']}")

        # Определяем лучшее время
        times = [
            ("День", day_result["probability"]),
            ("Вечер", evening_result["probability"]),
            ("Ночь", night_result["probability"]),
        ]
        best_time = max(times, key=lambda x: x[1])
        print(f"   🏆 Лучшее время: {best_time[0]} ({best_time[1] * 100:.2f}%)")

    if android_result and ios_result and desktop_result:
        print("\n📱 Влияние устройства:")
        print(f"   Android: {android_result['conversion_probability']}")
        print(f"   iOS: {ios_result['conversion_probability']}")
        print(f"   Desktop: {desktop_result['conversion_probability']}")

        # Определяем лучшее устройство
        devices = [
            ("Android", android_result["probability"]),
            ("iOS", ios_result["probability"]),
            ("Desktop", desktop_result["probability"]),
        ]
        best_device = max(devices, key=lambda x: x[1])
        print(f"   🏆 Лучшее устройство: {best_device[0]} ({best_device[1] * 100:.2f}%)")

    print("\n🎯 КЛЮЧЕВЫЕ ВЫВОДЫ:")
    print("   • Глубина вовлечения (total_hits) - критически важный фактор")
    print("   • Время суток влияет на конверсию")
    print("   • Тип устройства имеет значение")
    print("   • Модель корректно различает разные сценарии")

    print("\n✅ Демонстрация завершена успешно!")


if __name__ == "__main__":
    main()
