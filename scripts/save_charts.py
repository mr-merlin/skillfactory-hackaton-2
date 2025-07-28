#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")

# Настройка отображения
plt.style.use("default")
sns.set_palette("husl")
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 100)

# Настройка для сохранения графиков
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300


def main() -> None:
    print("📊 Загружаем данные...")
    sessions = pd.read_pickle("data/ga_sessions.pkl")
    hits = pd.read_pickle("data/ga_hits.pkl")

    print(f"📊 Сессии: {sessions.shape}")
    print(f"📊 Хиты: {hits.shape}")

    # Анализ событий
    unique_events = hits["event_action"].value_counts()
    target_keywords = [
        "заявка",
        "звонок",
        "оформление",
        "callback",
        "покупка",
        "order",
        "submit",
        "contact",
        "call",
        "chat",
        "auth",
    ]
    potential_targets = []

    for event in unique_events.index:
        event_lower = str(event).lower()
        for keyword in target_keywords:
            if keyword in event_lower:
                potential_targets.append((event, unique_events[event]))
                break

    target_actions = [event for event, _ in potential_targets]

    # Создание целевой переменной
    hits["is_target"] = hits["event_action"].apply(
        lambda x: 1 if any(key in str(x).lower() for key in target_actions) else 0
    )

    # Агрегация на уровне сессии
    session_metrics = (
        hits.groupby("session_id")
        .agg(
            {
                "is_target": "max",
                "hit_number": "count",
                "hit_page_path": "nunique",
                "hit_time": lambda x: max(x) - min(x) if len(x) > 1 else 0,
            }
        )
        .rename(
            columns={
                "hit_number": "total_hits",
                "hit_page_path": "unique_pages",
                "hit_time": "session_duration",
            }
        )
    )

    # Объединение с сессиями
    df = sessions.merge(session_metrics, on="session_id", how="left")
    df["is_target"] = df["is_target"].fillna(0).astype(int)
    df["session_duration"] = df["session_duration"].fillna(0)

    # Создание признаков
    try:
        date_str = df["visit_date"].astype(str)
        time_str = df["visit_time"].astype(str)
        datetime_str = date_str + " " + time_str
        df["visit_datetime"] = pd.to_datetime(datetime_str, errors="coerce")

        df["visit_hour"] = df["visit_datetime"].dt.hour
        df["visit_weekday"] = df["visit_datetime"].dt.weekday
        df["is_weekend"] = df["visit_weekday"].isin([5, 6]).astype(int)
    except Exception as e:
        print(f"⚠️ Ошибка временных признаков: {e}")
        df["visit_hour"] = 12
        df["visit_weekday"] = 0
        df["is_weekend"] = 0

    df["is_mobile"] = (df["device_category"] == "mobile").astype(int)
    df["is_android"] = (df["device_os"] == "Android").astype(int)
    df["is_ios"] = (df["device_os"] == "iOS").astype(int)
    df["is_desktop"] = (df["device_category"] == "desktop").astype(int)
    df["is_tablet"] = (df["device_category"] == "tablet").astype(int)
    df["is_moscow"] = (df["geo_city"] == "Moscow").astype(int)
    df["is_paid"] = (
        ~df["utm_medium"].isin(["organic", "referral", "(none)"]).astype(int)
    )
    df["avg_time_per_page"] = df["session_duration"] / df["total_hits"]
    df["avg_time_per_page"] = df["avg_time_per_page"].replace([np.inf, -np.inf], 0)
    df["bounce_rate"] = (df["total_hits"] == 1).astype(int)
    df["deep_engagement"] = (df["total_hits"] > 5).astype(int)
    df["long_session"] = (df["session_duration"] > 300).astype(int)

    print("📊 Создаем графики...")

    # 1. Распределение целевых действий
    plt.figure(figsize=(10, 6))
    sns.countplot(x="is_target", data=df)
    plt.title("Распределение целевых действий")
    plt.xlabel("Целевое действие")
    plt.ylabel("Количество сессий")
    plt.savefig(
        "charts/распределение целевых действий.png", bbox_inches="tight", dpi=300
    )
    plt.close()

    # 2. Конверсия по городам
    top_cities = df["geo_city"].value_counts().nlargest(10).index
    plt.figure(figsize=(14, 8))
    city_conversion = (
        df[df["geo_city"].isin(top_cities)]
        .groupby("geo_city")["is_target"]
        .agg(["mean", "count"])
        .reset_index()
    )
    city_conversion = city_conversion[city_conversion["count"] > 100]

    sns.barplot(data=city_conversion, x="geo_city", y="mean")
    plt.title("Конверсия по городам (%)")
    plt.xlabel("Город")
    plt.ylabel("Конверсия")
    plt.xticks(rotation=45)
    plt.savefig("charts/конверсия по городам.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 3. Конверсия по типам устройств
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 2, 1)
    device_conversion = (
        df.groupby("device_category")["is_target"].agg(["mean", "count"]).reset_index()
    )
    sns.barplot(data=device_conversion, x="device_category", y="mean")
    plt.title("Конверсия по типам устройств")
    plt.ylabel("Конверсия")

    plt.subplot(2, 2, 2)
    os_conversion = (
        df.groupby("device_os")["is_target"].agg(["mean", "count"]).reset_index()
    )
    top_os = os_conversion[os_conversion["count"] > 1000]
    sns.barplot(data=top_os, x="device_os", y="mean")
    plt.title("Конверсия по ОС")
    plt.ylabel("Конверсия")
    plt.xticks(rotation=45)

    plt.subplot(2, 2, 3)
    mobile_data = df[df["device_category"] == "mobile"]
    mobile_conversion = (
        mobile_data.groupby("device_os")["is_target"]
        .agg(["mean", "count"])
        .reset_index()
    )
    mobile_conversion = mobile_conversion[
        mobile_conversion["device_os"].isin(["Android", "iOS"])
    ]
    sns.barplot(data=mobile_conversion, x="device_os", y="mean")
    plt.title("Android vs iOS")
    plt.ylabel("Конверсия")

    plt.subplot(2, 2, 4)
    traffic_conversion = (
        df.groupby("utm_medium")["is_target"].agg(["mean", "count"]).reset_index()
    )
    top_traffic = traffic_conversion[traffic_conversion["count"] > 500]
    sns.barplot(data=top_traffic, x="utm_medium", y="mean")
    plt.title("Конверсия по источникам")
    plt.ylabel("Конверсия")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("charts/конверсия по типам устройств.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 4. Временные паттерны
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    hourly_conversion = (
        df.groupby("visit_hour")["is_target"].agg(["mean", "count"]).reset_index()
    )
    sns.barplot(data=hourly_conversion, x="visit_hour", y="mean")
    plt.title("Конверсия по часам суток")
    plt.xlabel("Час")
    plt.ylabel("Конверсия")

    plt.subplot(2, 2, 2)
    weekday_conversion = (
        df.groupby("visit_weekday")["is_target"].agg(["mean", "count"]).reset_index()
    )
    weekday_names = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    weekday_conversion["weekday_name"] = weekday_conversion["visit_weekday"].map(
        lambda x: weekday_names[x]
    )
    sns.barplot(data=weekday_conversion, x="weekday_name", y="mean")
    plt.title("Конверсия по дням недели")
    plt.xlabel("День недели")
    plt.ylabel("Конверсия")

    plt.subplot(2, 2, 3)
    session_data = df[df["session_duration"] < df["session_duration"].quantile(0.95)]
    sns.histplot(data=session_data, x="session_duration", hue="is_target", bins=30)
    plt.title("Распределение длительности сессий")
    plt.xlabel("Длительность (секунды)")
    plt.ylabel("Количество")

    plt.subplot(2, 2, 4)
    corr_features = [
        "is_target",
        "total_hits",
        "unique_pages",
        "session_duration",
        "is_mobile",
        "is_paid",
    ]
    correlation_matrix = df[corr_features].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", center=0)
    plt.title("Корреляционная матрица")

    plt.tight_layout()
    plt.savefig(
        "charts/корреляционная матрица числовых признаков.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close()

    # 5. Конверсия по часам суток (отдельный график)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=hourly_conversion, x="visit_hour", y="mean")
    plt.title("Конверсия по часам суток")
    plt.xlabel("Час")
    plt.ylabel("Конверсия")
    plt.savefig("charts/конверсия по часам суток.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 6. Длительность сессий (отдельный график)
    plt.figure(figsize=(12, 6))
    session_data = df[df["session_duration"] < df["session_duration"].quantile(0.95)]
    sns.histplot(data=session_data, x="session_duration", hue="is_target", bins=30)
    plt.title("Распределение длительности сессий")
    plt.xlabel("Длительность (секунды)")
    plt.ylabel("Количество")
    plt.savefig("charts/длительность сессий.png", bbox_inches="tight", dpi=300)
    plt.close()

    # 7. Конверсия по источникам трафика (отдельный график)
    plt.figure(figsize=(14, 8))
    traffic_conversion = (
        df.groupby("utm_medium")["is_target"].agg(["mean", "count"]).reset_index()
    )
    top_traffic = traffic_conversion[traffic_conversion["count"] > 500]
    sns.barplot(data=top_traffic, x="utm_medium", y="mean")
    plt.title("Конверсия по источникам трафика")
    plt.xlabel("Источник трафика")
    plt.ylabel("Конверсия")
    plt.xticks(rotation=45)
    plt.savefig(
        "charts/конверсия по источникам трафика.png", bbox_inches="tight", dpi=300
    )
    plt.close()

    print("✅ Все графики сохранены в папку charts/")


if __name__ == "__main__":
    main()
