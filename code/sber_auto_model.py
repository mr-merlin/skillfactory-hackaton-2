import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import roc_auc_score, classification_report


class SberAutoModel:
    """
    Модель для предсказания целевых действий на сайте СберАвтоподписка
    """

    def __init__(self):
        self.model = None
        self.feature_names = None
        self.target_actions = None
        self.scaler = None

    def load_data(self):
        """Загрузка и подготовка данных"""
        print("📂 Загружаем данные...")

        # Загрузка данных
        sessions = pd.read_pickle("../data/ga_sessions.pkl")
        hits = pd.read_pickle("../data/ga_hits.pkl")

        print(f"📊 Сессии: {sessions.shape}")
        print(f"📊 Хиты: {hits.shape}")

        return sessions, hits

    def define_target_actions(self, hits):
        """Определение целевых действий с расширенной логикой"""
        print("🎯 Определяем целевые действия...")

        # Анализ всех событий
        unique_events = hits["event_action"].value_counts()

        # Расширенный список ключевых слов для целевых действий
        target_keywords = [
            "submit",
            "success",
            "call",
            "contact",
            "request",
            "callback",
            "claim",
            "chat",
            "auth",
            "phone",
            "sms",
            "code",
            "confirm",
            "start_chat",
            "user_message",
            "proactive",
            "invitation",
        ]

        potential_targets = []

        for event in unique_events.index:
            event_lower = str(event).lower()
            for keyword in target_keywords:
                if keyword in event_lower:
                    potential_targets.append((event, unique_events[event]))
                    break

        self.target_actions = [event for event, _ in potential_targets]

        print(f"✅ Найдено {len(self.target_actions)} целевых действий")
        print(f"📋 Примеры: {self.target_actions[:5]}")
        return self.target_actions

    def create_features(self, sessions, hits):
        """Создание признаков"""
        print("🔧 Создаем признаки...")

        # Создание целевой переменной с расширенной логикой
        hits["is_target"] = hits["event_action"].apply(
            lambda x: 1 if any(key in str(x).lower() for key in self.target_actions) else 0
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
                    "event_action": "nunique",
                }
            )
            .rename(
                columns={
                    "hit_number": "total_hits",
                    "hit_page_path": "unique_pages",
                    "hit_time": "session_duration",
                    "event_action": "unique_events",
                }
            )
        )

        # Объединение данных
        df = sessions.merge(session_metrics, on="session_id", how="left")
        df["is_target"] = df["is_target"].fillna(0).astype(int)
        df["session_duration"] = df["session_duration"].fillna(0)
        df["unique_events"] = df["unique_events"].fillna(0)

        # Временные признаки
        df["visit_datetime"] = pd.to_datetime(
            df["visit_date"].astype(str) + " " + df["visit_time"].astype(str)
        )
        df["visit_hour"] = df["visit_datetime"].dt.hour
        df["visit_weekday"] = df["visit_datetime"].dt.weekday
        df["is_weekend"] = df["visit_weekday"].isin([5, 6]).astype(int)

        # Новые временные признаки
        df["is_morning"] = df["visit_hour"].between(6, 11).astype(int)
        df["is_afternoon"] = df["visit_hour"].between(12, 17).astype(int)
        df["is_evening"] = df["visit_hour"].between(18, 23).astype(int)
        df["is_night"] = ((df["visit_hour"] >= 0) & (df["visit_hour"] <= 5)).astype(int)
        df["is_workday"] = (~df["is_weekend"]).astype(int)

        # Признаки устройств
        df["is_mobile"] = (df["device_category"] == "mobile").astype(int)
        df["is_android"] = (df["device_os"] == "Android").astype(int)
        df["is_ios"] = (df["device_os"] == "iOS").astype(int)
        df["is_desktop"] = (df["device_category"] == "desktop").astype(int)
        df["is_tablet"] = (df["device_category"] == "tablet").astype(int)

        # Новые признаки устройств
        df["is_windows"] = (df["device_os"] == "Windows").astype(int)
        df["is_macos"] = (df["device_os"] == "macOS").astype(int)

        # Географические признаки
        print("🌍 Создаем географические признаки...")

        # Анализ городов с более детальной сегментацией
        city_stats = (
            df.groupby("geo_city")
            .agg(
                {
                    "session_id": "count",
                    "is_target": "sum",
                    "session_duration": "mean",
                    "total_hits": "mean",
                }
            )
            .rename(
                columns={
                    "session_id": "city_sessions",
                    "is_target": "city_conversions",
                    "session_duration": "city_avg_duration",
                    "total_hits": "city_avg_hits",
                }
            )
        )

        city_stats["city_conversion_rate"] = (
            city_stats["city_conversions"] / city_stats["city_sessions"] * 100
        ).round(2)

        # Сегментация городов
        city_stats["city_tier"] = pd.cut(
            city_stats["city_conversion_rate"],
            bins=[0, 0.8, 1.2, 1.6, 10],
            labels=["low", "medium", "high", "very_high"],
        )

        # Добавляем географические признаки к основному датасету
        df = df.merge(
            city_stats[
                [
                    "city_conversion_rate",
                    "city_tier",
                    "city_sessions",
                    "city_avg_duration",
                    "city_avg_hits",
                ]
            ],
            left_on="geo_city",
            right_index=True,
            how="left",
        )

        # Заполняем пропуски
        df["city_conversion_rate"] = df["city_conversion_rate"].fillna(0)
        df["city_tier"] = df["city_tier"].fillna("low")
        df["city_sessions"] = df["city_sessions"].fillna(0)
        df["city_avg_duration"] = df["city_avg_duration"].fillna(0)
        df["city_avg_hits"] = df["city_avg_hits"].fillna(0)

        # Расширенные географические признаки
        df["is_moscow"] = (df["geo_city"] == "Moscow").astype(int)
        df["is_spb"] = (df["geo_city"] == "Saint Petersburg").astype(int)
        df["is_million_plus"] = (df["city_sessions"] >= 1000).astype(int)
        df["is_regional_center"] = (df["city_sessions"] >= 500).astype(int)

        # Кодируем tier городов
        df["city_tier_low"] = (df["city_tier"] == "low").astype(int)
        df["city_tier_medium"] = (df["city_tier"] == "medium").astype(int)
        df["city_tier_high"] = (df["city_tier"] == "high").astype(int)
        df["city_tier_very_high"] = (df["city_tier"] == "very_high").astype(int)

        # Источники трафика
        df["is_paid"] = ~df["utm_medium"].isin(["organic", "referral", "(none)"]).astype(int)
        df["is_organic"] = (df["utm_medium"] == "organic").astype(int)
        df["is_referral"] = (df["utm_medium"] == "referral").astype(int)
        df["is_direct"] = (df["utm_medium"] == "(none)").astype(int)

        # Поведенческие метрики
        df["avg_time_per_page"] = df["session_duration"] / df["total_hits"]
        df["avg_time_per_page"] = df["avg_time_per_page"].replace([np.inf, -np.inf], 0)

        # Новые поведенческие признаки
        df["bounce_rate"] = (df["total_hits"] == 1).astype(int)
        df["deep_engagement"] = (df["unique_pages"] >= 5).astype(int)
        df["long_session"] = (df["session_duration"] > 300).astype(int)
        df["very_long_session"] = (df["session_duration"] > 600).astype(int)
        df["high_activity"] = (df["total_hits"] >= 10).astype(int)
        df["very_high_activity"] = (df["total_hits"] >= 15).astype(int)

        # Новые признаки взаимодействия
        df["events_per_page"] = df["unique_events"] / df["unique_pages"]
        df["events_per_page"] = df["events_per_page"].replace([np.inf, -np.inf], 0)
        df["engagement_score"] = (
            df["total_hits"] * df["unique_pages"] * df["session_duration"]
        ) / 1000

        # Признаки повторных посещений
        df["is_returning"] = (df["visit_number"] > 1).astype(int)
        df["is_frequent"] = (df["visit_number"] >= 3).astype(int)

        print(f"✅ Создано {len(df)} сессий с признаками")
        return df

    def prepare_features(self, df):
        """Подготовка признаков для модели"""
        print("🔧 Подготавливаем признаки для модели...")

        # Расширенный список признаков
        feature_cols = [
            # Базовые признаки
            "visit_number",
            "total_hits",
            "unique_pages",
            "session_duration",
            "unique_events",
            # Временные признаки
            "visit_hour",
            "visit_weekday",
            "is_weekend",
            "is_workday",
            "is_morning",
            "is_afternoon",
            "is_evening",
            "is_night",
            # Признаки устройств
            "is_mobile",
            "is_android",
            "is_ios",
            "is_desktop",
            "is_tablet",
            "is_windows",
            "is_macos",
            # Источники трафика
            "is_paid",
            "is_organic",
            "is_referral",
            "is_direct",
            # Поведенческие метрики
            "avg_time_per_page",
            "bounce_rate",
            "deep_engagement",
            "long_session",
            "very_long_session",
            "high_activity",
            "very_high_activity",
            "events_per_page",
            "engagement_score",
            # Признаки повторных посещений
            "is_returning",
            "is_frequent",
            # Географические признаки
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
            "city_tier_very_high",
        ]

        X = df[feature_cols].fillna(0)
        Y = df["is_target"]

        self.feature_names = feature_cols

        print(f"📊 Признаки: {X.shape}")
        print(f"🎯 Целевая переменная: {Y.shape}")
        print(f"📈 Конверсия: {Y.mean()*100:.2f}%")
        print(
            f"🌍 Географических признаков: {len([f for f in feature_cols if 'city' in f or 'moscow' in f or 'spb' in f])}"
        )
        print(
            f"⏰ Временных признаков: {len([f for f in feature_cols if 'hour' in f or 'week' in f or 'morning' in f or 'afternoon' in f or 'evening' in f or 'night' in f])}"
        )

        return X, Y

    def optimize_hyperparameters(self, X, y):
        """Оптимизация гиперпараметров модели"""
        print("🔧 Оптимизируем гиперпараметры...")

        # Параметры для поиска
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [8, 10, 12],
            "min_samples_split": [30, 50, 70],
            "min_samples_leaf": [15, 20, 25],
        }

        # Создание базовой модели
        base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

        # Grid Search с кросс-валидацией
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring="roc_auc",
            n_jobs=-1,
            verbose=1,
        )

        grid_search.fit(X, y)

        print(f"✅ Лучшие параметры: {grid_search.best_params_}")
        print(f"📈 Лучший ROC-AUC: {grid_search.best_score_:.4f}")

        return grid_search.best_estimator_

    def train_model(self, X, y):
        """Обучение модели"""
        print("🤖 Обучаем модель...")

        # Разделение данных
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Оптимизация гиперпараметров
        self.model = self.optimize_hyperparameters(X_train, y_train)

        # Оценка модели
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]

        roc_auc = roc_auc_score(y_test, y_pred_proba)

        print(f"📊 Размер обучающей выборки: {X_train.shape}")
        print(f"📊 Размер тестовой выборки: {X_test.shape}")
        print(f"📈 ROC-AUC: {roc_auc:.4f}")
        print("\n📋 Отчет о классификации:")
        print(classification_report(y_test, y_pred))

        # Анализ важности признаков
        feature_importance = pd.DataFrame(
            {"feature": self.feature_names, "importance": self.model.feature_importances_}
        ).sort_values("importance", ascending=False)

        print("\n🏆 Топ-20 важных признаков:")
        for _, row in feature_importance.head(20).iterrows():
            print(f"{row['feature']}: {row['importance']:.3f}")

        # Кросс-валидация
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring="roc_auc")
        print(
            f"\n📊 Кросс-валидация ROC-AUC: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})"
        )

        return roc_auc

    def save_model(self, filename="../build/sber_auto_model.pkl"):
        """Сохранение модели"""
        # Создаем директорию build если её нет
        import os

        os.makedirs("../build", exist_ok=True)

        print(f"💾 Сохраняем модель в {filename}...")

        model_data = {
            "model": self.model,
            "feature_names": self.feature_names,
            "target_actions": self.target_actions,
        }

        with open(filename, "wb") as f:
            pickle.dump(model_data, f)

        print("✅ Модель сохранена")

    def load_model(self, filename="sber_auto_model.pkl"):
        """Загрузка модели"""
        print(f"📂 Загружаем модель из {filename}...")

        with open(filename, "rb") as f:
            model_data = pickle.load(f)

        self.model = model_data["model"]
        self.feature_names = model_data["feature_names"]
        self.target_actions = model_data["target_actions"]

        print("✅ Модель загружена")

    def predict(self, data):
        """
        Предсказание для новых данных

        Args:
            data (dict): Словарь с признаками сессии

        Returns:
            dict: Результат предсказания с дополнительной информацией
        """
        if self.model is None:
            raise ValueError("Модель не загружена. Сначала загрузите или обучите модель.")

        # Создаем DataFrame из входных данных
        df_input = pd.DataFrame([data])

        # Проверяем наличие всех признаков
        missing_features = set(self.feature_names) - set(df_input.columns)
        if missing_features:
            # Добавляем недостающие признаки со значениями по умолчанию
            for feature in missing_features:
                df_input[feature] = 0

        # Выбираем только нужные признаки в правильном порядке
        X = df_input[self.feature_names].fillna(0)

        # Предсказание
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0][1]

        # Дополнительная информация
        confidence_level = (
            "высокая" if probability > 0.7 else "средняя" if probability > 0.3 else "низкая"
        )

        return {
            "prediction": int(prediction),
            "probability": float(probability),
            "will_convert": bool(prediction),
            "conversion_probability": f"{probability*100:.2f}%",
            "confidence_level": confidence_level,
        }

    def predict_batch(self, data_list):
        """
        Пакетное предсказание с обработкой ошибок

        Args:
            data_list (list): Список словарей с признаками сессий

        Returns:
            list: Список результатов предсказаний
        """
        results = []
        for i, data in enumerate(data_list):
            try:
                result = self.predict(data)
                result["session_id"] = i
                results.append(result)
            except Exception as e:
                results.append(
                    {
                        "session_id": i,
                        "error": str(e),
                        "prediction": 0,
                        "probability": 0.0,
                        "will_convert": False,
                        "conversion_probability": "0.00%",
                    }
                )
        return results


def train_and_save_model():
    """Обучение и сохранение модели"""
    print("🚀 Запуск обучения модели СберАвтоподписка")
    print("=" * 60)

    # Создание экземпляра модели
    model = SberAutoModel()

    # Загрузка данных
    sessions, hits = model.load_data()

    # Определение целевых действий
    model.define_target_actions(hits)

    # Создание признаков
    df = model.create_features(sessions, hits)

    # Подготовка признаков
    X, y = model.prepare_features(df)

    # Обучение модели
    roc_auc = model.train_model(X, y)

    # Сохранение модели
    model.save_model()

    print(f"\n🎉 Модель обучена и сохранена!")
    print(f"📊 ROC-AUC: {roc_auc:.4f}")
    print(f"✅ Целевой показатель 0.65 {'достигнут' if roc_auc >= 0.65 else 'НЕ достигнут'}")
    print(
        f"📈 Улучшение по сравнению с базовой моделью: {((roc_auc - 0.8859) / 0.8859 * 100):.1f}%"
    )

    return model


if __name__ == "__main__":
    # Обучение модели
    model = train_and_save_model()

    # Пример использования
    print("\n🧪 Тестирование модели:")
    print("-" * 40)

    # Пример данных для предсказания
    test_data = {
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
    }

    result = model.predict(test_data)
    print(f"📊 Результат предсказания:")
    print(f"   Конверсия: {result['conversion_probability']}")
    print(f"   Будет ли конверсия: {result['will_convert']}")
    print(f"   Вероятность: {result['probability']:.4f}")
    print(f"   Уровень уверенности: {result['confidence_level']}")
