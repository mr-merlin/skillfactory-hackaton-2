#!/bin/bash

echo "🚗 ЗАПУСК ПРОЕКТА СБЕРАВТОПОДПИСКА"
echo "=================================="

# Проверяем, существует ли виртуальное окружение
if [ ! -d "venv" ]; then
    echo "📦 Создание виртуального окружения..."
    python3 -m venv venv
fi

# Активируем виртуальное окружение
echo "🔧 Активация виртуального окружения..."
source venv/bin/activate

# Проверяем, установлены ли зависимости
if ! python -c "import pandas, numpy, matplotlib, seaborn, sklearn, flask" 2>/dev/null; then
    echo "📚 Установка зависимостей..."
    pip install -r requirements.txt
fi

# Проверяем наличие файлов данных
if [ ! -f "data/ga_sessions.pkl" ] || [ ! -f "data/ga_hits.pkl" ]; then
    echo "❌ Ошибка: Файлы данных не найдены в папке data/"
    echo "Убедитесь, что файлы ga_sessions.pkl и ga_hits.pkl находятся в папке data/"
    exit 1
fi

echo ""
echo "🎯 ВЫБЕРИТЕ ДЕЙСТВИЕ:"
echo "1. Обучение модели"
echo "2. Запуск API сервера"
echo "3. Полный анализ (Jupyter Notebook)"
echo "4. Тестирование модели"
echo "5. Полное тестирование проекта"
echo "6. Тестирование Jupyter Notebook"
echo "7. Демонстрация запросов"
echo "8. Выход"

read -p "Введите номер (1-8): " choice

case $choice in
    1)
        echo ""
        echo "🤖 ОБУЧЕНИЕ МОДЕЛИ..."
        echo "===================="
        cd code && python sber_auto_model.py && cd ..
        ;;
    2)
        echo ""
        echo "🌐 ЗАПУСК API СЕРВЕРА..."
        echo "======================="
        if [ ! -f "build/sber_auto_model.pkl" ]; then
            echo "⚠️ Модель не найдена. Сначала обучите модель (опция 1)"
            exit 1
        fi
        echo "🚀 Запуск API на http://localhost:5001"
        cd code && python api.py && cd ..
        ;;
    3)
        echo ""
        echo "📊 ЗАПУСК JUPYTER NOTEBOOK..."
        echo "============================="
        if ! python -c "import jupyter" 2>/dev/null; then
            echo "📚 Установка Jupyter..."
            pip install jupyter
        fi
        echo "📖 Запуск Jupyter Notebook..."
        jupyter notebook analysis_report.ipynb
        ;;
    4)
        echo ""
        echo "🧪 ТЕСТИРОВАНИЕ МОДЕЛИ..."
        echo "========================="
        if [ ! -f "build/sber_auto_model.pkl" ]; then
            echo "⚠️ Модель не найдена. Сначала обучите модель (опция 1)"
            exit 1
        fi
        echo "🚀 Запуск тестирования модели..."
        cd tests && python test_model.py && cd ..
        ;;
    5)
        echo ""
        echo "🧪 ПОЛНОЕ ТЕСТИРОВАНИЕ ПРОЕКТА..."
        echo "================================="
        cd tests && python test_project.py && cd ..
        ;;
    6)
        echo ""
        echo "📊 ТЕСТИРОВАНИЕ JUPYTER NOTEBOOK..."
        echo "==================================="
        cd tests && python test_notebook.py && cd ..
        ;;
    7)
        echo ""
        echo "🎯 ДЕМОНСТРАЦИЯ ЗАПРОСОВ К МОДЕЛИ..."
        echo "===================================="
        if [ ! -f "build/sber_auto_model.pkl" ]; then
            echo "⚠️ Модель не найдена. Сначала обучите модель (опция 1)"
            exit 1
        fi
        echo "🚀 Запуск демонстрации запросов..."
        cd example && python demo_queries.py && cd ..
        ;;
    8)
        echo "👋 До свидания!"
        exit 0
        ;;
    *)
        echo "❌ Неверный выбор. Попробуйте снова."
        exit 1
        ;;
esac

echo ""
echo "✅ Операция завершена!"
