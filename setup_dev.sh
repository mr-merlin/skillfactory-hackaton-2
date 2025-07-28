#!/bin/bash

echo "🚀 Настройка среды разработки с pre-commit"

# Проверяем, что мы в правильной директории
if [ ! -f "requirements.txt" ]; then
    echo "❌ Ошибка: requirements.txt не найден. Запустите скрипт из корневой директории проекта."
    exit 1
fi

echo "📦 Устанавливаем зависимости..."
pip install -r requirements.txt

echo "🔧 Настраиваем pre-commit..."
pre-commit install

echo "🧪 Устанавливаем дополнительные хуки..."
pre-commit install --hook-type pre-push

echo "✅ Настройка завершена!"
echo ""
echo "📋 Доступные команды:"
echo "  pre-commit run --all-files    # Запустить все проверки"
echo "  pre-commit run black          # Только форматирование"
echo "  pre-commit run isort          # Только сортировка импортов"
echo "  pre-commit run flake8         # Только проверка стиля"
echo "  pre-commit run bandit         # Только проверка безопасности"
echo "  pre-commit run mypy           # Только проверка типов"
echo ""
echo "🎯 Pre-commit будет автоматически запускаться при каждом коммите!"
