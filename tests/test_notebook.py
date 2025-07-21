#!/usr/bin/env python3
"""
🧪 Тест Jupyter ноутбука
"""

import requests
import json
import time
import os

# Добавляем путь к модулям
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'code'))

def test_jupyter_server():
    """Тест доступности Jupyter сервера"""
    print("🔍 Тестируем Jupyter сервер...")
    
    try:
        # Проверяем основной URL
        response = requests.get("http://localhost:8889", timeout=5)
        if response.status_code == 200:
            print("✅ Jupyter сервер доступен")
            return True
        else:
            print(f"❌ Jupyter сервер не отвечает: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Ошибка подключения к Jupyter: {e}")
        return False

def test_notebook_file():
    """Тест файла ноутбука"""
    print("\n🔍 Проверяем файл ноутбука...")
    
    try:
        with open('../analysis_report.ipynb', 'r', encoding='utf-8') as f:
            content = f.read()
            
        if '"cells"' in content and '"metadata"' in content:
            print("✅ Файл ноутбука корректен")
            return True
        else:
            print("❌ Файл ноутбука поврежден")
            return False
    except Exception as e:
        print(f"❌ Ошибка чтения файла: {e}")
        return False

def main():
    """Основная функция"""
    print("🚀 ТЕСТИРОВАНИЕ JUPYTER NOTEBOOK")
    print("=" * 40)
    
    tests = [
        ("Jupyter сервер", test_jupyter_server),
        ("Файл ноутбука", test_notebook_file)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                print(f"❌ Тест '{test_name}' провален")
        except Exception as e:
            print(f"❌ Тест '{test_name}' вызвал исключение: {e}")
    
    print("\n" + "=" * 40)
    print(f"📊 РЕЗУЛЬТАТЫ:")
    print(f"   Пройдено: {passed}/{total} тестов")
    
    if passed == total:
        print("🎉 JUPYTER NOTEBOOK ГОТОВ К РАБОТЕ!")
        print("\n📋 ДОСТУП:")
        print("   🌐 URL: http://localhost:8889")
        print("   📁 Файл: analysis_report.ipynb")
        print("\n💡 Для работы с ноутбуком:")
        print("   1. Откройте браузер")
        print("   2. Перейдите по адресу: http://localhost:8889")
        print("   3. Найдите файл analysis_report.ipynb")
        print("   4. Запустите ячейки по порядку")
    else:
        print("⚠️ ЕСТЬ ПРОБЛЕМЫ С JUPYTER")

if __name__ == "__main__":
    main() 