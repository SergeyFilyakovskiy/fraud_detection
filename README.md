# Выявление мошеннических операций с кредитными картами  
### Нейросети и методы одноклассовой классификации  
**Курсовой проект — БГУИР, 2025-2026**

---

## Датасет

### Рекомендуемый датасет: Credit Card Fraud Detection (ULB)

| Параметр | Значение |
|---|---|
| **Источник** | [Kaggle — mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) |
| **Объём** | 284 807 транзакций, 31 признак |
| **Доля мошенничества** | 0.17 % (492 из 284 807) — ярко выраженный дисбаланс |
| **Признаки** | V1–V28 (PCA-анонимизированные), `Time`, `Amount`, `Class` |
| **Формат** | CSV, ~144 МБ |

**Почему именно этот датасет?**

- Стандарт де-факто для задач одноклассовой классификации и обнаружения аномалий
- Содержит `Time` → можно строить временные ряды (ARIMA, GARCH)
- Сильный дисбаланс классов идеально демонстрирует преимущества one-class подходов
- Реальные данные европейских держателей карт (BNP Paribas Fortis, World Line)

**Как скачать:**

```bash
# Вариант 1: Kaggle CLI (рекомендуется)
pip install kaggle
kaggle datasets download -d mlg-ulb/creditcardfraud
unzip creditcardfraud.zip

# Вариант 2: Прямая загрузка через браузер
# → https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Зарегистрироваться (бесплатно) и нажать Download
```

> Положите файл `creditcard.csv` в ту же папку, что и `fraud_detection.py`.  
> Если файл не найден — скрипт автоматически сгенерирует синтетический аналог.

---

### Альтернативные датасеты

| Датасет | Размер | Особенности |
|---|---|---|
| [IEEE-CIS Fraud Detection](https://www.kaggle.com/c/ieee-fraud-detection) | ~590 MB | 400+ признаков, транзакции + идентификационные данные |
| [PaySim Synthetic](https://www.kaggle.com/datasets/ealaxi/paysim1) | ~470 MB | Синтетический, 6 типов транзакций, большой объём |
| [Sparkov CC Fraud (симулятор)](https://www.kaggle.com/datasets/kartik2112/fraud-detection) | ~200 MB | Больше демографических признаков |

---

## Установка зависимостей

```bash
pip install -r requirements.txt
```

Для GPU-ускорения (TensorFlow):
```bash
pip install tensorflow[and-cuda]   # NVIDIA GPU
# или
pip install tensorflow-metal       # Apple Silicon
```

---

## Запуск

```bash
python fraud_detection.py
```

Скрипт:
1. Загружает `creditcard.csv` (или генерирует синтетические данные)
2. Выполняет полный анализ (~10–30 мин в зависимости от CPU/GPU)
3. Сохраняет все графики в папку `results/`
4. Сохраняет обученные нейросети в папку `models/`

---

## Структура проекта

```
fraud_detection/
├── fraud_detection.py      ← главный скрипт (1300+ строк)
├── requirements.txt
├── README.md
├── creditcard.csv          ← добавить вручную (скачать с Kaggle)
├── results/                ← создаётся автоматически
│   ├── 01_eda_overview.png
│   ├── 02_correlation_matrix.png
│   ├── 03_feature_distributions.png
│   ├── 04_time_series_dynamics.png        # Диаграмма 5.1 по ТЗ
│   ├── 05_correlogram_*.png               # Диаграмма 5.2 по ТЗ
│   ├── 06_ts_decomposition.png
│   ├── 07_arima_forecast.png              # Диаграмма 5.3 по ТЗ
│   ├── 08_garch_model.png
│   ├── 09_autoencoder_analysis.png
│   ├── 10_lstm_ae_analysis.png
│   ├── 11_roc_curves.png
│   ├── 12_pr_curves.png
│   ├── 13_confusion_matrices.png
│   ├── 14_metrics_comparison.png
│   ├── 15_feature_importance.png
│   ├── feature_importance.csv
│   └── model_comparison.csv              ← итоговая таблица сравнения
└── models/                 ← создаётся автоматически
    ├── autoencoder.keras
    └── lstm_autoencoder.keras
```

---

## Соответствие требованиям ТЗ

### 3.2.1 Методы ИАД (≥ 4):

| # | Метод | Тип |
|---|---|---|
| 1 | **One-Class SVM** | Одноклассовая классификация |
| 2 | **Isolation Forest** | Ансамблевый детектор аномалий |
| 3 | **Local Outlier Factor** | Плотностной метод |
| 4 | **XGBoost** (Gradient Boosting) | Supervised baseline |
| 5 | **Random Forest** | Ансамблевый supervised baseline |

### 3.2.1 Нейронные сети:

| # | Метод | Описание |
|---|---|---|
| 6 | **Autoencoder** | Одноклассовая НС: обучается только на нормальных транзакциях; мошенничество → высокая ошибка реконструкции |
| 7 | **LSTM Autoencoder** | Учитывает временные зависимости последовательности транзакций |

### 3.2.1 Статистические методы (≥ 2):

| # | Метод | Целевой ряд |
|---|---|---|
| 8 | **ARIMA(p,d,q)** | Почасовая доля мошеннических транзакций |
| 9 | **GARCH(1,1)** | Волатильность средних сумм транзакций |

### 3.2.3 Анализ стационарности:
- Расширенный тест Дики–Фуллера (ADF) с автоподбором лагов
- Коррелограммы ACF и PACF
- Дифференцирование нестационарных рядов
- Декомпозиция (тренд, сезонность, остатки)

### 3.3.3 Метрики качества:

| Задача | Метрики |
|---|---|
| Временные ряды | RMSE, MAE, MAPE по горизонтам: **1, 3, 6, 12, 24 ч.** |
| Классификация | Precision, Recall, F1, ROC-AUC, PR-AUC |

### Кросс-валидация:
- `TimeSeriesSplit` (n=5) — скользящее окно без утечки данных из будущего

---

## Вывод по выбору датасета

Для темы «одноклассовая классификация + нейросети» датасет Credit Card Fraud Detection (ULB) является **оптимальным выбором** по следующим причинам:

1. **Крайний дисбаланс** (0.17%) — именно для таких задач созданы методы одноклассовой классификации  
2. **Столбец `Time`** позволяет строить временные ряды и применять ARIMA/GARCH  
3. **Размер** достаточен (285k строк) для обучения LSTM и Autoencoder  
4. **Чистота данных** — нет пропусков, признаки уже нормированы (PCA)  
5. **Научная база** — сотни опубликованных статей, можно сравнить результаты  
