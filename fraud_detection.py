#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
  ВЫЯВЛЕНИЕ МОШЕННИЧЕСКИХ ОПЕРАЦИЙ С КРЕДИТНЫМИ КАРТАМИ
  с использованием нейросетей и методов одноклассовой классификации

  Курсовой проект — БГУИР, 2025-2026
================================================================================

ДАТАСЕТ:
  Credit Card Fraud Detection (Kaggle / ULB Machine Learning Group)
  Источник: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
  Файл:    creditcard.csv  →  поместить рядом со скриптом

МЕТОДЫ ИАД (≥ 4):
  1.  One-Class SVM          — обучается только на норме (одноклассовый)
  2.  Isolation Forest       — ансамблевый детектор аномалий
  3.  Local Outlier Factor   — плотностной детектор (novelty mode)
  4.  XGBoost / Random Forest— Gradient Boosting (supervised baseline)
  5.  Random Forest          — ансамблевый supervised baseline

НЕЙРОННЫЕ СЕТИ:
  6.  Autoencoder            — одноклассовая НС, ошибка реконструкции
  7.  LSTM Autoencoder       — учитывает временные зависимости

СТАТИСТИЧЕСКИЕ МЕТОДЫ (≥ 2):
  8.  ARIMA                  — прогноз почасовой доли мошенничества
  9.  GARCH(1,1)             — волатильность сумм транзакций

МЕТРИКИ:
  Классификация:    Precision, Recall, F1, ROC-AUC, PR-AUC
  Временные ряды:   RMSE, MAE, MAPE (горизонты 1,3,6,12,24 ч.)

СТРУКТУРА РЕЗУЛЬТАТОВ (папка results/):
  01_eda_overview.png, 02_correlation_matrix.png, 03_feature_distributions.png,
  04_time_series_dynamics.png, 05_correlogram_*.png, 06_ts_decomposition.png,
  07_arima_forecast.png, 08_garch_model.png, 09_autoencoder_analysis.png,
  10_lstm_ae_analysis.png, 11_roc_curves.png, 12_pr_curves.png,
  13_confusion_matrices.png, 14_metrics_comparison.png,
  15_feature_importance.png, model_comparison.csv
================================================================================
"""

import os
import sys
import warnings
import logging
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('fraud_detection.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── Core ─────────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # headless — нет GUI-зависимостей
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# ── Sklearn ───────────────────────────────────────────────────────────────────
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import (
    confusion_matrix, roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
    mean_squared_error, mean_absolute_error,
)
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor

# ── XGBoost (опционально) ─────────────────────────────────────────────────────
try:
    import xgboost as xgb
    XGB_OK = True
except ImportError:
    XGB_OK = False
    log.warning("XGBoost не установлен → используем RandomForest как замену.")

# ── Statsmodels ───────────────────────────────────────────────────────────────
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import het_arch

# ── ARCH / GARCH (опционально) ────────────────────────────────────────────────
try:
    from arch import arch_model
    ARCH_OK = True
except ImportError:
    ARCH_OK = False
    log.warning("arch не установлен → GARCH-анализ пропущен.")

# ── TensorFlow (опционально) ──────────────────────────────────────────────────
try:
    import tensorflow as tf
    from tensorflow.keras import layers, Model, Input
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    TF_OK = True
    log.info(f"TensorFlow {tf.__version__} доступен.")
except ImportError:
    TF_OK = False
    log.warning("TensorFlow не установлен → нейросетевые модели пропущены.")

# ══════════════════════════════════════════════════════════════════════════════
#   КОНФИГУРАЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

SEED = 42
np.random.seed(SEED)
if TF_OK:
    tf.random.set_seed(SEED)

CFG = dict(
    data_path       = "creditcard.csv",
    output_dir      = "results",
    model_dir       = "models",
    time_bin        = "h",           # часовая агрегация (pandas 2.x: 'h' вместо 'H')
    seq_len         = 10,            # длина последовательности для LSTM
    test_frac       = 0.20,
    val_frac        = 0.10,
    ae = dict(
        enc_dim   = 14,
        epochs    = 60,
        batch     = 256,
        lr        = 1e-3,
    ),
    lstm_ae = dict(
        units  = [32, 16],
        epochs = 35,
        batch  = 64,
        lr     = 1e-3,
    ),
    arima = dict(
        max_p          = 3,
        max_q          = 3,
        horizons       = [1, 3, 6, 12, 24],
    ),
)

for d in (CFG["output_dir"], CFG["model_dir"]):
    Path(d).mkdir(exist_ok=True)

# Цвета
C = dict(normal="#2ecc71", fraud="#e74c3c", pred="#3498db", acc="#f39c12")

plt.rcParams.update({
    "figure.figsize": (14, 6),
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "figure.dpi": 100,
})
sns.set_style("whitegrid")

# ══════════════════════════════════════════════════════════════════════════════
#   ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ══════════════════════════════════════════════════════════════════════════════

def savefig(name: str, dpi: int = 150):
    path = f"{CFG['output_dir']}/{name}.png"
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close()
    log.info(f"График → {path}")


def ts_metrics(actual, pred) -> dict:
    """RMSE, MAE, MAPE для временных рядов."""
    a, p = np.array(actual, float), np.array(pred, float)
    mask = a != 0
    rmse = np.sqrt(mean_squared_error(a, p))
    mae  = mean_absolute_error(a, p)
    mape = float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100) if mask.any() else float("nan")
    return dict(RMSE=rmse, MAE=mae, MAPE=mape)


def clf_metrics(y_true, y_pred, y_score=None) -> dict:
    """Метрики классификации."""
    m = dict(
        Accuracy  = accuracy_score(y_true, y_pred),
        Precision = precision_score(y_true, y_pred, zero_division=0),
        Recall    = recall_score(y_true, y_pred, zero_division=0),
        F1        = f1_score(y_true, y_pred, zero_division=0),
    )
    if y_score is not None:
        m["ROC-AUC"] = roc_auc_score(y_true, y_score)
        m["PR-AUC"]  = average_precision_score(y_true, y_score)
    return m


def print_metrics(metrics_dict: dict, title: str = ""):
    if title:
        print(f"\n{'='*60}\n{title}\n{'='*60}")
    for name, m in metrics_dict.items():
        print(f"\n  {name}:")
        for k, v in m.items():
            print(f"    {k:<15}: {v:.4f}" if isinstance(v, float) else f"    {k:<15}: {v}")


# ══════════════════════════════════════════════════════════════════════════════
#   ЧАСТЬ 1.  ЗАГРУЗКА И РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ
# ══════════════════════════════════════════════════════════════════════════════

def load_data() -> pd.DataFrame:
    log.info("=" * 60)
    log.info("ЗАГРУЗКА ДАННЫХ")
    log.info("=" * 60)
    path = CFG["data_path"]
    if os.path.exists(path):
        df = pd.read_csv(path)
        log.info(f"Загружено: {len(df):,} строк, {df.shape[1]} столбцов.")
    else:
        log.warning(f"Файл {path} не найден — генерируем синтетические данные.")
        df = _synthetic_data()
    return df


def _synthetic_data(n: int = 284_807, fraud_frac: float = 0.00173) -> pd.DataFrame:
    """Синтетический датасет, имитирующий creditcard.csv."""
    rng = np.random.default_rng(SEED)
    n_fr = int(n * fraud_frac)
    n_ok = n - n_fr

    cov = np.eye(28)
    X_ok = rng.multivariate_normal(np.zeros(28), cov, n_ok)
    mu_fr = np.zeros(28)
    mu_fr[[0, 3, 4, 7, 10, 11, 14]] = [2.5, -3, 2, -4, 3, -2, 1.5]
    X_fr = rng.multivariate_normal(mu_fr, cov * 2, n_fr)

    t_ok = np.sort(rng.uniform(0, 172_792, n_ok))
    t_fr = np.sort(rng.uniform(0, 172_792, n_fr))
    a_ok = np.abs(rng.exponential(88, n_ok))
    a_fr = np.abs(rng.exponential(122, n_fr))

    cols = [f"V{i}" for i in range(1, 29)]
    df_ok = pd.DataFrame(X_ok, columns=cols)
    df_fr = pd.DataFrame(X_fr, columns=cols)
    df_ok["Time"] = t_ok;  df_ok["Amount"] = a_ok;  df_ok["Class"] = 0
    df_fr["Time"] = t_fr;  df_fr["Amount"] = a_fr;  df_fr["Class"] = 1
    df = pd.concat([df_ok, df_fr]).sort_values("Time").reset_index(drop=True)
    log.info(f"Синтетический датасет: {len(df):,} транзакций, "
             f"{int(df.Class.sum())} мошеннических.")
    return df


def perform_eda(df: pd.DataFrame) -> float:
    log.info("=" * 60)
    log.info("РАЗВЕДОЧНЫЙ АНАЛИЗ ДАННЫХ")
    log.info("=" * 60)

    # ── базовые факты ──────────────────────────────────────────────────────
    cnts   = df["Class"].value_counts()
    fr_pct = cnts.get(1, 0) / len(df) * 100
    print(f"\nВсего:            {len(df):>10,}")
    print(f"Нормальных:       {cnts.get(0,0):>10,}  ({100-fr_pct:.4f} %)")
    print(f"Мошеннических:    {cnts.get(1,0):>10,}  ({fr_pct:.4f} %)")
    print(f"Дисбаланс:        {cnts.get(0,0)/cnts.get(1,1):.1f}:1")

    miss = df.isnull().sum()
    print("\nПропущенные значения:", "нет" if not miss.any() else miss[miss > 0].to_dict())

    for cls, lbl in [(0, "Нормальные"), (1, "Мошеннические")]:
        a = df.loc[df.Class == cls, "Amount"]
        print(f"{lbl}: mean={a.mean():.2f}, median={a.median():.2f}, std={a.std():.2f}, max={a.max():.2f}")

    # ── визуализация 1: обзор ──────────────────────────────────────────────
    fig = plt.figure(figsize=(20, 11))
    gs  = gridspec.GridSpec(2, 3, figure=fig)

    ax0 = fig.add_subplot(gs[0, 0])
    ax0.pie(cnts, labels=["Норма", "Мошен."], colors=[C["normal"], C["fraud"]],
            autopct="%1.4f%%", startangle=90)
    ax0.set_title("Распределение классов", fontweight="bold")

    ax1 = fig.add_subplot(gs[0, 1])
    for cls, lbl, col in [(0, "Нормальные", C["normal"]), (1, "Мошеннические", C["fraud"])]:
        ax1.hist(df.loc[df.Class == cls, "Amount"].clip(upper=600),
                 bins=50, alpha=0.6, color=col, label=lbl, density=True)
    ax1.set(xlabel="Сумма ($)", ylabel="Плотность",
            title="Распределение сумм транзакций")
    ax1.legend()

    ax2 = fig.add_subplot(gs[0, 2])
    hrs = df["Time"] / 3600
    for cls, lbl, col in [(0, "Нормальные", C["normal"]), (1, "Мошеннические", C["fraud"])]:
        ax2.hist(hrs[df.Class == cls], bins=48, alpha=0.6, color=col, label=lbl, density=True)
    ax2.set(xlabel="Время (ч.)", ylabel="Плотность",
            title="Распределение по времени")
    ax2.legend()

    ax3 = fig.add_subplot(gs[1, 0])
    df.boxplot(column="Amount", by="Class", ax=ax3)
    ax3.set_ylim(0, 600)
    ax3.set(xlabel="Класс (0=Норма, 1=Мошен.)", ylabel="Сумма ($)",
            title="Box-plot сумм")
    plt.sca(ax3); plt.title("Box-plot сумм")

    ax4 = fig.add_subplot(gs[1, 1:])
    vcols = [f"V{i}" for i in range(1, 29)]
    ok_m  = df.loc[df.Class == 0, vcols].mean()
    fr_m  = df.loc[df.Class == 1, vcols].mean()
    x     = np.arange(len(vcols))
    ax4.bar(x - 0.18, ok_m, 0.36, label="Норма",   color=C["normal"], alpha=0.75)
    ax4.bar(x + 0.18, fr_m, 0.36, label="Мошен.",  color=C["fraud"],  alpha=0.75)
    ax4.set_xticks(x[::4]); ax4.set_xticklabels(vcols[::4], rotation=45)
    ax4.set(xlabel="Признак", ylabel="Среднее", title="Средние значения V1–V28 по классам")
    ax4.legend()

    plt.suptitle("Разведочный анализ данных", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig("01_eda_overview")

    # ── визуализация 2: корреляционная матрица ─────────────────────────────
    fig, ax = plt.subplots(figsize=(20, 16))
    corr = df[vcols + ["Amount", "Class"]].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                linewidths=0.3, ax=ax, cbar_kws={"label": "r"})
    ax.set_title("Корреляционная матрица признаков", fontweight="bold", fontsize=14)
    plt.tight_layout()
    savefig("02_correlation_matrix")

    # ── визуализация 3: топ-8 различающихся признаков ─────────────────────
    ttests = [(col, abs(stats.ttest_ind(df.loc[df.Class==0, col],
                                        df.loc[df.Class==1, col])[0]))
              for col in vcols]
    top8 = [c for c, _ in sorted(ttests, key=lambda x: x[1], reverse=True)[:8]]

    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for ax, col in zip(axes.flatten(), top8):
        for cls, lbl, col2 in [(0, "Норма", C["normal"]), (1, "Мошен.", C["fraud"])]:
            ax.hist(df.loc[df.Class == cls, col], bins=50, alpha=0.6,
                    color=col2, label=lbl, density=True)
        ax.set(title=f"Признак {col}", xlabel="Значение", ylabel="Плотность")
        ax.legend()
    plt.suptitle("Топ-8 наиболее различающихся признаков", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig("03_feature_distributions")

    log.info("EDA завершён.")
    return fr_pct


# ══════════════════════════════════════════════════════════════════════════════
#   ЧАСТЬ 2.  СОЗДАНИЕ ВРЕМЕННЫХ РЯДОВ И ПРИЗНАКОВАЯ ИНЖЕНЕРИЯ
# ══════════════════════════════════════════════════════════════════════════════

def create_time_series(df: pd.DataFrame):
    """Почасовая агрегация транзакций → временной ряд."""
    log.info("=" * 60)
    log.info("СОЗДАНИЕ ВРЕМЕННЫХ РЯДОВ")
    log.info("=" * 60)

    base = pd.Timestamp("2013-09-25")
    df   = df.copy()
    df["Datetime"] = base + pd.to_timedelta(df["Time"], unit="s")

    ts = df.groupby(pd.Grouper(key="Datetime", freq=CFG["time_bin"])).agg(
        total  = ("Class", "count"),
        frauds = ("Class", "sum"),
        amount = ("Amount", "sum"),
        mean_a = ("Amount", "mean"),
        std_a  = ("Amount", "std"),
    ).fillna(0)
    ts["fraud_rate"] = ts["frauds"] / ts["total"].replace(0, np.nan).fillna(0)

    log.info(f"Временной ряд: {len(ts)} часовых периодов, "
             f"средняя доля фрода = {ts.fraud_rate.mean():.5f}")

    # ── график динамики (Диаграмма 5.1 по ТЗ) ─────────────────────────────
    fig, axes = plt.subplots(4, 1, figsize=(16, 20))

    axes[0].plot(ts.index, ts["total"], color="navy", lw=1.5)
    axes[0].fill_between(ts.index, ts["total"], alpha=0.25, color="navy")
    axes[0].set(title="Общее число транзакций", ylabel="Кол-во")

    axes[1].plot(ts.index, ts["frauds"], color=C["fraud"], lw=1.5)
    axes[1].fill_between(ts.index, ts["frauds"], alpha=0.25, color=C["fraud"])
    axes[1].set(title="Мошеннические транзакции", ylabel="Кол-во")

    axes[2].plot(ts.index, ts["fraud_rate"] * 100, color="purple", lw=1.5)
    axes[2].fill_between(ts.index, ts["fraud_rate"] * 100, alpha=0.25, color="purple")
    axes[2].set(title="Доля мошенничества (%)", ylabel="%")

    axes[3].plot(ts.index, ts["mean_a"], color=C["acc"], lw=1.5, label="Среднее")
    axes[3].fill_between(ts.index,
                         (ts["mean_a"] - ts["std_a"].fillna(0)).clip(lower=0),
                         ts["mean_a"] + ts["std_a"].fillna(0),
                         alpha=0.2, color=C["acc"], label="±std")
    axes[3].set(title="Динамика сумм транзакций", ylabel="$ (среднее)")
    axes[3].legend()

    for ax in axes:
        ax.set_xlabel("Дата / Время")
        ax.grid(alpha=0.3)
    plt.suptitle("Диаграмма динамики результирующего и факторных признаков",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig("04_time_series_dynamics")

    return ts, df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("ПРИЗНАКОВАЯ ИНЖЕНЕРИЯ")
    log.info("=" * 60)

    df = df.copy().sort_values("Time").reset_index(drop=True)

    df["Hour"]      = df["Datetime"].dt.hour
    df["DayOfWeek"] = df["Datetime"].dt.dayofweek
    df["IsNight"]   = ((df["Hour"] >= 0) & (df["Hour"] <= 6)).astype(int)
    df["Log_Amount"]= np.log1p(df["Amount"])

    # Скользящие статистики (лаговые признаки)
    w = 10
    df["Amount_MA"]  = df["Amount"].rolling(w, min_periods=1).mean()
    df["Amount_STD"] = df["Amount"].rolling(w, min_periods=1).std().fillna(0)
    df["Amount_Dev"] = (df["Amount"] - df["Amount_MA"]) / (df["Amount_STD"] + 1e-6)

    new_cols = ["Hour", "DayOfWeek", "IsNight", "Log_Amount",
                "Amount_MA", "Amount_STD", "Amount_Dev"]
    log.info(f"Новые признаки: {new_cols}")
    log.info(f"Итого признаков: {df.shape[1]}")
    return df


def make_sequences(X, y, seq_len: int):
    """X → (n_samples, seq_len, n_features) для LSTM."""
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i : i + seq_len])
        ys.append(y[i + seq_len])
    return np.array(Xs), np.array(ys)


# ══════════════════════════════════════════════════════════════════════════════
#   ЧАСТЬ 3.  СТАТИСТИЧЕСКИЙ АНАЛИЗ ВРЕМЕННЫХ РЯДОВ
# ══════════════════════════════════════════════════════════════════════════════

def adf_test(series: pd.Series, name: str = "ряд") -> bool:
    """Расширенный тест Дики–Фуллера. Возвращает True если ряд стационарен."""
    res = adfuller(series.dropna(), autolag="AIC")
    stat, pval, lags, nobs, cv = res[0], res[1], res[2], res[3], res[4]
    print(f"\n  АДФ-тест: {name}")
    print(f"    Тест-статистика : {stat:10.5f}")
    print(f"    p-значение      : {pval:10.5f}")
    print(f"    Использовано лагов: {lags}")
    for level, crit in cv.items():
        print(f"    Крит. значение {level}: {crit:.4f}")
    is_stat = pval <= 0.05
    print(f"    → {'СТАЦИОНАРЕН ✓' if is_stat else 'НЕСТАЦИОНАРЕН ✗'}")
    return is_stat


def make_stationary(series: pd.Series, max_d: int = 2):
    """Дифференцирование до стационарности. Возвращает (ряд, порядок d)."""
    s, d = series.copy(), 0
    for d in range(max_d + 1):
        if adf_test(s, f"diff={d}"):
            break
        if d < max_d:
            s = s.diff().dropna()
    return s, d


def plot_correlograms(series: pd.Series, name: str, lags: int = 20):
    """АКФ и ЧАКФ (Диаграмма 5.2 по ТЗ)."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 5))
    plot_acf(series.dropna(),  lags=lags, ax=axes[0], zero=False, alpha=0.05)
    axes[0].set(title=f"ACF — {name}", xlabel="Лаг", ylabel="Корреляция")
    plot_pacf(series.dropna(), lags=lags, ax=axes[1], zero=False, alpha=0.05, method="ywm")
    axes[1].set(title=f"PACF — {name}", xlabel="Лаг", ylabel="Корреляция")
    plt.suptitle(f"Коррелограммы: {name}", fontsize=13, fontweight="bold")
    plt.tight_layout()
    savefig(f"05_correlogram_{name[:30].lower().replace(' ', '_')}")


def analyze_time_series(ts: pd.DataFrame) -> dict:
    """ARIMA + GARCH анализ."""
    log.info("=" * 60)
    log.info("СТАТИСТИЧЕСКИЙ АНАЛИЗ ВРЕМЕННЫХ РЯДОВ")
    log.info("=" * 60)

    results = {}
    fr = ts["fraud_rate"].copy()

    # ── 1. Ряд доли мошенничества ─────────────────────────────────────────
    print("\n" + "-" * 50)
    print("1. АНАЛИЗ РЯДА ДОЛИ МОШЕННИЧЕСТВА (→ ARIMA)")
    print("-" * 50)

    plot_correlograms(fr, "Доля мошенничества")

    # Декомпозиция (при достаточной длине)
    if len(fr) >= 24:
        try:
            decomp = seasonal_decompose(fr, model="additive",
                                        period=12, extrapolate_trend="freq")
            fig, axes = plt.subplots(4, 1, figsize=(16, 16))
            for ax, data, lbl in zip(
                axes,
                [decomp.observed, decomp.trend, decomp.seasonal, decomp.resid],
                ["Исходный ряд", "Тренд", "Сезонность", "Остатки"],
            ):
                ax.plot(data, color="navy", lw=1.5)
                ax.set(title=lbl)
                ax.grid(alpha=0.3)
            plt.suptitle("Декомпозиция: доля мошенничества", fontsize=13, fontweight="bold")
            plt.tight_layout()
            savefig("06_ts_decomposition")
        except Exception as e:
            log.warning(f"Декомпозиция не выполнена: {e}")

    is_stat = adf_test(fr, "Доля мошенничества")
    _, d = make_stationary(fr) if not is_stat else (fr, 0)

    arima_res = _fit_arima(fr, d)
    results["arima"] = arima_res

    # ── 2. Ряд сумм → GARCH ───────────────────────────────────────────────
    print("\n" + "-" * 50)
    print("2. ВОЛАТИЛЬНОСТЬ СУММ ТРАНЗАКЦИЙ (→ ARCH/GARCH)")
    print("-" * 50)

    mean_a = ts["mean_a"].dropna()
    plot_correlograms(mean_a, "Средняя сумма транзакций")
    adf_test(mean_a, "Средняя сумма транзакций")

    if ARCH_OK and len(mean_a) > 20:
        arch_lm, arch_p, *_ = het_arch(mean_a.values, nlags=5)
        print(f"  Тест ARCH: LM={arch_lm:.4f}, p={arch_p:.4f}")
        if arch_p < 0.05:
            print("  → ARCH-эффекты обнаружены, строим GARCH(1,1).")
            results["garch"] = _fit_garch(mean_a)
        else:
            print("  → ARCH-эффектов нет.")

    return results


def _fit_arima(series: pd.Series, d: int) -> dict:
    """Подбор ARIMA(p,d,q) по AIC + прогноз по горизонтам."""
    print("\nПодбор ARIMA (перебор p,q)…")
    best_aic, best_ord = np.inf, (1, d, 1)
    for p in range(CFG["arima"]["max_p"] + 1):
        for q in range(CFG["arima"]["max_q"] + 1):
            try:
                fit = ARIMA(series, order=(p, d, q)).fit()
                if fit.aic < best_aic:
                    best_aic, best_ord = fit.aic, (p, d, q)
            except Exception:
                pass
    print(f"Лучший порядок ARIMA{best_ord}, AIC={best_aic:.2f}")

    split = int(len(series) * 0.8)
    train, test = series.iloc[:split], series.iloc[split:]

    try:
        fit = ARIMA(train, order=best_ord).fit()
        print(fit.summary())

        # Прогноз + метрики по горизонтам
        h_metrics = {}
        for h in CFG["arima"]["horizons"]:
            if h > len(test):
                continue
            fc   = fit.forecast(steps=h)
            act  = test.values[:h]
            m    = ts_metrics(act, fc[:len(act)])
            h_metrics[h] = m
            mape_s = f"{m['MAPE']:.2f}" if not np.isnan(m["MAPE"]) else "N/A"
            print(f"  h={h:2d}: RMSE={m['RMSE']:.4f}  MAE={m['MAE']:.4f}  MAPE={mape_s}%")

        # Визуализация (Диаграмма 5.3 по ТЗ)
        fc_full = fit.forecast(steps=len(test))
        fig, axes = plt.subplots(2, 1, figsize=(16, 12))

        axes[0].plot(series.index, series.values, color="navy", lw=1.5, label="Факт")
        if len(test) > 0:
            axes[0].plot(test.index, fc_full[:len(test)],
                         color=C["fraud"], lw=2, ls="--",
                         label=f"Прогноз ARIMA{best_ord}")
            axes[0].axvline(train.index[-1], color="gray", ls=":", alpha=0.7,
                            label="Граница обуч./тест")
        axes[0].set(title=f"ARIMA{best_ord}: прогноз доли мошенничества",
                    xlabel="Время", ylabel="Доля")
        axes[0].legend()
        axes[0].grid(alpha=0.3)

        resid = pd.Series(fit.resid)
        axes[1].plot(resid.index, resid.values, color="gray", lw=0.8, alpha=0.8)
        axes[1].axhline(0, color="red", ls="--", lw=1)
        axes[1].set(title="Остатки ARIMA", xlabel="Индекс", ylabel="Остаток")
        axes[1].grid(alpha=0.3)

        plt.suptitle("ARIMA: прогноз временного ряда доли мошенничества",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        savefig("07_arima_forecast")

        # Таблица метрик
        print("\n  Сводная таблица метрик ARIMA по горизонтам:")
        print(f"  {'Горизонт':>10}  {'RMSE':>10}  {'MAE':>10}  {'MAPE(%)':>10}")
        print("  " + "-" * 46)
        for h, m in h_metrics.items():
            mape_s = f"{m['MAPE']:.2f}" if not np.isnan(m["MAPE"]) else "N/A"
            print(f"  {h:>10}  {m['RMSE']:>10.4f}  {m['MAE']:>10.4f}  {mape_s:>10}")

        return dict(order=best_ord, aic=best_aic, model=fit, h_metrics=h_metrics)
    except Exception as e:
        log.error(f"ARIMA: {e}")
        return {}


def _fit_garch(series: pd.Series) -> dict:
    """GARCH(1,1) на приростах средних сумм."""
    returns = series.pct_change().dropna() * 100
    try:
        gm  = arch_model(returns, vol="Garch", p=1, q=1, dist="normal")
        fit = gm.fit(disp="off")
        print(fit.summary())

        fig, axes = plt.subplots(3, 1, figsize=(16, 14))
        axes[0].plot(returns.index, returns.values, color="navy", lw=0.8, alpha=0.7)
        axes[0].set(title="Приросты средних сумм (%)", ylabel="% изм.")
        axes[0].grid(alpha=0.3)

        cv = fit.conditional_volatility
        axes[1].plot(cv.index, cv.values, color=C["fraud"], lw=1.5)
        axes[1].fill_between(cv.index, 0, cv.values, alpha=0.25, color=C["fraud"])
        axes[1].set(title="Условная волатильность GARCH(1,1)", ylabel="Волатильность")
        axes[1].grid(alpha=0.3)

        sr = fit.std_resid
        axes[2].plot(sr.index, sr.values, color="gray", lw=0.7, alpha=0.7)
        axes[2].axhline(0, color="red", ls="--", lw=1)
        axes[2].set(title="Стандартизированные остатки", ylabel="Стд. ост.")
        axes[2].grid(alpha=0.3)

        plt.suptitle("GARCH(1,1): волатильность сумм транзакций",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        savefig("08_garch_model")
        return dict(model=fit, aic=fit.aic, bic=fit.bic)
    except Exception as e:
        log.error(f"GARCH: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
#   ЧАСТЬ 4.  ПОДГОТОВКА ДАННЫХ ДЛЯ ML
# ══════════════════════════════════════════════════════════════════════════════

FEATURE_COLS = (
    [f"V{i}" for i in range(1, 29)]
    + ["Log_Amount", "Hour", "DayOfWeek", "IsNight",
       "Amount_MA", "Amount_STD", "Amount_Dev"]
)


def prepare_ml(df: pd.DataFrame) -> dict:
    log.info("=" * 60)
    log.info("ПОДГОТОВКА ДАННЫХ ДЛЯ ML")
    log.info("=" * 60)

    fc = [c for c in FEATURE_COLS if c in df.columns]
    X  = df[fc].values
    y  = df["Class"].values

    n = len(X)
    i_val  = int(n * (1 - CFG["test_frac"] - CFG["val_frac"]))
    i_test = int(n * (1 - CFG["test_frac"]))

    X_tr, y_tr = X[:i_val],        y[:i_val]
    X_v,  y_v  = X[i_val:i_test],  y[i_val:i_test]
    X_te, y_te = X[i_test:],       y[i_test:]

    X_tr_ok = X_tr[y_tr == 0]      # только нормальные для one-class

    scaler = RobustScaler().fit(X_tr)

    log.info(f"Train: {len(X_tr):,}  Val: {len(X_v):,}  Test: {len(X_te):,}")
    log.info(f"Нормальных в train: {len(X_tr_ok):,}")

    return dict(
        X_tr      = scaler.transform(X_tr),
        y_tr      = y_tr,
        X_tr_ok   = scaler.transform(X_tr_ok),
        X_v       = scaler.transform(X_v),
        y_v       = y_v,
        X_te      = scaler.transform(X_te),
        y_te      = y_te,
        scaler    = scaler,
        feat_cols = fc,
        n_feats   = len(fc),
    )


# ══════════════════════════════════════════════════════════════════════════════
#   ЧАСТЬ 5.  МЕТОДЫ ИАД (≥ 4)
# ══════════════════════════════════════════════════════════════════════════════

def _opt_threshold(val_scores, y_v):
    """Поиск порога по максимальному F1 на валидации."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.linspace(0.01, 0.99, 99):
        f1 = f1_score(y_v, (val_scores >= thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


# ── Метод 1: One-Class SVM ───────────────────────────────────────────────────
def train_ocsvm(d: dict) -> dict:
    log.info("\n--- Метод 1: One-Class SVM ---")
    best_f1, best_nu, best_gm = 0.0, 0.05, "scale"

    for nu in [0.01, 0.03, 0.05, 0.10]:
        for gm in ["scale", "auto"]:
            m = OneClassSVM(nu=nu, kernel="rbf", gamma=gm).fit(d["X_tr_ok"])
            p = (m.predict(d["X_v"]) == -1).astype(int)
            f = f1_score(d["y_v"], p, zero_division=0)
            if f > best_f1:
                best_f1, best_nu, best_gm = f, nu, gm

    print(f"  Best: nu={best_nu}, gamma={best_gm}, F1_val={best_f1:.4f}")
    m = OneClassSVM(nu=best_nu, kernel="rbf", gamma=best_gm).fit(d["X_tr_ok"])

    scores = -m.decision_function(d["X_te"])
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    preds  = (m.predict(d["X_te"]) == -1).astype(int)
    met    = clf_metrics(d["y_te"], preds, scores)
    print(f"  Test: {met}")
    return dict(predictions=preds, scores=scores, metrics=met,
                params=dict(nu=best_nu, gamma=best_gm))


# ── Метод 2: Isolation Forest ────────────────────────────────────────────────
def train_iforest(d: dict) -> dict:
    log.info("\n--- Метод 2: Isolation Forest ---")
    fr_rate = d["y_tr"].mean()
    best_f1, best_p = 0.0, dict(n_estimators=200, contamination=fr_rate)

    for n_est in [100, 200, 300]:
        for cont in [0.001, fr_rate, 0.005]:
            m = IsolationForest(n_estimators=n_est, contamination=cont,
                                random_state=SEED, n_jobs=-1).fit(d["X_tr_ok"])
            p = (m.predict(d["X_v"]) == -1).astype(int)
            f = f1_score(d["y_v"], p, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_p  = dict(n_estimators=n_est, contamination=cont)

    print(f"  Best: {best_p}, F1_val={best_f1:.4f}")
    m = IsolationForest(**best_p, random_state=SEED, n_jobs=-1).fit(d["X_tr_ok"])

    scores = -m.score_samples(d["X_te"])
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    preds  = (m.predict(d["X_te"]) == -1).astype(int)
    met    = clf_metrics(d["y_te"], preds, scores)
    print(f"  Test: {met}")
    return dict(predictions=preds, scores=scores, metrics=met, params=best_p)


# ── Метод 3: Local Outlier Factor ────────────────────────────────────────────
def train_lof(d: dict) -> dict:
    log.info("\n--- Метод 3: Local Outlier Factor ---")
    fr_rate = d["y_tr"].mean()
    best_f1, best_p = 0.0, dict(n_neighbors=20, contamination=fr_rate)

    for nn in [10, 20, 30, 50]:
        for cont in [0.001, fr_rate, 0.005]:
            m = LocalOutlierFactor(n_neighbors=nn, contamination=cont,
                                   novelty=True, n_jobs=-1).fit(d["X_tr_ok"])
            p = (m.predict(d["X_v"]) == -1).astype(int)
            f = f1_score(d["y_v"], p, zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_p  = dict(n_neighbors=nn, contamination=cont)

    print(f"  Best: {best_p}, F1_val={best_f1:.4f}")
    m = LocalOutlierFactor(**best_p, novelty=True, n_jobs=-1).fit(d["X_tr_ok"])

    scores = -m.score_samples(d["X_te"])
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-10)
    preds  = (m.predict(d["X_te"]) == -1).astype(int)
    met    = clf_metrics(d["y_te"], preds, scores)
    print(f"  Test: {met}")
    return dict(predictions=preds, scores=scores, metrics=met, params=best_p)


# ── Метод 4: Gradient Boosting (XGBoost / RandomForest) ─────────────────────
def train_gbm(d: dict) -> dict:
    name = "XGBoost" if XGB_OK else "Random Forest"
    log.info(f"\n--- Метод 4: {name} (supervised baseline) ---")

    w = int((d["y_tr"] == 0).sum() / max((d["y_tr"] == 1).sum(), 1))

    if XGB_OK:
        m = xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            scale_pos_weight=w, subsample=0.8, colsample_bytree=0.8,
            eval_metric="aucpr", random_state=SEED,
            use_label_encoder=False, n_jobs=-1, early_stopping_rounds=15,
        )
        m.fit(d["X_tr"], d["y_tr"],
              eval_set=[(d["X_v"], d["y_v"])], verbose=False)
    else:
        m = RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight={0: 1, 1: w},
            random_state=SEED, n_jobs=-1,
        ).fit(d["X_tr"], d["y_tr"])

    val_sc   = m.predict_proba(d["X_v"])[:, 1]
    best_thr, _ = _opt_threshold(val_sc, d["y_v"])
    scores   = m.predict_proba(d["X_te"])[:, 1]
    preds    = (scores >= best_thr).astype(int)
    met      = clf_metrics(d["y_te"], preds, scores)
    print(f"  Оптимальный порог: {best_thr:.2f}")
    print(f"  Test: {met}")
    return dict(model=m, predictions=preds, scores=scores, metrics=met,
                threshold=best_thr, name=name)


# ── Метод 5: Random Forest ────────────────────────────────────────────────────
def train_rf(d: dict) -> dict:
    log.info("\n--- Метод 5: Random Forest (Grid Search CV) ---")
    w = int((d["y_tr"] == 0).sum() / max((d["y_tr"] == 1).sum(), 1))

    gs = GridSearchCV(
        RandomForestClassifier(class_weight={0: 1, 1: w},
                               random_state=SEED, n_jobs=-1),
        param_grid=dict(n_estimators=[100, 200], max_depth=[8, 12, None]),
        cv=TimeSeriesSplit(n_splits=3),
        scoring="f1", n_jobs=-1, verbose=0,
    ).fit(d["X_tr"], d["y_tr"])

    print(f"  Best params: {gs.best_params_}, CV-F1={gs.best_score_:.4f}")
    m = gs.best_estimator_

    val_sc   = m.predict_proba(d["X_v"])[:, 1]
    best_thr, _ = _opt_threshold(val_sc, d["y_v"])
    scores   = m.predict_proba(d["X_te"])[:, 1]
    preds    = (scores >= best_thr).astype(int)
    met      = clf_metrics(d["y_te"], preds, scores)
    print(f"  Test: {met}")
    return dict(model=m, predictions=preds, scores=scores, metrics=met,
                threshold=best_thr)


# ══════════════════════════════════════════════════════════════════════════════
#   ЧАСТЬ 6.  НЕЙРОННЫЕ СЕТИ
# ══════════════════════════════════════════════════════════════════════════════

def _build_ae(n: int, enc: int) -> "Model":
    inp = Input(shape=(n,))
    x   = layers.Dense(64, activation="relu")(inp)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(32, activation="relu")(x)
    x   = layers.BatchNormalization()(x)
    btl = layers.Dense(enc, activation="relu", name="bottleneck")(x)
    x   = layers.Dense(32, activation="relu")(btl)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(n,  activation="linear")(x)
    ae  = Model(inp, out, name="Autoencoder")
    ae.compile(optimizer=Adam(CFG["ae"]["lr"]), loss="mse")
    return ae


def _build_lstm_ae(seq: int, n: int, units) -> "Model":
    inp = Input(shape=(seq, n))
    x   = layers.LSTM(units[0], activation="tanh", return_sequences=True)(inp)
    x   = layers.Dropout(0.2)(x)
    x   = layers.LSTM(units[1], activation="tanh", return_sequences=False)(x)
    x   = layers.RepeatVector(seq)(x)
    x   = layers.LSTM(units[1], activation="tanh", return_sequences=True)(x)
    x   = layers.Dropout(0.2)(x)
    x   = layers.LSTM(units[0], activation="tanh", return_sequences=True)(x)
    out = layers.TimeDistributed(layers.Dense(n, activation="linear"))(x)
    m   = Model(inp, out, name="LSTM_Autoencoder")
    m.compile(optimizer=Adam(CFG["lstm_ae"]["lr"]), loss="mse")
    return m


def _ae_threshold(model, X_ok_val, X_val, y_val):
    """Оптимальный порог по F1 на валидации (для AE)."""
    rec_ok = model.predict(X_ok_val, verbose=0)
    err_ok = np.mean((X_ok_val - rec_ok) ** 2, axis=1)

    rec_v  = model.predict(X_val, verbose=0)
    err_v  = np.mean((X_val - rec_v) ** 2, axis=1)

    best_f1, best_thr = 0.0, np.percentile(err_ok, 95)
    for pct in [88, 90, 92, 95, 97, 99]:
        thr = np.percentile(err_ok, pct)
        f1  = f1_score(y_val, (err_v > thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return best_thr, best_f1


def train_autoencoder(d: dict) -> dict | None:
    if not TF_OK:
        log.warning("TF недоступен — Autoencoder пропущен.")
        return None

    log.info("\n--- НС 1: Autoencoder ---")
    cfg = CFG["ae"]
    ae  = _build_ae(d["n_feats"], cfg["enc_dim"])
    ae.summary()

    cbs = [
        EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]
    hist = ae.fit(
        d["X_tr_ok"], d["X_tr_ok"],
        epochs=cfg["epochs"], batch_size=cfg["batch"],
        validation_split=0.1, callbacks=cbs,
        verbose=1, shuffle=True,
    )
    ae.save(f"{CFG['model_dir']}/autoencoder.keras")

    X_v_ok = d["X_v"][d["y_v"] == 0]
    thr, f1_v = _ae_threshold(ae, X_v_ok, d["X_v"], d["y_v"])
    print(f"  Порог={thr:.5f}, F1_val={f1_v:.4f}")

    rec    = ae.predict(d["X_te"], verbose=0)
    err    = np.mean((d["X_te"] - rec) ** 2, axis=1)
    preds  = (err > thr).astype(int)
    scores = (err - err.min()) / (err.max() - err.min() + 1e-10)
    met    = clf_metrics(d["y_te"], preds, scores)
    print(f"  Test: {met}")

    # Визуализация
    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    axes[0].plot(hist.history["loss"],     label="Train", color="navy")
    axes[0].plot(hist.history["val_loss"], label="Val",   color=C["fraud"])
    axes[0].set(title="Потери Autoencoder", xlabel="Эпоха", ylabel="MSE")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    for cls, lbl, col in [(0, "Норма", C["normal"]), (1, "Мошен.", C["fraud"])]:
        axes[1].hist(err[d["y_te"] == cls], bins=60, alpha=0.65,
                     color=col, label=lbl, density=True)
    axes[1].axvline(thr, color="black", ls="--", lw=2, label=f"Порог={thr:.4f}")
    axes[1].set(title="Ошибка реконструкции (Test)",
                xlabel="MSE", ylabel="Плотность")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    idx_ok = np.where(d["y_te"] == 0)[0][:2000]
    idx_fr = np.where(d["y_te"] == 1)[0]
    axes[2].scatter(range(len(idx_ok)), err[idx_ok],
                    c=C["normal"], alpha=0.2, s=4,  label="Норма")
    axes[2].scatter(range(len(idx_fr)), err[idx_fr],
                    c=C["fraud"],  alpha=0.7, s=15, label="Мошен.")
    axes[2].axhline(thr, color="black", ls="--", lw=2, label="Порог")
    axes[2].set(title="Ошибки по образцам", xlabel="Индекс", ylabel="MSE")
    axes[2].legend(); axes[2].grid(alpha=0.3)

    plt.suptitle("Анализ Autoencoder", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig("09_autoencoder_analysis")

    return dict(predictions=preds, scores=scores, metrics=met,
                errors=err, threshold=thr, history=hist)


def train_lstm_autoencoder(d: dict) -> dict | None:
    if not TF_OK:
        log.warning("TF недоступен — LSTM Autoencoder пропущен.")
        return None

    log.info("\n--- НС 2: LSTM Autoencoder ---")
    cfg = CFG["lstm_ae"]
    sl  = CFG["seq_len"]

    X_seqs_ok, _ = make_sequences(d["X_tr_ok"],
                                   np.zeros(len(d["X_tr_ok"])), sl)
    X_te_seqs, y_te_seqs = make_sequences(d["X_te"], d["y_te"], sl)

    log.info(f"Последовательности: train={X_seqs_ok.shape}, test={X_te_seqs.shape}")

    ae  = _build_lstm_ae(sl, d["n_feats"], cfg["units"])
    ae.summary()

    cbs = [
        EarlyStopping(monitor="val_loss", patience=7, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
    ]
    hist = ae.fit(
        X_seqs_ok, X_seqs_ok,
        epochs=cfg["epochs"], batch_size=cfg["batch"],
        validation_split=0.1, callbacks=cbs,
        verbose=1, shuffle=False,
    )
    ae.save(f"{CFG['model_dir']}/lstm_autoencoder.keras")

    rec  = ae.predict(X_te_seqs, verbose=0)
    err  = np.mean(np.mean((X_te_seqs - rec) ** 2, axis=2), axis=1)

    err_ok = err[y_te_seqs == 0]
    best_f1, best_thr = 0.0, np.percentile(err_ok, 95)
    for pct in [88, 90, 92, 95, 97, 99]:
        thr = np.percentile(err_ok, pct)
        f1  = f1_score(y_te_seqs, (err > thr).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr

    preds  = (err > best_thr).astype(int)
    scores = (err - err.min()) / (err.max() - err.min() + 1e-10)
    met    = clf_metrics(y_te_seqs, preds, scores)
    print(f"  Test: {met}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    axes[0].plot(hist.history["loss"],     label="Train", color="navy")
    axes[0].plot(hist.history["val_loss"], label="Val",   color=C["fraud"])
    axes[0].set(title="Потери LSTM AE", xlabel="Эпоха", ylabel="MSE")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    for cls, lbl, col in [(0, "Норма", C["normal"]), (1, "Мошен.", C["fraud"])]:
        axes[1].hist(err[y_te_seqs == cls], bins=60, alpha=0.65,
                     color=col, label=lbl, density=True)
    axes[1].axvline(best_thr, color="black", ls="--", lw=2, label="Порог")
    axes[1].set(title="Ошибки реконструкции LSTM AE", xlabel="MSE", ylabel="Плотность")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.suptitle("Анализ LSTM Autoencoder", fontsize=14, fontweight="bold")
    plt.tight_layout()
    savefig("10_lstm_ae_analysis")

    return dict(predictions=preds, scores=scores, metrics=met,
                errors=err, threshold=best_thr, y_te=y_te_seqs, history=hist)


# ══════════════════════════════════════════════════════════════════════════════
#   ЧАСТЬ 7.  ОЦЕНКА И СРАВНЕНИЕ МОДЕЛЕЙ
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_all(results: dict, d: dict) -> pd.DataFrame:
    log.info("=" * 60)
    log.info("СРАВНЕНИЕ МОДЕЛЕЙ")
    log.info("=" * 60)

    y_te = d["y_te"]
    met_table = {k: v["metrics"] for k, v in results.items() if v}
    print_metrics(met_table, "СВОДНАЯ ТАБЛИЦА МЕТРИК")

    df_m = pd.DataFrame(met_table).T
    df_m.to_csv(f"{CFG['output_dir']}/model_comparison.csv")
    print("\n" + df_m.to_string())

    _roc_curves(results, y_te)
    _pr_curves(results, y_te)
    _confusion_matrices(results, y_te)
    _bar_comparison(df_m)
    return df_m


def _roc_curves(results: dict, y_te):
    fig, ax = plt.subplots(figsize=(12, 8))
    pal = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
    for (name, res), col in zip(results.items(), pal):
        if not res or "scores" not in res:
            continue
        y = res.get("y_te", y_te)
        s = res["scores"]
        if len(y) != len(s) or y.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y, s)
        auc = roc_auc_score(y, s)
        ax.plot(fpr, tpr, color=col, lw=2, label=f"{name} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set(title="ROC-кривые", xlabel="FPR", ylabel="TPR", xlim=[0, 1], ylim=[0, 1.02])
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout(); savefig("11_roc_curves")


def _pr_curves(results: dict, y_te):
    fig, ax = plt.subplots(figsize=(12, 8))
    pal = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12", "#9b59b6", "#1abc9c", "#e67e22"]
    baseline = y_te.mean()
    for (name, res), col in zip(results.items(), pal):
        if not res or "scores" not in res:
            continue
        y = res.get("y_te", y_te)
        s = res["scores"]
        if len(y) != len(s) or y.sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y, s)
        ap = average_precision_score(y, s)
        ax.plot(rec, prec, color=col, lw=2, label=f"{name} (AP={ap:.3f})")

    ax.axhline(baseline, color="gray", ls="--", lw=1, label=f"Baseline={baseline:.4f}")
    ax.set(title="Кривые Precision–Recall", xlabel="Recall", ylabel="Precision")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout(); savefig("12_pr_curves")


def _confusion_matrices(results: dict, y_te):
    valid = {k: v for k, v in results.items() if v and "predictions" in v}
    n = len(valid)
    if n == 0:
        return

    nc, nr = min(3, n), (n + 2) // 3
    fig, axes = plt.subplots(nr, nc, figsize=(6 * nc, 5 * nr))
    axes_flat = np.array(axes).flatten() if n > 1 else [axes]

    for idx, (name, res) in enumerate(valid.items()):
        y = res.get("y_te", y_te)
        p = res["predictions"]
        if len(y) != len(p):
            continue
        cm = confusion_matrix(y, p)
        m  = res["metrics"]
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["Норма", "Мошен."],
                    yticklabels=["Норма", "Мошен."],
                    ax=axes_flat[idx], cbar=False)
        axes_flat[idx].set(
            title=f"{name}\nF1={m.get('F1',0):.3f}  P={m.get('Precision',0):.3f}  R={m.get('Recall',0):.3f}",
            xlabel="Предсказано", ylabel="Истина",
        )

    for ax in axes_flat[n:]:
        ax.set_visible(False)

    plt.suptitle("Матрицы ошибок", fontsize=14, fontweight="bold")
    plt.tight_layout(); savefig("13_confusion_matrices")


def _bar_comparison(df_m: pd.DataFrame):
    metrics = [c for c in ["Precision", "Recall", "F1", "ROC-AUC"] if c in df_m.columns]
    if not metrics:
        return
    nc = len(metrics)
    fig, axes = plt.subplots(1, nc, figsize=(6 * nc, 7))
    if nc == 1:
        axes = [axes]
    for ax, met in zip(axes, metrics):
        vals = df_m[met].sort_values(ascending=False)
        bars = ax.barh(vals.index, vals.values, color=plt.cm.RdYlGn(vals.values))
        for bar, v in zip(bars, vals.values):
            ax.text(v + 0.01, bar.get_y() + bar.get_height() / 2,
                    f"{v:.3f}", va="center", fontsize=10)
        ax.set(xlabel=met, title=f"Сравнение: {met}", xlim=[0, 1.15])
        ax.grid(alpha=0.3, axis="x")
    plt.suptitle("Сравнительный анализ моделей", fontsize=14, fontweight="bold")
    plt.tight_layout(); savefig("14_metrics_comparison")


# ══════════════════════════════════════════════════════════════════════════════
#   ЧАСТЬ 8.  ВАЖНОСТЬ ПРИЗНАКОВ
# ══════════════════════════════════════════════════════════════════════════════

def feature_importance(results: dict, d: dict):
    log.info("=" * 60)
    log.info("ВАЖНОСТЬ ПРИЗНАКОВ")
    log.info("=" * 60)

    for name in ["XGBoost", "Random Forest", "Gradient Boosting"]:
        res = results.get(name) or results.get("GBM")
        if res and "model" in res:
            m   = res["model"]
            imp = m.feature_importances_
            df_imp = pd.DataFrame(dict(Feature=d["feat_cols"], Importance=imp))\
                       .sort_values("Importance", ascending=False)

            fig, ax = plt.subplots(figsize=(12, 8))
            top = df_imp.head(20)
            ax.barh(top["Feature"], top["Importance"],
                    color=plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(top)))[::-1])
            ax.invert_yaxis()
            ax.set(title=f"Важность признаков — {name}",
                   xlabel="Importance (gain)")
            ax.grid(alpha=0.3, axis="x")
            plt.tight_layout(); savefig("15_feature_importance")

            print(f"\nТоп-15 признаков ({name}):")
            print(df_imp.head(15).to_string(index=False))
            df_imp.to_csv(f"{CFG['output_dir']}/feature_importance.csv", index=False)
            break


# ══════════════════════════════════════════════════════════════════════════════
#   ЧАСТЬ 9.  КРОСС-ВАЛИДАЦИЯ НА СКОЛЬЗЯЩЕМ ОКНЕ
# ══════════════════════════════════════════════════════════════════════════════

def sliding_window_cv(d: dict) -> dict:
    log.info("=" * 60)
    log.info("КРОСС-ВАЛИДАЦИЯ НА СКОЛЬЗЯЩЕМ ОКНЕ (TimeSeriesSplit, n=5)")
    log.info("=" * 60)

    X, y = d["X_tr"], d["y_tr"]
    tscv = TimeSeriesSplit(n_splits=5)
    cv_results = {}

    models_cv = {
        "Isolation Forest (CV)": lambda: IsolationForest(
            n_estimators=100, contamination=y.mean(), random_state=SEED),
        "Random Forest (CV)": lambda: RandomForestClassifier(
            n_estimators=100, class_weight="balanced",
            max_depth=8, random_state=SEED, n_jobs=-1),
    }

    for mname, mfactory in models_cv.items():
        f1s, aucs = [], []
        for fold, (tr_idx, v_idx) in enumerate(tscv.split(X)):
            Xtr, Xv = X[tr_idx], X[v_idx]
            ytr, yv = y[tr_idx], y[v_idx]
            m = mfactory()

            if "Isolation" in mname:
                X_ok = Xtr[ytr == 0]
                m.fit(X_ok)
                scores = -m.score_samples(Xv)
                thr    = np.percentile(-m.score_samples(X_ok), 97)
                preds  = (scores > thr).astype(int)
            else:
                m.fit(Xtr, ytr)
                preds  = m.predict(Xv)
                scores = m.predict_proba(Xv)[:, 1]

            if yv.sum() > 0:
                f1  = f1_score(yv, preds, zero_division=0)
                auc = roc_auc_score(yv, scores)
                f1s.append(f1); aucs.append(auc)
                print(f"  {mname} Fold {fold+1}: F1={f1:.4f}  AUC={auc:.4f}")

        cv_results[mname] = dict(
            F1_mean=np.mean(f1s),  F1_std=np.std(f1s),
            AUC_mean=np.mean(aucs), AUC_std=np.std(aucs),
        )

    print("\nИтоговая CV-таблица:")
    for n, r in cv_results.items():
        print(f"  {n}: F1={r['F1_mean']:.4f}±{r['F1_std']:.4f}  "
              f"AUC={r['AUC_mean']:.4f}±{r['AUC_std']:.4f}")
    return cv_results


# ══════════════════════════════════════════════════════════════════════════════
#   ГЛАВНАЯ ФУНКЦИЯ
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 72)
    print("  ВЫЯВЛЕНИЕ МОШЕННИЧЕСКИХ ОПЕРАЦИЙ С КРЕДИТНЫМИ КАРТАМИ")
    print("  нейросети + методы одноклассовой классификации | БГУИР 2025-26")
    print("=" * 72)
    print(f"  Начало: {datetime.now():%Y-%m-%d %H:%M:%S}\n")

    # 1. Загрузка данных
    df = load_data()

    # 2. EDA
    perform_eda(df)

    # 3. Временные ряды и признаки
    ts, df = create_time_series(df)
    df_feat = feature_engineering(df)

    # 4. Статистические методы (ARIMA + GARCH)
    stat_results = analyze_time_series(ts)

    # 5. Подготовка ML-данных
    d = prepare_ml(df_feat)

    # 6. Обучение моделей
    print("\n" + "=" * 60)
    print("МЕТОДЫ ИАД (одноклассовая классификация)")
    print("=" * 60)
    results = {}
    results["One-Class SVM"]    = train_ocsvm(d)
    results["Isolation Forest"] = train_iforest(d)
    results["LOF"]              = train_lof(d)

    print("\n" + "=" * 60)
    print("SUPERVISED BASELINE (для сравнения)")
    print("=" * 60)
    gbm_res = train_gbm(d)
    results[gbm_res.get("name", "GBM")] = gbm_res
    results["Random Forest"]    = train_rf(d)

    print("\n" + "=" * 60)
    print("НЕЙРОННЫЕ СЕТИ")
    print("=" * 60)
    ae_res   = train_autoencoder(d)
    lstm_res = train_lstm_autoencoder(d)
    if ae_res:
        results["Autoencoder"]      = ae_res
    if lstm_res:
        results["LSTM Autoencoder"] = lstm_res

    # 7. Оценка и сравнение
    df_metrics = evaluate_all(results, d)

    # 8. Важность признаков
    feature_importance(results, d)

    # 9. Кросс-валидация
    cv_results = sliding_window_cv(d)

    # 10. Итоговый отчёт
    print("\n" + "=" * 72)
    print("ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 72)
    if not df_metrics.empty and "F1" in df_metrics.columns:
        best_f1 = df_metrics["F1"].idxmax()
        print(f"\n  Лучшая модель по F1:      {best_f1}  "
              f"(F1={df_metrics.loc[best_f1,'F1']:.4f})")
    if not df_metrics.empty and "ROC-AUC" in df_metrics.columns:
        best_auc = df_metrics["ROC-AUC"].idxmax()
        print(f"  Лучшая модель по ROC-AUC: {best_auc}  "
              f"(AUC={df_metrics.loc[best_auc,'ROC-AUC']:.4f})")
    print(f"\n  Все графики → {CFG['output_dir']}/")
    print(f"  Сводная таблица → {CFG['output_dir']}/model_comparison.csv")
    print(f"\n  Завершено: {datetime.now():%Y-%m-%d %H:%M:%S}")

    return dict(
        df=df, ts=ts, d=d,
        stat_results=stat_results,
        results=results,
        df_metrics=df_metrics,
        cv_results=cv_results,
    )


if __name__ == "__main__":
    main()
