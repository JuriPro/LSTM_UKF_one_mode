import os
import random
import pickle
import datetime
import math
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer
from typing import Tuple, Dict, Optional, List, Any
import warnings

# Evaluate
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import time
# Метрики качества
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score
)

import logging
logger = logging.getLogger(__name__)

# Предполагается, что эти модули реализованы
from Decomposition import get_emd_components
from volatility import yang_zhang, rogers_satchell, hodges_tompkins, garman_klass, parkinson

# =================================================================
# КЛАССЫ ДИФФЕРЕНЦИРУЕМОГО UKF
# =================================================================
class SpectralCovarianceParam(tf.Module):
    """SPD-гарантированная параметризация ковариационных матриц для скалярного случая (state_dim=1)"""
    def __init__(self, name=None):
        super().__init__(name=name)
        # Инициализируем логарифм дисперсии для скалярного случая
        # Для state_dim=1 используем скаляр
        self.d_raw = tf.Variable(
            tf.math.log(0.1),  # скаляр
            trainable=True,
            name='d_raw'
        )
        self._min_eigenvalue = 0.01  # Снижаем минимальное значение для гибкости
        self._max_eigenvalue = 8.0   # Увеличиваем максимальное значение в 2 раза для адаптации к волатильности

    def get_P_and_sqrt(self):
        """Вычисляет P и sqrt(P) для скалярного случая (state_dim=1)"""
        # Применяем softplus для гарантии положительности
        d_pos = tf.nn.softplus(self.d_raw) + self._min_eigenvalue
        # Обрезаем для численной стабильности
        d_pos = tf.clip_by_value(d_pos, self._min_eigenvalue, self._max_eigenvalue)
        # Для скалярного случая ковариационная матрица - просто число
        P = tf.reshape(d_pos, [1, 1])
        P_sqrt = tf.reshape(tf.sqrt(d_pos), [1, 1])
        return P, P_sqrt

    def get_spectrum_info(self):
        """Получить информацию о спектре для скалярного случая (state_dim=1)"""
        d_pos = tf.nn.softplus(self.d_raw) + self._min_eigenvalue
        d_pos = tf.clip_by_value(d_pos, self._min_eigenvalue, self._max_eigenvalue)
        return {
            'min_eigenvalue': d_pos,
            'max_eigenvalue': d_pos,
            'condition_number': tf.constant(1.0, dtype=tf.float32)
        }


class ImplicitKalmanUpdate(tf.Module):
    """Численно стабильное обновление фильтра с Joseph формой"""

    def __init__(self, name=None):
        super().__init__(name=name)

    @tf.function
    def update(self, x_pred, P_pred, z, R):
        """Численно стабильное обновление Kalman фильтра - оптимизировано для state_dim=1"""
        batch_size = tf.shape(x_pred)[0]

        # ========== СКАЛЯРНЫЙ СЛУЧАЙ (state_dim=1, meas_dim=1) ==========
        # Простые арифметические операции вместо матричных

        P_pred_scalar = tf.reshape(P_pred[:, 0, 0], [batch_size])
        R_scalar = tf.reshape(R[:, 0, 0], [batch_size])
        z_scalar = tf.reshape(z[:, 0], [batch_size])
        x_pred_scalar = tf.reshape(x_pred[:, 0], [batch_size])

        # Коэффициент Kalman
        K_scalar = P_pred_scalar / (P_pred_scalar + R_scalar + 1e-8)
        K = tf.reshape(K_scalar, [batch_size, 1, 1])

        # Инновация (остаток)
        innov_scalar = z_scalar - x_pred_scalar  # [batch_size]
        innov = tf.reshape(innov_scalar, [batch_size, 1, 1])

        # Обновленное состояние
        x_upd = x_pred + tf.reshape(K_scalar * innov_scalar, [batch_size, 1])

        # Joseph форма ковариации (для скаляра очень простая)
        P_upd_scalar = (1 - K_scalar) * P_pred_scalar * (1 - K_scalar) + K_scalar * R_scalar * K_scalar
        P_upd = tf.reshape(P_upd_scalar, [batch_size, 1, 1])

        return x_upd, P_upd, innov, K

class MinMaxClip(tf.keras.constraints.Constraint):
    """Кастомный constraint для ограничения веса в диапазоне [min_value, max_value]"""
    def __init__(self, min_value=0.0, max_value=1.0):
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {'min_value': self.min_value, 'max_value': self.max_value}

class VolatilityRegimeSelector(tf.keras.layers.Layer):
    def __init__(self, num_regimes=3, history_window=100, learnable_centers=True, name="volatility_regime_selector"):
        super(VolatilityRegimeSelector, self).__init__(name=name)

        self.num_regimes = num_regimes
        self.history_window = history_window
        self.learnable_centers = learnable_centers

        # === ИНИЦИАЛИЗАЦИЯ base_centers ===
        self.base_centers = tf.constant([0.1, 0.3, 0.6], dtype=tf.float32)

        # История волатильности
        self._vol_history = tf.Variable(
            tf.zeros([1, history_window], dtype=tf.float32),
            trainable=False,
            name='volatility_history'
        )

        if self.learnable_centers:
            initial_centers = [0.1, 0.3, 0.6]
            self.center_logits = self.add_weight(
                name="center_logits",
                shape=(num_regimes,),
                initializer=lambda shape, dtype: tf.constant(initial_centers, dtype=dtype),
                trainable=True,
                dtype=tf.float32
            )
        else:
            self.center_logits = None

        # --- ИСПРАВЛЕНО: Кастомный MinMaxClip ---
        self.temperature = self.add_weight(
            name="temperature",
            shape=(),
            initializer="ones",
            trainable=True,
            dtype=tf.float32,
            constraint=MinMaxClip(min_value=0.3, max_value=10.0)  # ✅ ↓min, ↑max
        )

        # Масштабы режимов
        self.regime_scales = self.add_weight(
            name="regime_scales",
            shape=(num_regimes,),
            initializer="random_normal",
            trainable=True,
            dtype=tf.float32,
            constraint=tf.keras.constraints.NonNeg()  # ✅ Существует
        )

        # Инициализация начальных значений regime_scales
        initial_scales = [1.0, 1.5, 2.0]
        self.regime_scales.assign(initial_scales)

        self.target_width_logits = tf.Variable(
            initial_value=tf.zeros(num_regimes),
            trainable=True,
            dtype=tf.float32,
            name='target_width_logits'
        )

    def get_config(self):
        config = super().get_config()
        config.update({
            "num_regimes": self.num_regimes,
            "history_window": self.history_window,
            "learnable_centers": self.learnable_centers,
        })
        return config

    @tf.function
    def update_history(self, vol_current: tf.Tensor) -> None:
        """Обновляет историю волатильности.

        Args:
            vol_current: [B] — текущие значения волатильности для каждого элемента батча
        """
        # Усредняем по батчу → получаем одно значение
        vol_mean = tf.reduce_mean(vol_current)  # скаляр (в пределах графа)
        new_val = tf.reshape(vol_mean, [1, 1])   # [[mean]] → [1, 1]

        # Получаем текущую историю
        old_hist = self._vol_history.value()     # [1, history_window]

        # Сдвигаем окно: удаляем первый элемент, добавляем новый в конец
        updated = tf.concat([
            old_hist[:, 1:],      # все кроме первого → [1, H-1]
            new_val               # новый элемент → [1, 1]
        ], axis=1)                # → [1, H]

        # Присваиваем обратно
        self._vol_history.assign(updated)

    def get_centers(self) -> tf.Tensor:
        """Вычисляет адаптивные центры режимов волатильности.
        - Использует сортировку вместо tf.quantile (без зависимости от tensorflow-probability)
        - Требует минимум 20 валидных точек
        - Принудительно разводит центры на min_sep=0.03
        """
        if not hasattr(self, '_vol_history') or self._vol_history is None:
            return self.base_centers

        hist_t = self._vol_history.value()
        flat = tf.reshape(hist_t, [-1])  # [H]
        valid_mask = flat > 0.0
        valid = tf.boolean_mask(flat, valid_mask)  # [Hv]
        n = tf.shape(valid)[0]

        # Проверка: достаточно ли данных?
        min_valid = 20
        use_fallback = tf.less(n, min_valid)
        fallback_centers = tf.constant([0.1, 0.3, 0.6], dtype=tf.float32)

        def _quantile(x, q):
            """Расчёт квантиля порядка q (0-100) для 1D тензора x"""
            x_sorted = tf.sort(x)  # по возрастанию
            n_x = tf.shape(x_sorted)[0]
            # Индекс: (q/100) * (n-1), округлён до ближайшего int
            index_float = (tf.cast(q, tf.float32) / 100.0) * tf.cast(n_x - 1, tf.float32)
            index = tf.cast(tf.round(index_float), tf.int32)
            # Защита от выхода за границы
            index = tf.clip_by_value(index, 0, n_x - 1)
            return x_sorted[index]

        def compute_adaptive():
            q10 = _quantile(valid, 10.0)
            q50 = _quantile(valid, 50.0)
            q90 = _quantile(valid, 90.0)
            centers = tf.stack([q10, q50, q90])

            # Принудительное разведение центров (min_sep = 0.03)
            min_sep = 0.03
            for i in range(1, 3):
                gap = centers[i] - centers[i-1]
                correction = tf.nn.relu(min_sep - gap)
                indices = [[i]]
                updates = [correction]
                centers = tf.tensor_scatter_nd_add(centers, indices, updates)
            return centers

        adaptive_centers = tf.cond(
            use_fallback,
            lambda: fallback_centers,
            compute_adaptive
        )

        # Базовые центры (фиксированные или learnable)
        base_centers = self.base_centers
        if self.learnable_centers:
            center_logits = self.center_logits
            centers_raw = tf.exp(center_logits)
        else:
            centers_raw = base_centers

        # Смешивание базовых и адаптивных центров
        alpha = 0.7  # Вес адаптивной компоненты
        mixed_centers = alpha * adaptive_centers + (1 - alpha) * centers_raw

        return mixed_centers

    def get_center_separation_loss(self) -> tf.Tensor:
        """Штраф за близость центров режимов.
        - Минимальное расстояние между соседними центрами: 0.15
        - Минимальное расстояние между крайними: 0.30
        """
        centers = self.get_centers()
        min_center_dist = 0.15
        min_extreme_dist = 0.30

        # Расстояния между соседними центрами
        center_dists = centers[1:] - centers[:-1]  # [LOW-MID, MID-HIGH]
        dist_loss = tf.reduce_mean(tf.nn.relu(min_center_dist - center_dists))

        # Расстояние между LOW и HIGH
        extreme_dist = centers[2] - centers[0]
        extreme_loss = tf.nn.relu(min_extreme_dist - extreme_dist)

        return dist_loss * 20.0 + extreme_loss * 10.0

    @tf.function
    def assign_soft_regimes(self, vol_current: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Мягкое распределение текущей волатильности по режимам.

        ПАТЧ:
        - возвращаем logits (для supervised CE по regime_labels_batch)
        - добавляем per-regime scale (regime_scales), чтобы границы реально обучались
        - делаем вычисление устойчивым (eps, клипы, stable entropy)
        """
        vol_current = tf.cast(vol_current, tf.float32)               # [B]
        vol_current = tf.reshape(vol_current, [-1])                  # [B]
        batch_size = tf.shape(vol_current)[0]

        # Centers: [K]
        centers = tf.cast(self.get_centers(), tf.float32)            # [K]
        centers = tf.reshape(centers, [1, self.num_regimes])         # [1,K]

        # Per-regime scales: [K] (обучаемые; должны существовать как tf.Variable)
        # ВАЖНО: чтобы scale был >0 и не коллапсировал, используем softplus + floor.
        if hasattr(self, "regime_scales"):
            scales = tf.nn.softplus(tf.cast(self.regime_scales, tf.float32)) + 1e-3  # [K]
        else:
            # fallback: одинаковые масштабы
            scales = tf.ones([self.num_regimes], dtype=tf.float32)
        scales = tf.reshape(scales, [1, self.num_regimes])           # [1,K]

        # Temperature (global): scalar
        temp = tf.cast(self.temperature, tf.float32)
        temp = temp + 1e-6

        # Distances: [B,1] -> [B,K]
        vol_exp = tf.expand_dims(vol_current, axis=1)                # [B,1]
        # Используем квадратичное расстояние + асимметричные масштабы
        distances_sq = tf.square(vol_exp - centers)  # [B,K]

        # Асимметричные масштабы для каждого режима
        if hasattr(self, "regime_scales_left") and hasattr(self, "regime_scales_right"):
            # Разные масштабы слева и справа от центра
            scales_left = tf.nn.softplus(self.regime_scales_left) + 1e-3
            scales_right = tf.nn.softplus(self.regime_scales_right) + 1e-3
            scales = tf.where(vol_exp < centers, scales_left, scales_right)
        else:
            scales = tf.nn.softplus(self.regime_scales) + 1e-3

        logits = -distances_sq / (2.0 * tf.square(scales) * temp)
        soft_weights = tf.nn.softmax(logits, axis=1)

        # Hard assignment
        regime_assignment = tf.argmax(soft_weights, axis=1, output_type=tf.int32)  # [B]

        # Entropy (stable)
        p = tf.clip_by_value(soft_weights, 1e-8, 1.0)
        entropy = -tf.reduce_sum(p * tf.math.log(p), axis=1)         # [B]

        return {
            "logits": logits,                        # [B,K]  <-- для CE
            "soft_weights": soft_weights,            # [B,K]
            "regime_assignment": regime_assignment,  # [B]
            "entropy": entropy,                      # [B]
            "centers": tf.reshape(centers, [self.num_regimes]),      # [K] удобно для дебага
            "scales": tf.reshape(scales, [self.num_regimes]),        # [K] удобно для дебага
            "temperature": temp                      # scalar (тензор)
        }

    @tf.function
    def get_regime_scales(self, soft_weights: tf.Tensor) -> tf.Tensor:
        """
        Получить адаптивные масштабы CI на основе мягкого распределения.

        Args:
            soft_weights: [B, num_regimes] - результат assign_soft_regimes

        Returns:
            regime_scale: [B, 1] - масштаб CI для каждого элемента батча
        """
        # Взвешенное усреднение масштабов режимов
        # soft_weights: [B, num_regimes]
        # regime_scales: [num_regimes]

        base_target_width = 1.0  # базовое значение, можно сделать обучаемым
        width_scales = tf.sigmoid(self.target_width_logits) * 2.0 + 0.5  # [0.5, 2.5]
        target_width_per_regime = base_target_width * width_scales

        # Получить адаптивные масштабы CI на основе мягкого распределения
        regime_scale = tf.matmul(soft_weights, tf.expand_dims(width_scales, axis=1))
        # результат: [B, 1]

        # Нормировка regime_scale
        # ⚠️ ПРОБЛЕМА: Деление на среднее по батчу сбрасывает масштаб!
        # regime_scale = regime_scale / (tf.reduce_mean(regime_scale) + 1e-8)

        return regime_scale  # [B, 1]

    def get_spectrum_info(self) -> Dict[str, tf.Tensor]:
        """
        Информация о параметрах режимов для мониторинга.
        🔑 ИСПРАВЛЕНО: безопасный вызов get_centers() без @tf.function проблем
        """
        # 🔑 БЕЗОПАСНОЕ ПОЛУЧЕНИЕ ЦЕНТРОВ (без вызова @tf.function из Python)
        if self.learnable_centers:
            centers_val = tf.nn.softplus(self.center_logits).numpy()
        else:
            centers_val = self.centers.numpy()

        return {
            'regime_scales': self.regime_scales,
            'centers': centers_val,  # ← numpy array, не tf.Tensor
            'temperature': self.temperature
        }

    @tf.function
    def get_regime_entropy_loss(self, soft_weights: tf.Tensor) -> tf.Tensor:
        """
        ГИБРИДНЫЙ ПОДХОД: Штраф за коллапс режимов волатильности.
        ✅ ИСПРАВЛЕНО: Batch-size нормализация, стабильность энтропии
        """
        # Энтропия: H = -Σ(p × log(p))
        p_clipped = tf.clip_by_value(soft_weights, 1e-8, 1.0)
        entropy = -tf.reduce_sum(
            p_clipped * tf.math.log(p_clipped + 1e-8),
            axis=1
        )  # [B]
        
        # Максимальная энтропия для 3 режимов
        max_entropy = tf.math.log(tf.cast(self.num_regimes, tf.float32))  # ≈1.099
        
        # Нормализованная энтропия [0, 1]
        normalized_entropy = entropy / (max_entropy + 1e-8)
        
        # Целевая энтропия: 85% от максимума
        target_entropy = 0.85
        
        # Штрафуем только если энтропия ниже целевой
        entropy_loss = tf.reduce_mean(
            tf.nn.relu(target_entropy - normalized_entropy)
        )
        
        # Дополнительно: штраф за доминирование одного режима (>70%)
        max_regime_weight = tf.reduce_max(tf.reduce_mean(soft_weights, axis=0))
        dominance_penalty = tf.nn.relu(max_regime_weight - 0.70)
        
        # Дополнительно: штраф за пустые режимы (<10%)
        min_regime_weight = tf.reduce_min(tf.reduce_mean(soft_weights, axis=0))
        empty_penalty = tf.nn.relu(0.10 - min_regime_weight)
        
        # 🔑 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Нормализация на batch_size
        batch_size = tf.cast(tf.shape(soft_weights)[0], tf.float32)
        
        return (
            entropy_loss + 
            (5.0 / batch_size) * tf.square(dominance_penalty) + 
            (3.0 / batch_size) * tf.square(empty_penalty)
        )


class EntropyRegularizer(tf.Module):
    """
    Вычисляет энтропийную регуляризацию скрытых состояний LSTM

    Подход: Нормализуем выходы LSTM на [0, 1] и рассматриваем как
    вероятностное распределение, вычисляя его энтропию.
    """

    def __init__(self,
                 name: str = 'entropy_regularizer',
                 entropy_type: str = 'distribution',  # 'distribution' или 'spatial'
                 normalize_method: str = 'sigmoid'):  # 'sigmoid' или 'softmax'
        super().__init__(name=name)
        self.entropy_type = entropy_type
        self.normalize_method = normalize_method
        self._min_entropy_threshold = 0.1  # Не штрафуем уж совсем случайные h

    @tf.function
    def compute_entropy_loss(self,
                             lstm_hidden_states: tf.Tensor,
                             attention_output: tf.Tensor = None) -> tf.Tensor:
        """
        Вычисляет энтропийную потерю для скрытых состояний LSTM

        Args:
            lstm_hidden_states: [B, T, hidden_dim] - скрытые состояния LSTM
            attention_output: [B, T, hidden_dim] - опционально, выход attention

        Returns:
            entropy_loss: скалярная потеря
        """
        # Получаем размерности
        B = tf.shape(lstm_hidden_states)[0]
        T = tf.shape(lstm_hidden_states)[1]
        hidden_dim = tf.shape(lstm_hidden_states)[2]

        # Нормализуем h на [0, 1] как "вероятности"
        if self.normalize_method == 'sigmoid':
            # Сигмоид для поэлементной нормализации
            h_normalized = tf.nn.sigmoid(lstm_hidden_states)  # [B, T, hidden_dim]
        else:
            # Softmax по скрытому измерению
            h_normalized = tf.nn.softmax(lstm_hidden_states, axis=-1)  # [B, T, hidden_dim]

        # Обрезаем для численной стабильности
        h_normalized = tf.clip_by_value(h_normalized, 1e-8, 1.0 - 1e-8)

        # Вычисляем энтропию: H = -Σ(p × log(p))
        entropy_per_timestep = -tf.reduce_sum(
            h_normalized * tf.math.log(h_normalized + 1e-8),
            axis=-1  # Суммируем по скрытому измерению
        )  # [B, T]

        # Усредняем по времени и батчу
        mean_entropy = tf.reduce_mean(entropy_per_timestep)  # скаляр

        # ✅ ИСПРАВЛЕНО: Корректное вычисление максимальной энтропии
        if self.normalize_method == 'sigmoid':
            # Для sigmoid: максимальная энтропия = hidden_dim * ln(2)
            max_entropy = tf.cast(hidden_dim, tf.float32) * 0.69314718056  # ln(2) для каждого измерения
        else:
            # Для softmax: максимальная энтропия = ln(hidden_dim)
            max_entropy = tf.math.log(tf.cast(hidden_dim, tf.float32) + 1e-8)

        # ✅ ИСПРАВЛЕНО: Нормализуем энтропию для получения стабильных значений
        normalized_entropy = mean_entropy / (max_entropy + 1e-8)

        # ✅ ИСПРАВЛЕНО: Реалистичная целевая энтропия (50% от максимума)
        target_normalized = 0.5

        # ✅ ИСПРАВЛЕНО: Квадратичный штраф с адаптивным масштабированием
        entropy_deviation = tf.square(normalized_entropy - target_normalized)

        # ✅ ИСПРАВЛЕНО: Фиксированный коэффициент для контроля влияния
        scaling_factor = 0.01

        # ✅ ИСПРАВЛЕНО: Гарантированно положительный лосс
        entropy_loss = scaling_factor * entropy_deviation

        return entropy_loss

    @tf.function
    def compute_spatial_entropy(self,
                                lstm_hidden_states: tf.Tensor) -> tf.Tensor:
        """
        Альтернативный подход: энтропия по пространственному распределению h
        Штрафуем за слишком "спайковые" активации (все веса в одном нейроне)
        """
        # Нормализуем каждый timestep отдельно (Softmax по скрытому)
        h_softmax = tf.nn.softmax(lstm_hidden_states, axis=-1)  # [B, T, hidden_dim]
        h_softmax = tf.clip_by_value(h_softmax, 1e-8, 1.0)

        # Энтропия: H = -Σ(p × log(p)) по скрытому измерению
        entropy = -tf.reduce_sum(
            h_softmax * tf.math.log(h_softmax + 1e-8),
            axis=-1
        )  # [B, T]

        # Штрафуем за LOW entropy (концентрация весов)
        max_possible_entropy = tf.math.log(
            tf.cast(tf.shape(lstm_hidden_states)[-1], tf.float32)
        )

        normalized_entropy = entropy / (max_possible_entropy + 1e-8)  # [0, 1]
        loss = tf.reduce_mean(
            tf.nn.relu(0.5 - normalized_entropy)  # Штрафуем если < 0.5 от максимума
        )

        return loss

    def get_entropy_stats(self, lstm_hidden_states: tf.Tensor) -> dict:
        """Получить статистику энтропии для мониторинга"""
        h_normalized = tf.nn.sigmoid(lstm_hidden_states)
        h_normalized = tf.clip_by_value(h_normalized, 1e-8, 1.0 - 1e-8)

        entropy = -tf.reduce_sum(
            h_normalized * tf.math.log(h_normalized + 1e-8),
            axis=-1
        )

        return {
            'entropy_mean': tf.reduce_mean(entropy),
            'entropy_std': tf.math.reduce_std(entropy),
            'entropy_min': tf.reduce_min(entropy),
            'entropy_max': tf.reduce_max(entropy),
        }


class DifferentiableUKF(tf.Module):
    def __init__(self, state_dim=1, alpha=0.7, beta=2.0, kappa=2.0, name=None):
        super().__init__(name=name)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        # Компоненты дифференцируемого UKF
        self.spec_param = SpectralCovarianceParam(name='spectral_param')
        self.kalman_update = ImplicitKalmanUpdate(name='kalman_update')

        # Для мониторинга
        self.debug_mode = False

        self.min_P = 0.01
        self.max_P = 8.0

    def initialize(self, x0, P0_diag=None):
        """Инициализация состояния и ковариации"""
        if P0_diag is None:
            # Используем глобальные обучаемые начальные дисперсии
            if hasattr(self, 'global_model') and hasattr(self.global_model, 'log_initial_vars'):
                mode_idx = int(self.name.split('_')[-1])  # Предполагаем, что имя содержит индекс режима
                P0_diag = [tf.nn.softplus(self.global_model.log_initial_vars[mode_idx]).numpy() + 1e-6]
            else:
                P0_diag = [0.1] * self.state_dim

        # Инициализация спектральных параметров
        if isinstance(P0_diag, (list, tuple, np.ndarray)):
            P0_diag = P0_diag[0]  # Берем первое значение для скалярного случая
        # P0_diag — список, берем первый элемент
        P0_val = P0_diag[0] if isinstance(P0_diag, (list, tuple, np.ndarray)) else P0_diag
        d_init = tf.math.log(tf.maximum(P0_val, 1e-6))
        self.spec_param.d_raw.assign(d_init)  # скаляр
        return x0

    @tf.function
    def predict(self, x, P, Q, relax_factor, alpha_t, kappa_t):
        """
        Устойчивый predict для скалярного состояния (state_dim=1) при модели random-walk:
            x_pred = x
            P_pred = (relax_factor^2) * P + Q

        alpha_t/kappa_t оставлены в сигнатуре, чтобы не ломать вызовы,
        но в этом варианте не используются (так и должно быть без f()).
        """
        batch_size = tf.shape(x)[0]

        # Формы
        x = tf.ensure_shape(x, [None, 1])
        P = tf.ensure_shape(P, [None, 1, 1])
        Q = tf.ensure_shape(Q, [None, 1, 1])

        relax_factor = tf.reshape(relax_factor, [batch_size])
        relax_factor = tf.clip_by_value(relax_factor, 0.8, 1.2)

        # Извлекаем скаляры
        x_scalar = tf.squeeze(x, axis=-1)                 # [B]
        P_scalar = tf.squeeze(P, axis=[-2, -1])           # [B]
        Q_scalar = tf.squeeze(Q, axis=[-2, -1])           # [B]

        # Предсказание состояния (random-walk)
        x_pred_scalar = x_scalar                           # [B]

        # Предсказание ковариации
        P_pred_scalar = tf.square(relax_factor) * tf.maximum(P_scalar, 1e-8) + tf.maximum(Q_scalar, 1e-8)

        # Синхронизированные ограничения (как у вас)
        min_P = tf.cast(self.min_P, tf.float32)
        max_P = tf.cast(self.max_P, tf.float32)
        P_pred_scalar = tf.clip_by_value(P_pred_scalar, min_P, max_P)

        # Возврат форм
        x_pred = tf.expand_dims(x_pred_scalar, axis=-1)            # [B, 1]
        P_pred = tf.reshape(P_pred_scalar, [batch_size, 1, 1])     # [B, 1, 1]

        x_pred = tf.ensure_shape(x_pred, [None, 1])
        P_pred = tf.ensure_shape(P_pred, [None, 1, 1])
        return x_pred, P_pred


    @tf.function
    def update(self, x_pred, P_pred, z, R):
        """Этап обновления UKF (оптимизирован для state_dim=1, батч [B, ...])"""
        # Прямое извлечение (без reshape!)
        x_pred_scalar = x_pred[:, 0]  # [B]
        P_pred_scalar = P_pred[:, 0, 0]  # [B]
        z_scalar = z[:, 0]  # [B]
        R_scalar = R[:, 0, 0]  # [B]

        # Коэффициент Kalman
        K_scalar = P_pred_scalar / (P_pred_scalar + R_scalar + 1e-8)  # [B]

        # Инновация (остаток)
        innov_scalar = z_scalar - x_pred_scalar  # [B]

        # Обновленное состояние
        x_upd_scalar = x_pred_scalar + K_scalar * innov_scalar  # [B]
        x_upd = tf.reshape(x_upd_scalar, [-1, 1])  # [B, 1]

        # Joseph форма ковариации
        P_upd_scalar = (1 - K_scalar) * P_pred_scalar * (1 - K_scalar) + K_scalar * R_scalar * K_scalar
        P_upd = tf.reshape(P_upd_scalar, [-1, 1, 1])  # [B, 1, 1]

        # Формируем выходы в правильной форме
        innov = tf.reshape(innov_scalar, [-1, 1, 1])  # [B, 1, 1]
        K = tf.reshape(K_scalar, [-1, 1, 1])  # [B, 1, 1]

        return x_upd, P_upd, innov, K

    def get_spectrum_info(self):
        """Получить информацию о спектре для мониторинга"""
        return self.spec_param.get_spectrum_info()

@tf.function
def compute_adaptive_threshold(
    inflation_factors,
    current_volatility,
    threshold_ema_var,
    target_anomaly_ratio=0.20
    ):
    """
    Адаптирована для реального распределения данных с расширенными границами.
    """
    inflation_factors_flat = tf.reshape(inflation_factors, [-1])
    sorted_values = tf.sort(inflation_factors_flat)
    total_count = tf.cast(tf.shape(sorted_values)[0], tf.float32)

    # Используем более агрессивный подход к определению аномалий
    target_percentile = 25.0  # Более чувствительный порог

    percentile_position = (total_count - 1.0) * (target_percentile / 100.0)
    lower_idx = tf.cast(tf.math.floor(percentile_position), tf.int32)
    upper_idx = tf.cast(tf.math.ceil(percentile_position), tf.int32)
    lower_idx = tf.maximum(lower_idx, 0)
    upper_idx = tf.minimum(upper_idx, tf.cast(total_count, tf.int32) - 1)

    lower_val = sorted_values[lower_idx]
    upper_val = sorted_values[upper_idx]
    weight_upper = percentile_position - tf.floor(percentile_position)
    percentile_val = (1.0 - weight_upper) * lower_val + weight_upper * upper_val

    # Расширяем диапазон адаптации
    mean_inflation = tf.reduce_mean(inflation_factors_flat)
    std_inflation = tf.math.reduce_std(inflation_factors_flat) + 1e-8
    base_threshold = mean_inflation + std_inflation  # mean + 1*std
    volatility_factor = 0.5 * tf.nn.sigmoid(current_volatility - 0.3)
    vol_adjusted_threshold = base_threshold + tf.reduce_mean(volatility_factor)

    # Снижаем вес процентиля для более агрессивной адаптации
    dynamic_threshold = 0.7 * vol_adjusted_threshold + 0.3 * percentile_val
    dynamic_threshold = tf.clip_by_value(dynamic_threshold, mean_inflation, mean_inflation + 5.0 * std_inflation)

    all_inflation_anomalies = tf.greater(inflation_factors_flat, dynamic_threshold)
    anomaly_ratio = tf.reduce_mean(tf.cast(all_inflation_anomalies, tf.float32))

    def increase_threshold():
        adjustment_factor = 1.0 + 3.0 * tf.maximum(0.0, anomaly_ratio - target_anomaly_ratio)
        adjusted = dynamic_threshold * adjustment_factor
        return tf.clip_by_value(adjusted, mean_inflation, mean_inflation + 7.0 * std_inflation)

    def keep_threshold():
        return dynamic_threshold

    dynamic_threshold = tf.cond(
        anomaly_ratio > (target_anomaly_ratio * 1.5),  # Более чувствительный порог
        increase_threshold,
        keep_threshold
    )

    # EMA сглаживание с более агрессивным коэффициентом
    ema_factor = 0.8
    smoothed_threshold = ema_factor * threshold_ema_var + (1 - ema_factor) * dynamic_threshold
    threshold_ema_var.assign(smoothed_threshold)

    return smoothed_threshold, anomaly_ratio
# =================================================================
# ГИБРИДНАЯ LSTM-IMM-UKF МОДЕЛЬ С КОРРЕКТНЫМ УПРАВЛЕНИЕМ СОСТОЯНИЕМ
# =================================================================
class LSTMIMMUKF(tf.Module):
    """
    Гибридная модель LSTM-IMM-UKF для прогнозирования временных рядов
    с адаптивным переключением режимов.
    """
    def __init__(
        self,
        seq_len: int = 72,
        feature_columns: Optional[List[str]] = None,
        vol_window: int = 36,
        vol_window_long: int = 150,
        rolling_window_percentile: int = 100,
        emd_window: int = 350,
        min_history_for_features: int = 350,
        seed: int = 42,
        use_diff_ukf: bool = True,
        debug_mode = False,
        save_dir='./model_checkpoints'
    ):
        """
        Инициализация LSTM-UKF модели с контекстной волатильностью вместо IMM-режимов.
        """
        super().__init__()
        print("=" * 80)
        print("🚀 ИНИЦИАЛИЗАЦИЯ LSTM-UKF МОДЕЛИ С КОНТЕКСТНОЙ ВОЛАТИЛЬНОСТЬЮ")
        print("=" * 80)

        # Настройка воспроизводимости
        self._setup_reproducibility(seed=seed)
        print(f"✅ Воспроизводимость настроена с seed={seed}")

        # GPU/CPU настройки
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            print(f"✅ Обнаружено GPU устройств: {len(gpus)}")
            # policy = tf.keras.mixed_precision.Policy('mixed_float16')
            # tf.keras.mixed_precision.set_global_policy(policy)
            # print(f"✅ Mixed precision policy установлен: {policy.name}")
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            self._gpu_available = True
        else:
            print("⚠️ GPU не обнаружены, используем CPU")
            self._gpu_available = False

        self.device = '/GPU:0' if self._gpu_available else '/CPU:0'
        print(f'✅ Устройство для вычислений: {self.device}')

        # Дебаг режим
        self.debug_mode = debug_mode

        # Базовые параметры - ЕДИНСТВЕННЫЙ РЕЖИМ вместо 3
        self.state_dim = 1
        self.num_modes = 1  # КРИТИЧЕСКИЕ ИЗМЕНЕНИЯ: от IMM к единому режиму
        self.seq_len = seq_len
        self.vol_window_short = vol_window
        self.vol_window_long = vol_window_long
        self.rolling_window_percentile = rolling_window_percentile
        self.emd_window = emd_window
        self.min_history_for_features = min_history_for_features

        # Признаки
        if feature_columns is not None:
            self.feature_columns = feature_columns
        else:
            self.feature_columns = self._default_feature_columns()
        print(f"✅ Инициализированы {len(self.feature_columns)} признаков:")
        print(f"   {', '.join(self.feature_columns)}")

        # Режим UKF
        self.use_diff_ukf = use_diff_ukf
        print(f"🔧 UKF режим: {'DIFFERENTIABLE' if self.use_diff_ukf else 'STANDARD'}")

        # Инициализация дифференцируемого UKF (ОДИН вместо трёх)
        if self.use_diff_ukf:
            print("✅ Инициализация дифференцируемого UKF...")
            self.diff_ukf_component = DifferentiableUKF(
                state_dim=self.state_dim,
                alpha=0.7,
                kappa=2.0,
                name='diff_ukf_main'
            )
            self.diff_ukf_component.initialize(tf.zeros([self.state_dim]), P0_diag=[1.0])
            print("✅ Дифференцируемый UKF инициализирован для единого режима")

        # Инициализация Volatility Regime Selector (скорректировано под масштабирование)
        self.regime_selector = VolatilityRegimeSelector(
            num_regimes=3,
            history_window=100,
            learnable_centers=True,
            name='volatility_regime_selector'
        )
        # Синхронизированные параметры для целевого покрытия 90% (с учетом улучшенного алгоритма)
        # 🔑 ГИБРИДНЫЙ ПОДХОД v5: СОГЛАСОВАННЫЕ НАЧАЛЬНЫЕ ЗНАЧЕНИЯ regime_scales
        # Не перезаписываем значения из VolatilityRegimeSelector.__init__()
        # Они уже установлены в [2.96, 4.44, 6.16] что соответствует целевому покрытию 85-90%
        print(f"✅ Regime scales оставлены по умолчанию: {self.regime_selector.regime_scales.numpy()}")
        # Примечание: если нужно изменить — делайте это в VolatilityRegimeSelector.__init__()

        # Центроиды режимов адаптированы под реальную статистику волатильности
        self.regime_selector.center_logits.assign(tf.constant(
            np.log(np.array([0.12, 0.35, 0.75], dtype=np.float32)),
            dtype=tf.float32
        ))
        # Более резкая температура для четкого распределения
        self.regime_selector.temperature.assign(0.4)
        print("✅ Regime Selector инициализирован (LOW/MID/HIGH режимы)")

        # Модель будет инициализирована позже
        self.model = None
        self.feature_scalers = None
        self.y_scalers = None

        # === ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЙ UKF ДЛЯ ЕДИНОГО РЕЖИМА ===
        with tf.device(self.device):
            # Начальное состояние (для одного режима)
            self._last_state = tf.Variable(
                tf.zeros([1], dtype=tf.float32),  # ← Явно [1], а не [state_dim] которое тоже [1] но может быть интерпретировано как скаляр
                trainable=False,
                name='last_state',
                dtype=tf.float32
            )
            print(f"✅ _last_state: shape={self._last_state.shape}")

            # _last_P: фиксированная инициализация (не обучаемая), без base_q_logit/base_r_logit
            initial_P = tf.reshape(
                tf.eye(self.state_dim, dtype=tf.float32) * tf.constant(0.15, tf.float32),
                [1, self.state_dim, self.state_dim]
            )
            self._last_P = tf.Variable(
                initial_P,
                trainable=False,
                name='last_P',
                dtype=tf.float32
            )
            print(f"✅ _last_P: shape={self._last_P.shape} (единая форма [1, 1, 1])")

            self._last_volatility = tf.Variable(
                [0.1],
                trainable=False,
                dtype=tf.float32,
                name='last_volatility'
            )
            print(f"✅ _last_volatility: shape={self._last_volatility.shape} (инициализировано 0.1)")

            # Q ceiling used consistently in both explicit one-step prediction and UKF filtering
            self.Q_max_pred = tf.constant(5.0, dtype=tf.float32)   # старт для теста (потом 0.3/1.0)

            # === ДОБАВЛЕНО: ПАРАМЕТРЫ МАКСИМАЛЬНОЙ ШИРИНЫ ДОВЕРИТЕЛЬНОГО ИНТЕРВАЛА ПО РЕЖИМАМ ===
            self.max_width_factors_logits = tf.Variable(
                initial_value=tf.constant([0.5, 0.5, 1.0], dtype=tf.float32),  # → softplus(~0.65, 0.65, 1.3) → ~2.2x
                trainable=True,
                name="max_width_factors_logits",
                dtype=tf.float32
            )

        # --- ДОБАВЛЕНО: обучаемые цели покрытия для каждого режима ---
        initial_target_coverages = np.array([0.95, 0.90, 0.92], dtype=np.float32)  # Увеличиваем покрытие для HIGH режима
        logits_init = np.log(initial_target_coverages / (1.0 - initial_target_coverages))

        self.target_cov_logits = tf.Variable(
            initial_value=logits_init,
            trainable=True,
            dtype=tf.float32,
            name='target_cov_logits',
            synchronization=tf.VariableSynchronization.ON_READ,
            aggregation=tf.VariableAggregation.MEAN
        )

        # Добавляем инициализацию целевых ширин (как рекомендовалось ранее)
        initial_width_scales = np.array([0.8, 1.0, 1.3], dtype=np.float32)  # Увеличиваем ширину для высокого режима
        width_logits_init = np.log(initial_width_scales)
        self.target_width_logits = tf.Variable(
            initial_value=width_logits_init,
            trainable=True,
            dtype=tf.float32,
            name='target_width_logits'
        )

        # === ДОБАВЛЕНО: инициализация _prev_train_coverage
        self._prev_train_coverage = tf.Variable(
            initial_value=tf.constant(0.90, dtype=tf.float32),
            trainable=False,
            name="prev_train_coverage"
        )

        # === ДОБАВЛЕНО: инициализация coverage_mixing_alpha для смешивания стратегий покрытия ===
        self.coverage_mixing_alpha = tf.Variable(
            initial_value=tf.constant(0.405, dtype=tf.float32),  # log(0.6/(1-0.6))
            trainable=True,
            name='coverage_mixing_alpha'
        )

        # Инициализация оптимизатора
        # Опитимизатор инициализируется в методе fit()

        # Ранняя остановка
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_weights_dict = None
        self.best_scalers = None
        self.patience_counter = 0

        # Группы признаков для масштабирования
        self.scale_groups = {
            'robust': ['velocity', 'acceleration', 'energy', 'st_comp_diff',
                       'extreme_pos_momentum', 'tail_weight_indicator',
                       'log_vol_short', 'rel_vol_short_long', 'vol_accel_rel', 'rel_entropy'],
            'standard': ['amplitude', 'yz', 'gc', 'p', 'rs', 'ht'],
            'minmax': ['percentile_pos'],
            'none': ['asymmetry_ratio', 'percentile_pos_fisher', 'skew']
        }

        # Установка директории для сохранения
        self.save_dir = save_dir
        os.makedirs(self.save_dir, exist_ok=True)
        print(f"📁 Директория для сохранения модели: {os.path.abspath(self.save_dir)}")

        # === ИНИЦИАЛИЗАЦИЯ ЭНТРОПИЙНОГО РЕГУЛЯРИЗАТОРА ===
        print("✅ Инициализация энтропийного регуляризатора...")
        self.entropy_regularizer = EntropyRegularizer(
            name='lstm_entropy_regularizer',
            entropy_type='distribution',
            normalize_method='sigmoid'
        )

        # Гиперпараметр регуляризации (начальное значение)
        self.lambda_entropy = tf.Variable(
            0.03,  # увеличено до 2% от основной loss
            trainable=False,
            dtype=tf.float32,
            name='lambda_entropy'
        )
        print(f"✅ Энтропийный регуляризатор инициализирован с lambda={self.lambda_entropy.numpy():.4f}")

        # Инициализация EMA для адаптивного порога аномалий
        self.threshold_ema = tf.Variable(2.7, trainable=False, dtype=tf.float32)

        self.forecast_bias_correction = tf.Variable(
            initial_value=tf.zeros(3, dtype=tf.float32),  # [LOW, MID, HIGH]
            trainable=True,
            name='forecast_bias_correction'
        )
        print("✅ Добавлена обучаемая коррекция смещения прогноза по режимам")

        self._prev_val_coverage = tf.Variable(
            initial_value=tf.constant(0.89, dtype=tf.float32),  # ✅ Целевое покрытие 89%
            trainable=False,
            name="prev_val_coverage"
        )
        # ✅ ДОБАВЛЕНО: _prev_train_coverage для domain adaptation
        self._prev_train_coverage = tf.Variable(
            initial_value=tf.constant(0.89, dtype=tf.float32),
            trainable=False,
            name="prev_train_coverage"
        )
        
        print("=" * 80)
        print("✅ LSTM-UKF МОДЕЛЬ С КОНТЕКСТНОЙ ВОЛАТИЛЬНОСТЬЮ ПОЛНОСТЬЮ ИНИЦИАЛИЗИРОВАНА")
        print(f"⏰ Завершено: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

    def _setup_reproducibility(self, seed: int = 42):
        """Установка глобальных seeds для воспроизводимости."""
        os.environ['PYTHONHASHSEED'] = str(seed)
        os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        logger.info(f"✅ Reproducibility seed set to {seed}")

    def _default_feature_columns(self):
        """Обновленный список признаков с включением всех используемых компонентов"""
        return [
            'level', 'st_comp_diff', 'log_vol_short', 'skew', 'percentile_pos', 'percentile_pos_fisher',
            'energy', 'amplitude', 'yz', 'gc', 'p', 'rs', 'ht',
            'extreme_pos_momentum', 'asymmetry_ratio', 'tail_weight_indicator',
            'velocity', 'acceleration', 'rel_vol_short_long', 'vol_accel_rel', 'rel_entropy'
        ]

    def prepare_honest_datasets(
        self,
        full_df: Optional[pd.DataFrame] = None,
        cache_path: str = './cache/honest_prepared',  # ← Путь БЕЗ расширения .pkl
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        force_recompute: bool = False,
        n_jobs: int = -1,
        buffer_size: int = 50,
        block_size: int = 200,  # ← Используется при adaptive_blocks=False
        adaptive_blocks: bool = False,  # ← НОВЫЙ ПАРАМЕТР (по умолчанию выключен для обратной совместимости)
        min_regime_per_block: int = 3,  # ← Мин. окон каждого режима в адаптивном блоке
        max_block_size: int = 300       # ← Макс. размер адаптивного блока (защита)
    ) -> Tuple[Dict, Dict, Dict]:
        """
        УМНЫЙ ИНТЕРФЕЙС: автоматически проверяет кэш и загружает/готовит данные с поддержкой адаптивной стратификации.

        🔑 КЛЮЧЕВЫЕ РЕЖИМЫ РАБОТЫ:
        • Режим 1 (классический): adaptive_blocks=False → используется фиксированный block_size
        • Режим 2 (адаптивный):   adaptive_blocks=True  → блоки формируются динамически до достижения
                                  баланса режимов (минимум min_regime_per_block окон каждого режима)

        ⚠️ КРИТИЧЕСКИ ВАЖНО: СОГЛАСОВАННОСТЬ СОСТОЯНИЯ
        При загрузке из кэша применяется защита от конфликта состояний:
        • Если модель уже содержит обученные веса → НЕ перезаписываем скейлеры из кэша
        • Если модель НЕ обучена → синхронизируем скейлеры и квантили из кэша

        Важно: все пути передаются БЕЗ расширения .pkl — оно добавляется автоматически.

        Возвращает:
            (train_data, val_data, test_data) — словари с ключами:
            'X_seq_scaled', 'y_filter', 'y_target_scaled', 'timestamps', 'regime_labels'
        """
        import os
        from dataPreparator import HonestDataPreparator

        # === ШАГ 1: КОРРЕКТНАЯ ПРОВЕРКА КЭША (с расширением .pkl) ===
        cache_file_with_ext = f"{cache_path}.pkl"  # ← КРИТИЧЕСКИ ВАЖНО: проверяем именно .pkl файл

        if not force_recompute and os.path.exists(cache_file_with_ext):
            print(f"📥 Кэш найден: {cache_file_with_ext}")
            print("   Загружаем предварительно обработанные данные...")

            # Создаём подготовщик с ТЕМИ ЖЕ ПАРАМЕТРАМИ, что и при сохранении
            preparator = HonestDataPreparator(
                model=self,
                seq_len=self.seq_len,
                min_history_for_features=self.min_history_for_features,
                buffer_size=buffer_size,
                block_size=block_size,
                min_windows_per_regime=5,
                adaptive_blocks=adaptive_blocks,
                min_regime_per_block=min_regime_per_block,
                max_block_size=max_block_size,
                seed=42
            )

            # Загрузка данных из кэша
            train_data, val_data, test_data = preparator.load_prepared_datasets(cache_path)

            # 🔑 КРИТИЧЕСКАЯ ЗАЩИТА ОТ КОНФЛИКТА СОСТОЯНИЙ
            # Проверяем: содержит ли модель уже обученные веса?
            model_has_weights = (
                self.model is not None and
                hasattr(self.model, 'trainable_variables') and
                len(self.model.trainable_variables) > 0
            )

            # Проверяем: есть ли у модели уже инициализированные скейлеры?
            model_has_scalers = (
                hasattr(self, 'feature_scalers') and
                self.feature_scalers is not None and
                'Y' in self.feature_scalers
            )

            if model_has_weights and model_has_scalers:
                # ⚠️ Модель уже обучена — НЕ перезаписываем скейлеры из кэша!
                # Это предотвращает конфликт: веса из model.load() vs скейлеры из кэша
                print("⚠️  Модель уже содержит обученные веса и скейлеры.")
                print("   Загрузка данных из кэша БЕЗ перезаписи скейлеров (для сохранения согласованности).")
                print("   ⚠️ ВАЖНО: убедитесь, что кэш данных был подготовлен С ТЕМИ ЖЕ СКЕЙЛЕРАМИ,")
                print("            что использовались при обучении модели!")
            else:
                # ✅ Полная синхронизация скейлеров и квантилей из кэша
                # Скейлеры хранятся напрямую в self (preparator.model == self)
                if hasattr(self, 'feature_scalers') and self.feature_scalers is not None:
                    print(f"✅ Скейлеры синхронизированы из кэша: {list(self.feature_scalers.keys())}")
                else:
                    print("⚠️  Скейлеры отсутствуют в кэше! Будет использовано масштабирование по умолчанию.")

                # Синхронизация квантилей волатильности для онлайн-режима
                if hasattr(self, 'volatility_quantiles') and self.volatility_quantiles is not None:
                    print(f"✅ Квантили волатильности синхронизированы: "
                          f"q33={self.volatility_quantiles['q33']:.6f}, q67={self.volatility_quantiles['q67']:.6f}")
                else:
                    print("⚠️  Квантили волатильности отсутствуют в кэше! "
                          "Онлайн-прогнозирование может использовать несогласованную классификацию режимов.")

            # 🔑 КРИТИЧЕСКАЯ ВАЛИДАЦИЯ скейлера 'Y' (обязателен для обратного преобразования!)
            if not hasattr(self, 'feature_scalers') or self.feature_scalers is None or 'Y' not in self.feature_scalers:
                raise RuntimeError(
                    "❌ КРИТИЧЕСКАЯ ОШИБКА: скейлер 'Y' (целевая переменная) отсутствует!\n"
                    "   Без него невозможны обратные преобразования и интерпретация прогнозов.\n"
                    "   Решение:\n"
                    "   • Вариант A: пересохраните модель через model.save(path) с полным состоянием\n"
                    "   • Вариант B: выполните полную подготовку данных с force_recompute=True"
                )
            else:
                print("✅ Критический скейлер 'Y' для целевой переменной присутствует и валиден")

            # Сохраняем для внутреннего использования в fit()
            self._honest_preparation = {
                'train_data': train_data,
                'val_data': val_data,
                'test_data': test_data,
                'preparator': preparator,
                'cache_path': cache_path,
                'adaptive_blocks': adaptive_blocks,
                'block_size': block_size,
                'min_regime_per_block': min_regime_per_block,
                'max_block_size': max_block_size
            }

            # Информируем пользователя о режиме стратификации
            print(f"✅ Данные успешно загружены из кэша: {cache_file_with_ext}")
            print(f"   • Стратификация: {'АДАПТИВНАЯ' if adaptive_blocks else 'ФИКСИРОВАННАЯ'}")
            if adaptive_blocks:
                print(f"   • Мин. окон на режим в блоке: {min_regime_per_block}")
                print(f"   • Макс. размер блока: {max_block_size}")
            else:
                print(f"   • Размер блока: {block_size}")
            return train_data, val_data, test_data

        # === ШАГ 2: ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ ===
        if full_df is None:
            raise ValueError(
                "full_df обязателен при первом запуске (когда кэш не существует). "
                "При загрузке из кэша передайте только cache_path."
            )

        # === ШАГ 3: ПОЛНАЯ ПОДГОТОВКА ===
        print(f"⚠️  Кэш не найден или требуется пересчёт: {cache_file_with_ext}")
        print(f"   Запускаем полную честную подготовку данных (без утечки будущего)...")
        print(f"   • Стратификация: {'АДАПТИВНАЯ' if adaptive_blocks else 'ФИКСИРОВАННАЯ'}")

        preparator = HonestDataPreparator(
            model=self,
            seq_len=self.seq_len,
            min_history_for_features=self.min_history_for_features,
            buffer_size=buffer_size,
            block_size=block_size,
            min_windows_per_regime=5,
            adaptive_blocks=adaptive_blocks,
            min_regime_per_block=min_regime_per_block,
            max_block_size=max_block_size,
            seed=42
        )

        train_data, val_data, test_data = preparator.prepare_datasets(
            df=full_df,
            save_path=cache_path,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            n_jobs=n_jobs,
            force_recompute=force_recompute,
            use_adaptive=adaptive_blocks
        )

        # 🔑 КРИТИЧЕСКАЯ ВАЛИДАЦИЯ скейлера 'Y' ПОСЛЕ ПОДГОТОВКИ
        if not hasattr(self, 'feature_scalers') or self.feature_scalers is None or 'Y' not in self.feature_scalers:
            raise RuntimeError(
                "❌ КРИТИЧЕСКАЯ ОШИБКА: после подготовки данных отсутствует скейлер 'Y'!\n"
                "   Это указывает на внутреннюю ошибку в логике масштабирования.\n"
                "   Решение: проверьте реализацию HonestDataPreparator._scale_features_batch"
            )
        else:
            print("✅ Критический скейлер 'Y' для целевой переменной успешно инициализирован")

        # Сохраняем для внутреннего использования
        self._honest_preparation = {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'preparator': preparator,
            'cache_path': cache_path,
            'adaptive_blocks': adaptive_blocks,
            'block_size': block_size,
            'min_regime_per_block': min_regime_per_block,
            'max_block_size': max_block_size
        }

        print(f"✅ Данные подготовлены и сохранены в кэш: {cache_file_with_ext}")
        print(f"   • Стратификация: {'АДАПТИВНАЯ' if adaptive_blocks else 'ФИКСИРОВАННАЯ'}")
        if adaptive_blocks:
            print(f"   • Мин. окон на режим в блоке: {min_regime_per_block}")
            print(f"   • Макс. размер блока: {max_block_size}")
        else:
            print(f"   • Размер блока: {block_size}")

        return train_data, val_data, test_data

    def _build_model(self, input_shape: Tuple[int, int], training: bool = True) -> tf.keras.Model:
        """Архитектура LSTM с расширенным выходом для декомпозиции параметров"""
        l2_reg = tf.keras.regularizers.l2(5e-4)
        inputs = tf.keras.Input(shape=input_shape)

        # Первый LSTM слой
        lstm1 = tf.keras.layers.LSTM(
            256,
            recurrent_dropout=0.2,
            return_sequences=True,
            kernel_initializer='he_normal',
            recurrent_initializer='orthogonal',
            kernel_regularizer=l2_reg,
            recurrent_regularizer=l2_reg,
            name='lstm_layer_1'
        )
        h1 = lstm1(inputs, training=training)  # [B, T, 256]
        h1 = tf.keras.layers.LayerNormalization()(h1)
        h1 = tf.keras.layers.Dropout(0.4)(h1, training=training)

        # Второй LSTM слой
        lstm2 = tf.keras.layers.LSTM(
            128,
            recurrent_dropout=0.2,
            return_sequences=True,
            kernel_initializer='he_normal',
            recurrent_initializer='orthogonal',
            kernel_regularizer=l2_reg,
            recurrent_regularizer=l2_reg,
            name='lstm_layer_2'
        )
        h2 = lstm2(h1, training=training)  # [B, T, 128]
        h2 = tf.keras.layers.LayerNormalization()(h2)
        h2 = tf.keras.layers.Dropout(0.3)(h2, training=training)  # ↑ регуляризация активаций

        # Multi-Head Attention слой
        attention = tf.keras.layers.MultiHeadAttention(
            num_heads=4,
            key_dim=8,
            dropout=0.3,
            kernel_regularizer=l2_reg,
            name='attention_layer'
        )
        attention_output = attention(
            query=h2,
            key=h2,
            value=h2,
            training=training
        )

        # Остаточное соединение и нормализация
        h_attn = tf.keras.layers.Add()([h2, attention_output])
        h_attn = tf.keras.layers.LayerNormalization()(h_attn)
        h_attn = tf.keras.layers.Dropout(0.25)(h_attn, training=training)

        # Выходной слой для генерации параметров с поддержкой асимметричных границ (42 параметра)
        params = tf.keras.layers.Dense(
            37,
            activation=None,
            kernel_initializer='he_normal',
            bias_initializer='zeros',
            dtype='float32',
            kernel_regularizer=l2_reg,
            name='adaptive_params_layer'
        )(h_attn)

        # НОВОЕ: модель возвращает ВСЕ скрытые представления
        model = tf.keras.Model(inputs=inputs, outputs={
            'params': params,
            'h_lstm1': h1,      # [B, T, 256]
            'h_lstm2': h2,      # [B, T, 128]
            'h_attention': h_attn  # [B, T, 128]
        })

        return model

    def process_lstm_output(self, params_output):
        """Новая обработка выхода LSTM для адаптивной волатильности"""
        params_clipped = tf.clip_by_value(params_output, -8.0, 8.0)
        B = tf.shape(params_clipped)[0]
        T = tf.shape(params_clipped)[1]

        # === ИЗВЛЕЧЕНИЕ НОВЫХ ПАРАМЕТРОВ ===
        # ✅ ПРАВИЛЬНОЕ РАЗДЕЛЕНИЕ 36 ПАРАМЕТРОВ
        vol_context_params = params_clipped[..., :7]      # 7 параметров
        base_ukf_params = params_clipped[..., 7:19]       # 12 параметров
        inflation_params = params_clipped[..., 19:28]     # 9 параметров
        student_t_params = params_clipped[..., 28:37]     # 9 параметров

        # === ОБРАБОТКА ПАРАМЕТРОВ ===
        # 1. Параметры контекстной волатильности
        vol_context = {
            'sensitivity_short': tf.nn.softplus(vol_context_params[..., 0:1]) + 0.1,
            'sensitivity_medium': tf.nn.softplus(vol_context_params[..., 1:2]) + 0.1,
            'sensitivity_long': tf.nn.softplus(vol_context_params[..., 2:3]) + 0.1,
            'entropy_weight': tf.nn.sigmoid(vol_context_params[..., 3:4]),
            'accel_weight': tf.nn.sigmoid(vol_context_params[..., 4:5]),
            'memory_factor': 0.8 + 0.2 * tf.nn.sigmoid(vol_context_params[..., 5:6]),
            'leverage_effect_strength': 0.5 + 1.5 * tf.nn.sigmoid(vol_context_params[..., 6:7])
        }

        # 2. Базовые параметры UKF
        ukf_params = {
            'q_base': tf.nn.softplus(base_ukf_params[..., 0:1]) + 1e-6,
            'r_base': tf.clip_by_value(tf.nn.softplus(base_ukf_params[..., 1:2]) + 1e-6, 1e-6, 3.0),  # ← верхний лимит 3.0
            'relax_base': 0.8 + 0.7 * tf.nn.sigmoid(base_ukf_params[..., 2:3]),
            'alpha_base': 0.5 + 0.5 * tf.nn.sigmoid(base_ukf_params[..., 3:4]),
            'kappa_base': 1.0 + 1.5 * tf.nn.sigmoid(base_ukf_params[..., 4:5]),
            'q_sensitivity': tf.nn.softplus(base_ukf_params[..., 5:6]) + 0.1,
            'r_sensitivity': tf.clip_by_value(tf.nn.softplus(base_ukf_params[..., 6:7]) + 0.1, 0.1, 2.0),  # ← верхний лимит 2.0
            'relax_sensitivity': tf.nn.sigmoid(base_ukf_params[..., 7:8]),
            'alpha_sensitivity': tf.nn.sigmoid(base_ukf_params[..., 8:9]),
            'kappa_sensitivity': tf.nn.sigmoid(base_ukf_params[..., 9:10]),
            'q_floor': tf.nn.softplus(base_ukf_params[..., 10:11]) + 1e-8,
            'r_floor': tf.clip_by_value(tf.nn.softplus(base_ukf_params[..., 11:12]) + 1e-8, 1e-8, 1.5),  # ← верхний лимит 1.5
        }

        # 3. Параметры для adaptive inflation
        # ✅ ИСПРАВЛЕННАЯ ВЕРСИЯ ДЛЯ inflation_config
        inflation_config = {
            'base_inflation': 2.0 + 8.0 * tf.nn.sigmoid(inflation_params[..., 0:1]),
            'vol_sensitivity': tf.nn.sigmoid(inflation_params[..., 1:2]),
            'decay_rate': 0.3 + 0.5 * tf.nn.sigmoid(inflation_params[..., 2:3]),
            'anomaly_threshold': 2.5 + 1.0 * tf.nn.sigmoid(inflation_params[..., 3:4]),
            # ✅ ИСПРАВЛЕНО: Храним float версии для всех параметров
            'min_duration_float': 1.0 + 5.0 * tf.nn.sigmoid(inflation_params[..., 4:5]),
            'max_duration_float': 3.0 + 10.0 * tf.nn.sigmoid(inflation_params[..., 5:6]),
            'asymmetry_factor': 0.5 + 1.0 * tf.nn.sigmoid(inflation_params[..., 6:7]),
            'memory_decay': 0.7 + 0.3 * tf.nn.sigmoid(inflation_params[..., 7:8]),
            'inflation_limit': 20.0 * tf.nn.sigmoid(inflation_params[..., 8:9])
        }

        # 4. Параметры Student-t распределения с асимметричными границами (ИСПРАВЛЕНО: удалены дубликаты)
        student_t_config = {
            'dof_base': 3.0 + 8.0 * tf.nn.sigmoid(student_t_params[..., 0:1]),
            'dof_sensitivity': tf.nn.sigmoid(student_t_params[..., 1:2]),
            'asymmetry_pos': 0.5 + 1.5 * tf.nn.sigmoid(student_t_params[..., 2:3]),
            'asymmetry_neg': 0.5 + 1.5 * tf.nn.sigmoid(student_t_params[..., 3:4]),
            'calibration_sensitivity': tf.nn.sigmoid(student_t_params[..., 4:5]),
            'tail_weight_pos': 1.0 + 3.0 * tf.nn.sigmoid(student_t_params[..., 5:6]),
            'tail_weight_neg': 1.0 + 3.0 * tf.nn.sigmoid(student_t_params[..., 6:7]),
            'confidence_floor': 0.7 + 0.2 * tf.nn.sigmoid(student_t_params[..., 7:8]),
            'confidence_ceil': 0.95 + 0.04 * tf.nn.sigmoid(student_t_params[..., 8:9]),
        }

        # === НОВОЕ: РЕЖИМНАЯ КАЛИБРОВКА ДИ ===
        # Вычисляем текущую волатильность из vol_context
        # vol_context['sensitivity_short'] уже содержит влияние текущей волатильности
        vol_level = vol_context['sensitivity_short']  # [B, T, 1]
        vol_squeezed = tf.squeeze(vol_level, axis=-1)  # [B, T]

        # Для каждого элемента батча и шага времени: мягкое распределение по режимам
        B = tf.shape(vol_context['sensitivity_short'])[0]
        T = tf.shape(vol_context['sensitivity_short'])[1]

        # Процесс по времени: для каждого шага вычисляем soft-веса
        # vol_squeezed: [B, T] → нужно для каждого (b, t) вычислить soft_weights

        # Переформируем: [B, T] → [B*T]
        vol_flat = tf.reshape(vol_squeezed, [B * T])

        # Получаем мягкие веса для каждого элемента
        regime_info = self.regime_selector.assign_soft_regimes(vol_flat)
        soft_weights = regime_info['soft_weights']  # [B*T, num_regimes]

        # Получаем масштабы режимов
        regime_scale = self.regime_selector.get_regime_scales(soft_weights)  # [B*T, 1]

        # Переформируем обратно: [B*T, 1] → [B, T, 1]
        regime_scale = tf.reshape(regime_scale, [B, T, 1])

        # Добавляем масштаб в student_t_config
        student_t_config['regime_scale'] = regime_scale  # [B, T, 1]
        student_t_config['regime_soft_weights'] = tf.reshape(soft_weights, [B, T, self.regime_selector.num_regimes])

        return vol_context, ukf_params, inflation_config, student_t_config

    @tf.function
    def compute_adaptive_Q_R_with_leverage(
        self, innov_prev,
        leverage_strength_t,
        q_base_t, q_sensitivity_t, q_floor_t,
        r_base_t, r_sensitivity_t, r_floor_t,
        volatility_level
        ):
        """Вычисление адаптивных Q и R с учетом асимметрии волатильности (leverage effect)"""
        B = tf.shape(innov_prev)[0]
        leverage_strength_t = tf.reshape(leverage_strength_t, [B, 1])
        leverage_strength_t = tf.clip_by_value(leverage_strength_t, 0.5, 5.0)  # Расширяем диапазон

        base_vol = volatility_level  # [B]
        direction = tf.sign(innov_prev)  # [B]
        direction = tf.reshape(direction, [B, 1])  # [B, 1]

        # Адаптивный множитель волатильности - более гибкий для экстремальных рынков
        vol_multiplier = tf.where(
            direction < 0,
            leverage_strength_t * (1.0 + 0.5 * base_vol),  # При падении: больше усиление при высокой волатильности
            1.0 / (leverage_strength_t * (1.0 + 0.5 * base_vol) + 1e-8)  # При росте: больше сглаживание
        )

        vol_adaptive = base_vol * vol_multiplier  # [B, 1]
        vol_adaptive = tf.clip_by_value(vol_adaptive, 0.0, 10.0)  # Расширяем верхний предел

        # Вычисление Q и R с правильными размерностями
        q_val = q_base_t * (1.0 + q_sensitivity_t * vol_adaptive)  # [B, 1]
        q_val = tf.maximum(q_val, q_floor_t)  # [B, 1]
        Q_t = tf.reshape(q_val, [B, 1, 1])

        r_val = r_base_t * (1.0 + r_sensitivity_t * vol_adaptive)  # [B, 1]
        r_val = tf.maximum(r_val, r_floor_t)  # [B, 1]

        # 🔑 АДАПТИВНЫЙ ВЕРХНИЙ ЛИМИТ НА R В ЗАВИСИМОСТИ ОТ ВОЛАТИЛЬНОСТИ
        vol_level_normalized = tf.clip_by_value(volatility_level, 0.0, 1.0)
        max_r_adaptive = 2.0 + 3.0 * vol_level_normalized
        max_r_adaptive = tf.clip_by_value(max_r_adaptive, 2.0, 8.0)
        r_val = tf.minimum(r_val, max_r_adaptive)

        R_t = tf.reshape(r_val, [B, 1, 1])
        return Q_t, R_t, vol_adaptive

    def _process_filter_params(self, filter_params):
        """Обработка параметров для фильтрации текущих шагов"""
        params_clipped = tf.clip_by_value(filter_params, -8.0, 8.0)
        B = tf.shape(params_clipped)[0]
        T = tf.shape(params_clipped)[1]

        # Базовые параметры
        M_params = params_clipped[..., :9]
        q_raw = params_clipped[..., 9:12]
        r_raw = params_clipped[..., 12:15]
        hyperparams = params_clipped[..., 15:22]

        # Обработка матрицы переходов M
        M_matrix = tf.reshape(M_params, [B, T, self.num_modes, self.num_modes])

        # Вычисление обучаемой temperature с ограничением на минимальное значение
        temperature = tf.nn.softplus(self.m_temp_logit) + 0.1  # ≥ 0.1
        M_matrix = tf.nn.softmax(M_matrix / temperature, axis=-1)

        # === ОБУЧАЕМЫЕ МАСШТАБНЫЕ КОЭФФИЦИЕНТЫ ДЛЯ Q И R ===
        # Вычисление масштабов через softplus для положительности
        scale_factors_q = tf.nn.softplus(self.q_scale_logits) + 1e-6
        scale_factors_r = tf.nn.softplus(self.r_scale_logits) + 1e-6

        # Обработка Q параметров с иерархией
        q_sorted = tf.sort(tf.math.softplus(q_raw) + 1e-6, axis=-1)
        q_params = q_sorted * scale_factors_q
        q_params = tf.clip_by_value(q_params, 1e-5, 5.0)

        # Обработка R параметров с иерархией
        r_sorted = tf.sort(tf.math.softplus(r_raw) + 1e-6, axis=-1)
        r_params = r_sorted * scale_factors_r
        r_params = tf.clip_by_value(r_params, 1e-6, 2.0)

        # Обработка гиперпараметров дифференцируемого UKF
        ukf_hyperparams = {}

        # 1. Relax factor для сигма-точек
        sigma_relax_raw = hyperparams[..., 0:1]
        sigma_relax = 0.5 + 1.0 * tf.nn.sigmoid(sigma_relax_raw)
        ukf_hyperparams['sigma_relax'] = tf.ensure_shape(
            tf.squeeze(sigma_relax, axis=-1),
            [None, None]  # Гарантируем [B, T]!
        )

        # 2. Alpha factor для каждого шага
        alpha_factor_raw = hyperparams[..., 1:2]
        alpha_factor_t = 0.5 + 0.5 * tf.nn.sigmoid(alpha_factor_raw)  # [0.5, 1.0]
        ukf_hyperparams['alpha_factor_t'] = tf.squeeze(alpha_factor_t, axis=-1)  # [B, T]

        # 3. Kappa factor для каждого шага
        kappa_factor_raw = hyperparams[..., 2:3]
        kappa_factor_t = 0.5 + 2.0 * tf.nn.sigmoid(kappa_factor_raw)  # [0.5, 2.5]
        ukf_hyperparams['kappa_factor_t'] = tf.squeeze(kappa_factor_t, axis=-1)  # [B, T]

        # 4. Joseph blend
        joseph_blend_raw = hyperparams[..., 3:4]
        joseph_blend = tf.nn.sigmoid(joseph_blend_raw)
        ukf_hyperparams['joseph_blend'] = tf.squeeze(joseph_blend, axis=-1)

        # 5. Spectral regularization
        spectral_reg_raw = hyperparams[..., 4:5]
        spectral_reg = 0.01 + 0.1 * tf.nn.sigmoid(spectral_reg_raw)
        ukf_hyperparams['spectral_reg'] = tf.squeeze(spectral_reg, axis=-1)

        # 6-7. Регуляризация alpha_t/kappa_t
        alpha_t_reg_target = hyperparams[..., 5:6]
        kappa_t_reg_target = hyperparams[..., 6:7]
        ukf_hyperparams['alpha_t_reg_target'] = tf.squeeze(alpha_t_reg_target, axis=-1)
        ukf_hyperparams['kappa_t_reg_target'] = tf.squeeze(kappa_t_reg_target, axis=-1)

        return M_matrix, q_params, r_params, ukf_hyperparams, scale_factors_q, scale_factors_r

    def _process_forecast_params(self, forecast_params):
        """Обработка параметров специально для прогнозирования следующего шага"""
        params_clipped = tf.clip_by_value(forecast_params, -8.0, 8.0)
        B = tf.shape(params_clipped)[0]
        T = tf.shape(params_clipped)[1]

        # Гиперпараметры для прогнозирования (14 параметров)
        ukf_hyperparams_forecast = {}

        # 1. Relax factor для прогноза (для каждого режима)
        sigma_relax_raw = params_clipped[..., 0:3]  # 3 значения для 3 режимов
        sigma_relax = 0.7 + 1.5 * tf.nn.sigmoid(sigma_relax_raw)  # Расширенный диапазон для прогноза
        ukf_hyperparams_forecast['sigma_relax'] = sigma_relax[:, -1, :]  # Берем последний шаг

        # 2. Alpha factor для прогноза
        alpha_factor_raw = params_clipped[..., 3:6]
        alpha_factor_t = 0.3 + 1.2 * tf.nn.sigmoid(alpha_factor_raw)  # Более гибкие значения для прогноза
        ukf_hyperparams_forecast['alpha_factor_t'] = alpha_factor_t[:, -1, :]

        # 3. Kappa factor для прогноза
        kappa_factor_raw = params_clipped[..., 6:9]
        kappa_factor_t = 0.2 + 3.0 * tf.nn.sigmoid(kappa_factor_raw)  # Более широкий диапазон
        ukf_hyperparams_forecast['kappa_factor_t'] = kappa_factor_t[:, -1, :]

        # 4. Дополнительные параметры для прогноза
        q_forecast_scale = params_clipped[..., 9:12]  # Масштабы шумов процесса для прогноза
        ukf_hyperparams_forecast['q_forecast_scale'] = tf.nn.softplus(q_forecast_scale[:, -1, :]) + 1e-6

        # 5. Параметры коррекции прогноза
        forecast_adjustment = params_clipped[..., 12:14]  # 2 дополнительных параметра для коррекции
        ukf_hyperparams_forecast['forecast_adjustment'] = tf.tanh(forecast_adjustment[:, -1, :])

        # 6. Параметр boost_factor для экстремальных режимов
        jump_boost_factor = params_clipped[..., 13:14]  # последний параметр
        ukf_hyperparams_forecast['jump_boost_factor'] = tf.nn.softplus(jump_boost_factor[:, -1, :]) + 1.0

        return ukf_hyperparams_forecast

    @tf.function
    def _student_t_update(
        self,
        x_pred, P_pred, z, R,
        volatility_level,
        dof_adaptive,
        asymmetry_pos,
        asymmetry_neg
    ):
        """Student-t обновление UKF с точными heavy tails и асимметрией"""
        batch_size = tf.shape(x_pred)[0]

        # ✅ Гарантируем правильные размерности
        x_pred = tf.reshape(x_pred, [batch_size, 1])  # [B, 1]
        P_pred = tf.reshape(P_pred, [batch_size, 1, 1])  # [B, 1, 1]
        z = tf.reshape(z, [batch_size, 1])  # [B, 1]
        R = tf.reshape(R, [batch_size, 1, 1])  # [B, 1, 1]

        x_pred_scalar = tf.squeeze(x_pred, -1)  # [B]
        P_pred_scalar = tf.squeeze(P_pred, [-2, -1])  # [B]
        z_scalar = tf.squeeze(z, -1)  # [B]
        R_scalar = tf.squeeze(R, [-2, -1])  # [B]

        # ✅ Стандартные параметры Student-t
        dof_scalar = tf.squeeze(dof_adaptive, -1)  # [B]
        asymmetry_pos_scalar = tf.squeeze(asymmetry_pos, -1)  # [B]
        asymmetry_neg_scalar = tf.squeeze(asymmetry_neg, -1)  # [B]

        # ✅ Стандартное калманово обновление
        K_scalar = P_pred_scalar / (P_pred_scalar + R_scalar + 1e-8)  # [B]
        innov_scalar = z_scalar - x_pred_scalar  # [B]
        x_upd_scalar = x_pred_scalar + K_scalar * innov_scalar  # [B]

        # ✅ Joseph форма для стабильности
        P_upd_scalar = (1.0 - K_scalar) * P_pred_scalar * (1.0 - K_scalar) + K_scalar * R_scalar * K_scalar  # [B]

        # === ТОЧНЫЕ STUDENT-T ПАРАМЕТРЫ ===
        dof_scalar = tf.clip_by_value(dof_scalar, 3.1, 30.0)  # ✅ Избежать div0

        # ✅ ТОЧНАЯ формула дисперсии Student-t: var(t_ν) = ν/(ν-2)
        tail_factor = tf.sqrt(dof_scalar / (dof_scalar - 2.0 + 1e-4))  # [1.15, 1.41]
        tail_factor = tf.clip_by_value(tail_factor, 1.0, 1.5)  # Защита [file:1]

        # ✅ Нормализуем инновацию в единицах сигмы
        sigma_total = tf.sqrt(P_pred_scalar + R_scalar + 1e-8)  # [B]
        normalized_innov = innov_scalar / sigma_total  # [B]

        # ✅ Обнаружение хвостов (>2σ)
        tail_adjustment = tf.nn.softplus(1.2 * (tf.abs(normalized_innov) - 2.0))  # Агрессивно
        tail_adjustment = tf.clip_by_value(tail_adjustment, 0.0, 2.0)  # До 2x [file:1]

        # ✅ Адаптивная асимметрия с волатильностью
        vol_factor = tf.clip_by_value(0.8 + 0.6 * volatility_level, 0.8, 1.4)
        asymmetry_pos_scalar = tf.clip_by_value(asymmetry_pos_scalar, 0.6, 1.6)  # Pos хвост
        asymmetry_neg_scalar = tf.clip_by_value(asymmetry_neg_scalar, 0.6, 1.8)  # Neg толще [file:1]

        # ✅ Асимметричное взвешивание ТЕКУЩЕЙ инновации
        asymmetry_weight = tf.where(
            innov_scalar >= 0,
            asymmetry_pos_scalar,  # Положительная innov → pos вес
            asymmetry_neg_scalar   # Отрицательная → neg вес (толще)
        )  # [B]

        # ✅ ФИНАЛЬНАЯ ТОЧНАЯ КОРРЕКЦИЯ (Student-t логика)
        correction_factor = 1.0 + tail_factor * tail_adjustment * asymmetry_weight  # [1.0, 3.0]
        correction_factor = tf.clip_by_value(correction_factor, 1.0, 3.0)  # Экстремальный лимит
        P_upd_scalar *= correction_factor  # ✅ Расширение дисперсии
        P_upd_scalar = tf.maximum(P_upd_scalar, 1e-6)  # Стабильность [file:1]

        # ✅ Форматирование выходов
        x_upd = tf.reshape(x_upd_scalar, [batch_size, 1])  # [B, 1]
        P_upd = tf.reshape(P_upd_scalar, [batch_size, 1, 1])  # [B, 1, 1] ← heavy tails здесь!
        innov = tf.reshape(innov_scalar, [batch_size, 1, 1])  # [B, 1, 1]
        K = tf.reshape(K_scalar, [batch_size, 1, 1])  # [B, 1, 1]

        return x_upd, P_upd, innov, K

    def _apply_inflation_limits(self, inflation_factor, steps_above_threshold,
                              max_threshold=15.0, max_steps=5):
        """Сброс инфляции, если она застряла на высоком уровне
        inflation_factor: [B] - текущий фактор инфляции
        steps_above_threshold: [B] - счетчик шагов с высокой инфляцией
        max_threshold: максимальное значение инфляции перед сбросом
        max_steps: максимальное количество шагов перед сбросом

        Возвращает:
        inflation_factor: [B] - скорректированный фактор инфляции
        steps_above_threshold: [B] - обновленный счетчик шагов
        """
        # ✅ Маска для сброса: инфляция > порога И количество шагов > максимума
        # ✅ Сброс инфляции, если она застряла на высоком уровне
        reset_mask = tf.logical_and(
        inflation_factor > max_threshold,
        steps_above_threshold > max_steps
        )  # [B]

        # ✅ СНИЖЕНО пороговое значение сброса с 5.0 до 2.0
        inflation_factor = tf.where(
        reset_mask,
        tf.ones_like(inflation_factor) * 2.0,  # БЫЛО 5.0, СТАЛО 2.0
        inflation_factor
        )  # [B]

        # ✅ Обновление счетчика: сброс на 0 при сбросе инфляции, иначе инкремент
        steps_above_threshold = tf.where(
            reset_mask,
            tf.zeros_like(steps_above_threshold),
            tf.minimum(steps_above_threshold + 1, max_steps + 1)  # ограничиваем рост счетчика
        )  # [B]

        # ✅ Финальное ограничение счетчика снизу
        return inflation_factor, tf.maximum(steps_above_threshold, 0)  # [B], [B]

    @tf.function(jit_compile=False)
    def adaptive_ukf_filter(
        self,
        Xbatch: tf.Tensor,
        y_level_batch: tf.Tensor,
        vol_context: Dict[str, tf.Tensor],
        ukf_params: Dict[str, tf.Tensor],
        inflation_config: Dict[str, tf.Tensor],
        student_t_config: Dict[str, tf.Tensor],
        initial_state: tf.Tensor,
        initial_covariance: tf.Tensor,
        inflation_state_input: Optional[Dict[str, tf.Tensor]] = None,
        initial_volatility: Optional[tf.Tensor] = None  # ← ДОБАВЛЕНО
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Адаптивный UKF фильтр с контекстной волатильностью.

        Входные размерности:
        - Xbatch: [B, T, n_features]
        - y_level_batch: [B, T]
        - vol_context, ukf_params, inflation_config, student_t_config: [B, T, 1] каждый
        - initial_state: [B, 1]
        - initial_covariance: [B, 1, 1]

        Выходные размерности:
        - states: [B, T, 1]
        - innovations: [B, T, 1]
        - volatility: [B, T, 1]
        - inflation_factors: [B, T, 1]
        - final_state: [B, 1]
        - final_covariance: [B, 1, 1]
        """
        T = self.seq_len
        B = tf.shape(Xbatch)[0]
        state_dim = self.state_dim
        innov_window_size = 20  # Размер окна для истории инноваций

        # if self.debug_mode:
        #     tf.print("\n[DEBUG FILTER] Entry | Batch size:", B, "| Seq len:", T)
        #     tf.print("  Initial state shape:", tf.shape(initial_state))
        #     tf.print("  Initial cov shape:", tf.shape(initial_covariance))
        #     tf.print("  d_raw value:", self.diff_ukf_component.spec_param.d_raw)

        # Инициализация буфера для динамической коррекции порога
        if not hasattr(self, 'anomaly_buffer') or not hasattr(self, 'buffer_index'):
            self.anomaly_buffer_size = 100
            self.anomaly_buffer = tf.Variable(
                tf.zeros([self.anomaly_buffer_size], dtype=tf.float32),
                trainable=False,
                name='anomaly_buffer'
            )
            self.buffer_index = tf.Variable(0, dtype=tf.int32, trainable=False, name='buffer_index')

        # ЯВНОЕ УПРАВЛЕНИЕ РАЗМЕРНОСТЯМИ
        Xbatch = tf.cast(Xbatch, tf.float32)
        y_level_batch = tf.cast(y_level_batch, tf.float32)
        initial_state = tf.cast(initial_state, tf.float32)
        initial_covariance = tf.cast(initial_covariance, tf.float32)

        if not hasattr(self, 'feature_to_idx'):
            self.feature_to_idx = {feature: idx for idx, feature in enumerate(self.feature_columns)}

        # ✅ СОЗДАНИЕ TensorArray С ЯВНЫМИ ТИПАМИ И ФОРМАМИ
        states_hist = tf.TensorArray(tf.float32, size=T, element_shape=tf.TensorShape([None]),
                                   dynamic_size=False, clear_after_read=False)
        innovations_hist = tf.TensorArray(tf.float32, size=T, element_shape=tf.TensorShape([None]),
                                        dynamic_size=False, clear_after_read=False)
        volatility_levels = tf.TensorArray(tf.float32, size=T, element_shape=tf.TensorShape([None]),
                                         dynamic_size=False, clear_after_read=False)
        inflation_factors_hist = tf.TensorArray(tf.float32, size=T, element_shape=tf.TensorShape([None]),
                                              dynamic_size=False, clear_after_read=False)
        high_infl_steps_hist = tf.TensorArray(tf.int32, size=T, element_shape=tf.TensorShape([None]),
                                            dynamic_size=False, clear_after_read=False)
        correction_adaptive_hist = tf.TensorArray(tf.float32, size=T, element_shape=tf.TensorShape([None]))

        # АДАПТИВНАЯ инициализация текущих состояний с учётом контекста
        current_state = initial_state  # [B, state_dim]

        # ✅ ПРИОРИТЕТ: если передана историческая волатильность — используем её как основу
        if initial_volatility is not None:
            # initial_volatility: [B] — волатильность с предыдущего шага
            # Комбинируем с текущими данными для плавного перехода
            data_vol = tf.math.reduce_std(y_level_batch[:, :10], axis=1) + 1e-6  # [B]
            # Взвешенное усреднение: 70% историческая + 30% текущие данные
            blended_vol = 0.7 * tf.maximum(initial_volatility, 1e-6) + 0.3 * data_vol
            initial_vol = blended_vol  # [B]
        else:
            # Fallback: инициализация только из данных
            initial_vol = tf.math.reduce_std(y_level_batch[:, :10], axis=1) + 1e-6  # [B]

        initial_cov_value = tf.maximum(initial_vol * 0.5, 0.1)  # Минимум 0.1 для стабильности
        current_covariance = tf.reshape(initial_cov_value, [B, 1, 1])  # [B, 1, 1]
        current_volatility = tf.zeros([B], dtype=tf.float32)  # [B]

        # ✅ ИНИЦИАЛИЗАЦИЯ ПАРАМЕТРОВ ДЛЯ ADAPTIVE INFLATION С ЕДИНООБРАЗНЫМИ ТИПАМИ
        if inflation_state_input is None:
            # Все значения с типом tf.int32 для единообразия
            inflation_factor_init = tf.ones([B], dtype=tf.float32)  # [B]
            remaining_steps_init = tf.zeros([B], dtype=tf.int32)    # [B]
            last_anomaly_time_init = tf.fill([B], tf.constant(-100, dtype=tf.int32))  # [B] - ВСЕ tf.int32
            high_inflation_steps_init = tf.zeros([B], dtype=tf.int32)  # [B]
        else:
            inflation_factor_init = tf.cast(inflation_state_input.get('factor', tf.ones([B], dtype=tf.float32)), tf.float32)
            remaining_steps_init = tf.cast(inflation_state_input.get('remaining_steps', tf.zeros([B], dtype=tf.int32)), tf.int32)
            last_anomaly_time_init = tf.cast(inflation_state_input.get('last_anomaly_time', tf.fill([B], tf.constant(-100, dtype=tf.int32))), tf.int32)
            high_inflation_steps_init = tf.cast(inflation_state_input.get('high_inflation_steps', tf.zeros([B], dtype=tf.int32)), tf.int32)

        # ✅ ИНИЦИАЛИЗАЦИЯ ОКНА ИННОВАЦИЙ
        initial_std = 0.1
        innov_window_init = tf.random.normal(
            [B, innov_window_size],
            mean=0.0,
            stddev=initial_std, # небольшой начальный шум вместо нулей
            dtype=tf.float32
        )

        def cond(t, state, cov, vol, innov_win, inf_factor, rem_steps, last_anom_time,
                 s_hist, i_hist, v_hist, f_hist, high_infl_steps, corr_adapt_hist):
            return t < T

        def body(t, state, cov, vol, innov_win, inf_factor, rem_steps, last_anom_time,
                 s_hist, i_hist, v_hist, f_hist, high_infl_steps, corr_adapt_hist):
            """
            Тело цикла tf.while_loop с гарантированными формами.
            """
            # ✅ ГАРАНТИРУЕМ ФОРМЫ ВХОДНЫХ ПЕРЕМЕННЫХ
            state = tf.ensure_shape(state, [None, 1])
            cov = tf.ensure_shape(cov, [None, 1, 1])
            vol = tf.ensure_shape(vol, [None])
            innov_win = tf.ensure_shape(innov_win, [None, innov_window_size])
            inf_factor = tf.ensure_shape(inf_factor, [None])
            rem_steps = tf.ensure_shape(rem_steps, [None])
            last_anom_time = tf.ensure_shape(last_anom_time, [None])
            # high_infl_steps = tf.ensure_shape(high_infl_steps, [None])

            # ✅ СТАБИЛЬНОЕ ОПРЕДЕЛЕНИЕ РАЗМЕРА БАТЧА ЧЕРЕЗ B
            B_batch = B

            # Извлечение текущих признаков и наблюдений
            current_features = tf.gather(Xbatch, t, axis=1)  # [B, n_features]
            current_observation = tf.gather(y_level_batch, t, axis=1)  # [B]
            t_idx = tf.cast(t, tf.int32)

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 1: ИЗВЛЕЧЕНИЕ ПАРАМЕТРОВ С ПОМОЩЬЮ tf.gather
            # ════════════════════════════════════════════════════════════════

            # --- Извлечение параметров из vol_context ---
            leverage_strength_t = tf.gather(vol_context['leverage_effect_strength'], t_idx, axis=1)  # [B, 1]
            sensitivity_short_t = tf.gather(vol_context['sensitivity_short'], t_idx, axis=1)  # [B, 1]
            sensitivity_medium_t = tf.gather(vol_context['sensitivity_medium'], t_idx, axis=1)  # [B, 1]
            sensitivity_long_t = tf.gather(vol_context['sensitivity_long'], t_idx, axis=1)  # [B, 1]
            entropy_weight_t = tf.gather(vol_context['entropy_weight'], t_idx, axis=1)  # [B, 1]
            accel_weight_t = tf.gather(vol_context['accel_weight'], t_idx, axis=1)  # [B, 1]
            memory_factor_t = tf.gather(vol_context['memory_factor'], t_idx, axis=1)  # [B, 1]

            # --- Извлечение параметров из ukf_params ---
            q_base_t = tf.gather(ukf_params['q_base'], t_idx, axis=1)  # [B, 1]
            q_sensitivity_t = tf.gather(ukf_params['q_sensitivity'], t_idx, axis=1)  # [B, 1]
            q_floor_t = tf.gather(ukf_params['q_floor'], t_idx, axis=1)  # [B, 1]
            r_base_t = tf.gather(ukf_params['r_base'], t_idx, axis=1)  # [B, 1]
            r_sensitivity_t = tf.gather(ukf_params['r_sensitivity'], t_idx, axis=1)  # [B, 1]
            r_floor_t = tf.gather(ukf_params['r_floor'], t_idx, axis=1)  # [B, 1]
            relax_base_t = tf.gather(ukf_params['relax_base'], t_idx, axis=1)  # [B, 1]
            relax_sensitivity_t = tf.gather(ukf_params['relax_sensitivity'], t_idx, axis=1)  # [B, 1]
            alpha_base_t = tf.gather(ukf_params['alpha_base'], t_idx, axis=1)  # [B, 1]
            alpha_sensitivity_t = tf.gather(ukf_params['alpha_sensitivity'], t_idx, axis=1)  # [B, 1]
            kappa_base_t = tf.gather(ukf_params['kappa_base'], t_idx, axis=1)  # [B, 1]
            kappa_sensitivity_t = tf.gather(ukf_params['kappa_sensitivity'], t_idx, axis=1)  # [B, 1]

            # --- Извлечение параметров из inflation_config ---
            base_inflation_t = tf.gather(inflation_config['base_inflation'], t_idx, axis=1)  # [B, 1]
            vol_sensitivity_t = tf.gather(inflation_config['vol_sensitivity'], t_idx, axis=1)  # [B, 1]
            decay_rate_t = tf.gather(inflation_config['decay_rate'], t_idx, axis=1)  # [B, 1]
            anomaly_threshold_t = tf.gather(inflation_config['anomaly_threshold'], t_idx, axis=1)  # [B, 1]
            min_duration_float_t = tf.gather(inflation_config['min_duration_float'], t_idx, axis=1)  # [B, 1]
            max_duration_float_t = tf.gather(inflation_config['max_duration_float'], t_idx, axis=1)  # [B, 1]
            asymmetry_factor_t = tf.gather(inflation_config['asymmetry_factor'], t_idx, axis=1)  # [B, 1]
            memory_decay_t = tf.gather(inflation_config['memory_decay'], t_idx, axis=1)  # [B, 1]
            inflation_limit_t = tf.gather(inflation_config['inflation_limit'], t_idx, axis=1)  # [B, 1]

            # 🔑 КРИТИЧЕСКИ ВАЖНО: определяем ДО использования в ЭТАПЕ 12
            inflation_limit_val = tf.squeeze(inflation_limit_t, axis=-1)  # [B]

            # --- Извлечение параметров из student_t_config ---
            dof_base_t = tf.gather(student_t_config['dof_base'], t_idx, axis=1)  # [B, 1]
            dof_sensitivity_t = tf.gather(student_t_config['dof_sensitivity'], t_idx, axis=1)  # [B, 1]
            asymmetry_pos_t = tf.gather(student_t_config['asymmetry_pos'], t_idx, axis=1)  # [B, 1]
            asymmetry_neg_t = tf.gather(student_t_config['asymmetry_neg'], t_idx, axis=1)  # [B, 1]

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 2: ВЫЧИСЛЕНИЕ RAW VOLATILITY
            # ════════════════════════════════════════════════════════════════

            vol_short_idx = self.feature_to_idx.get('log_vol_short', 2)
            rel_vol_short_long_idx = self.feature_to_idx.get('rel_vol_short_long', 18)
            vol_accel_idx = self.feature_to_idx.get('vol_accel_rel', 19)
            rel_entropy_idx = self.feature_to_idx.get('rel_entropy', 20)

            rel_vol_short_long = tf.gather(current_features, rel_vol_short_long_idx, axis=1)  # [B]
            vol_accel = tf.gather(current_features, vol_accel_idx, axis=1)  # [B]
            rel_entropy = tf.gather(current_features, rel_entropy_idx, axis=1)  # [B]
            vol_medium_est = tf.abs(tf.gather(current_features, self.feature_to_idx.get('yz', 8), axis=1))  # [B]
            vol_long_est = tf.abs(tf.gather(current_features, self.feature_to_idx.get('ht', 12), axis=1))  # [B]

            # ✅ ИСПРАВЛЕНО: распаковка [B, 1] → [B] перед умножением
            sensitivity_short = tf.squeeze(sensitivity_short_t, axis=-1)  # [B]
            sensitivity_medium = tf.squeeze(sensitivity_medium_t, axis=-1)  # [B]
            sensitivity_long = tf.squeeze(sensitivity_long_t, axis=-1)  # [B]
            entropy_weight = tf.squeeze(entropy_weight_t, axis=-1)  # [B]
            accel_weight = tf.squeeze(accel_weight_t, axis=-1)  # [B]

            # Формирование компонентов волатильности - [B] * [B] = [B] ✅
            vol_components = tf.stack([
                sensitivity_short * rel_vol_short_long,
                sensitivity_medium * (vol_medium_est / (vol_long_est + 1e-8)),
                sensitivity_long * (vol_long_est / (tf.reduce_mean(vol_long_est) + 1e-8)),
                entropy_weight * tf.abs(rel_entropy),
                accel_weight * tf.abs(vol_accel)
            ], axis=1)  # [B, 5]
            raw_volatility = tf.reduce_mean(vol_components, axis=1)  # [B]

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 3: ЭКСПОНЕНЦИАЛЬНОЕ СГЛАЖИВАНИЕ ВОЛАТИЛЬНОСТИ
            # ════════════════════════════════════════════════════════════════

            memory_factor_val = tf.squeeze(memory_factor_t, axis=-1)  # [B, 1] → [B]
            new_volatility = memory_factor_val * vol + (1.0 - memory_factor_val) * raw_volatility  # [B]
            new_volatility = tf.ensure_shape(new_volatility, [None])
            volatility_level = tf.nn.sigmoid(new_volatility)  # [0, 1]

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 4: ПОЛУЧЕНИЕ ПРЕДЫДУЩЕЙ ИННОВАЦИИ
            # ════════════════════════════════════════════════════════════════

            innov_prev = tf.cond(
                t > 0,
                lambda: tf.ensure_shape(i_hist.read(t - 1), [None]),  # [B]
                lambda: tf.zeros([B_batch], dtype=tf.float32)  # [B]
            )

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 5: ВЫЧИСЛЕНИЕ АДАПТИВНЫХ Q И R
            # ════════════════════════════════════════════════════════════════

            Q_t, R_t, vol_adaptive = self.compute_adaptive_Q_R_with_leverage(
                innov_prev,
                leverage_strength_t,
                q_base_t, q_sensitivity_t, q_floor_t,
                r_base_t, r_sensitivity_t, r_floor_t,
                tf.reshape(volatility_level, [B_batch, 1])  # ← volatility_level ∈ [0, 1]!
            )

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 6: АДАПТИВНЫЕ ПАРАМЕТРЫ UKF
            # ════════════════════════════════════════════════════════════════

            # ✅ ИСПРАВЛЕНО: явное преобразование [B, 1] → [B]
            relax_base_val = tf.squeeze(relax_base_t, axis=-1)  # [B]
            alpha_base_val = tf.squeeze(alpha_base_t, axis=-1)  # [B]
            kappa_base_val = tf.squeeze(kappa_base_t, axis=-1)  # [B]

            relax_sensitivity_val = tf.squeeze(relax_sensitivity_t, axis=-1)  # [B]
            alpha_sensitivity_val = tf.squeeze(alpha_sensitivity_t, axis=-1)  # [B]
            kappa_sensitivity_val = tf.squeeze(kappa_sensitivity_t, axis=-1)  # [B]

            relax_factor = relax_base_val * (1.0 + relax_sensitivity_val * volatility_level)  # [B]
            alpha_t = alpha_base_val * (1.0 + alpha_sensitivity_val * volatility_level)  # [B]
            kappa_t = kappa_base_val * (1.0 + kappa_sensitivity_val * volatility_level)  # [B]

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 7: ПРИМЕНЕНИЕ ИНФЛЯЦИИ К ШУМУ ПРОЦЕССА Q (ДЛЯ ПРЕДСКАЗАНИЯ)
            # ════════════════════════════════════════════════════════════════

            # 🔑 АДАПТИВНЫЕ ГРАНИЦЫ ИНФЛЯЦИИ С ЗАВИСИМОСТЬЮ ОТ ВОЛАТИЛЬНОСТИ
            inflation_cap = 1.8 + 1.2 * volatility_level  # Базовый максимум 1.8 с адаптацией под волатильность
            inflation_floor = 1.0 - 0.2 * volatility_level  # Минимум снижается при высокой волатильности
            inflation_factor_limited = tf.clip_by_value(inf_factor, inflation_floor, inflation_cap)  # [B]

            # Инфляция для Q (для этапа предсказания) — МЕНЕЕ АГРЕССИВНАЯ
            inflation_factor_for_Q = tf.reshape(inflation_factor_limited, [B_batch, 1, 1])
            time_penalty_Q = tf.exp(-0.03 * tf.cast(t, tf.float32))
            inflation_factor_for_Q = inflation_factor_for_Q * (0.6 + 0.4 * time_penalty_Q)
            Q_inflated = Q_t * inflation_factor_for_Q
            Q_inflated = tf.clip_by_value(Q_inflated, 1e-8, self.Q_max_pred)

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 8: UKF ПРЕДСКАЗАНИЕ (PREDICT) — Q_inflated УЖЕ ОПРЕДЕЛЕН!
            # ════════════════════════════════════════════════════════════════

            x_pred, P_pred = self.diff_ukf_component.predict(
                state,          # [B, 1] — текущее состояние
                cov,            # [B, 1, 1] ← ДОБАВЛЕНО: ТЕКУЩАЯ КОВАРИАЦИЯ!
                Q_inflated,     # [B, 1, 1] — шум процесса С ИНФЛЯЦИЕЙ
                relax_factor=relax_factor,
                alpha_t=alpha_t,
                kappa_t=kappa_t
            )  # → x_pred: [B, 1], P_pred: [B, 1, 1]

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 9: ГИБРИДНЫЙ ДЕТЕКТОР СКАЧКОВ
            # ════════════════════════════════════════════════════════════════

            # Используем историю инноваций для расчета статистик
            innov_window = innov_win  # [B, innov_window_size]

            # === 1. Вычисляем статистику инноваций за последние 20 шагов ===
            innov_mean = tf.math.reduce_mean(innov_window, axis=1)  # [B]

            # === 2. ОПРЕДЕЛЕНИЕ detection_strength (постепенная активация детектора)
            min_warmup_steps = 10
            is_warmup_phase = tf.cast(t < min_warmup_steps, tf.float32)
            detection_strength = tf.minimum(1.0, tf.cast(t, tf.float32) / min_warmup_steps)

            # === 3. Нормализованная инновация ===
            current_observation_scalar = tf.squeeze(current_observation)  # [B]
            state_scalar = tf.squeeze(state[:, 0])  # [B]
            current_innov = current_observation_scalar - state_scalar  # [B]

            # ✅ УЛУЧШЕННАЯ ЗАЩИТА ОТ ДЕЛЕНИЯ НА НОЛЬ
            innov_std = tf.math.reduce_std(innov_window, axis=1)
            # Гарантируем минимальное стандартное отклонение
            min_std = tf.maximum(0.01, tf.reduce_mean(tf.abs(innov_window)) * 0.1)
            innov_std = tf.maximum(innov_std, min_std) + 1e-8  # [B]

            # ✅ ПРАВИЛЬНЫЙ РАСЧЕТ СТАНДАРТНОГО ОТКЛОНЕНИЯ НА ОСНОВЕ P_pred И Q_inflated
            P_sqrt = tf.sqrt(tf.maximum(P_pred[:, 0, 0], 1e-8))  # [B]
            Q_sqrt = tf.sqrt(tf.maximum(Q_inflated[:, 0, 0], 1e-8))  # [B]
            std_dev_scalar = tf.sqrt(P_sqrt**2 + Q_sqrt**2)  # Комбинированное std для прогноза

            # Ограничение для численной стабильности
            std_dev_scalar = tf.maximum(std_dev_scalar, 1e-8)

            normalized_innov = tf.abs(current_innov) / std_dev_scalar

            # === 4. УЛУЧШЕННЫЙ АДАПТИВНЫЙ ПОРОГ С ОГРАНИЧЕННЫМ ДИАПАЗОНОМ ===
            adaptive_threshold_val = tf.squeeze(anomaly_threshold_t, axis=-1)  # [B]

            # Используем обучаемый confidence_base для нормализации порога
            # LSTM-only: порог задаётся anomaly_threshold_t + мягкая поправка на volatility_level
            adaptive_threshold = adaptive_threshold_val * (1.0 + 0.35 * volatility_level)  # [B]
            adaptive_threshold = tf.clip_by_value(adaptive_threshold, 2.3, 3.8)

            is_anomaly_t = normalized_innov > adaptive_threshold
            is_anomaly = tf.cast(is_anomaly_t, tf.float32) * detection_strength  # [B]

            # === ДИНАМИЧЕСКАЯ КОРРЕКЦИЯ ПОРОГА НА ОСНОВЕ СТАТИСТИКИ ===
            # Обновляем буфер аномалий
            current_anomaly_rate = tf.reduce_mean(is_anomaly, axis=0)
            buffer_pos = tf.math.mod(self.buffer_index, self.anomaly_buffer_size)
            self.anomaly_buffer.scatter_nd_update([[buffer_pos]], [current_anomaly_rate])
            self.buffer_index.assign_add(1)

            # Рассчитываем скользящее среднее по аномалиям
            effective_buffer_size = tf.minimum(self.buffer_index, self.anomaly_buffer_size)
            rolling_anomaly_mean = tf.reduce_mean(self.anomaly_buffer.value()[:effective_buffer_size])

            # Динамический коэффициент доверия: при высоком проценте аномалий повышаем порог
            # Целевой уровень аномалий: 20%
            target_anomaly_rate = 0.20
            threshold_boost = 1.0 + tf.maximum(0.0, rolling_anomaly_mean - target_anomaly_rate) * 3.0

            adaptive_threshold = adaptive_threshold * threshold_boost
            # Пересчитываем anomaly с учётом динамической коррекции порога
            is_anomaly_t = normalized_innov > adaptive_threshold
            is_anomaly = tf.cast(is_anomaly_t, tf.float32) * detection_strength  # [B]

            # === 5. ПОСТЕПЕННАЯ АКТИВАЦИЯ ГИСТЕРЕЗИСА ===
            min_anomaly_interval = 7  # Минимальное количество шагов между аномалиями
            max_consecutive_anomalies = 3  # Максимальное число последовательных аномалий

            # Рассчитываем степень заполнения буфера аномалий
            buffer_fill_ratio = tf.cast(self.buffer_index, tf.float32) / self.anomaly_buffer_size
            # Постепенная активация гистерезиса: 0.0 на старте → 1.0 при заполнении 50% буфера
            hysteresis_activation = tf.minimum(1.0, buffer_fill_ratio * 2.0)

            # Базовые условия блокировки
            time_since_last_anomaly = t - last_anom_time  # [B]
            is_blocked = time_since_last_anomaly < min_anomaly_interval  # [B]
            is_too_many_consecutive = last_anom_time >= t - max_consecutive_anomalies  # [B]

            # Условие блокировки при высоком проценте аномалий
            rolling_anomaly_mean = tf.reduce_mean(self.anomaly_buffer.value()[:tf.minimum(self.buffer_index, self.anomaly_buffer_size)])
            is_high_anomaly_rate = rolling_anomaly_mean > 0.35  # Блокируем при >35% аномалий

            # Объединяем все условия блокировки
            should_block_base = tf.logical_or(
                tf.logical_or(is_blocked, is_too_many_consecutive),
                is_high_anomaly_rate
            )  # [B]

            # Применяем постепенную активацию гистерезиса
            hysteresis_mask = tf.cast(hysteresis_activation > 0.5, tf.bool)  # Активируем только после 50% заполнения
            should_block_detection = tf.logical_and(should_block_base, hysteresis_mask)  # [B]

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 10: РЕЗЕРВНАЯ КОРРЕКЦИЯ ДЛЯ ПРОПУЩЕННЫХ СКАЧКОВ
            # ════════════════════════════════════════════════════════════════

            fallback_multiplier = tf.ones([B_batch, 1], dtype=tf.float32)
            is_missed_jump_flag = tf.zeros([B_batch, 1], dtype=tf.float32)

            if t > 0:
                # ✅ ИСПРАВЛЕНО: читаем из s_hist (история состояний), а не из i_hist (история инноваций)
                x_pred_prev = tf.reshape(s_hist.read(t - 1), [B_batch, 1])  # [B] → [B, 1]
                x_pred_prev = tf.ensure_shape(x_pred_prev, [None, 1])
                # ✅ ОБНОВЛЯЕМ ПЕРЕМЕННЫЕ ТОЛЬКО ЕСЛИ t > 0
                fallback_multiplier_new, is_missed_jump_flag_new, correction_adaptive_val = \
                    self._fallback_inflation_correction(
                        t,
                        x_pred_prev,  # [B, 1]
                        x_pred,  # [B, 1]
                        P_pred,  # [B, 1, 1]
                        tf.reshape(inf_factor, [B_batch, 1]),  # [B] → [B, 1]
                        base_inflation_t,  # [B, 1]
                        vol_sensitivity_t,  # [B, 1]
                        decay_rate_t,  # [B, 1]
                        anomaly_threshold_t,  # [B, 1]
                        min_duration_float_t,  # [B, 1]
                        max_duration_float_t,  # [B, 1]
                        asymmetry_factor_t,  # [B, 1]
                        memory_decay_t,  # [B, 1]
                        inflation_limit_t,  # [B, 1]
                        tf.reshape(new_volatility, [B_batch, 1])  # [B] → [B, 1]
                    )
                # ✅ БЕЗОПАСНОЕ ПРИСВАИВАНИЕ С УЧЕТОМ ФОРМЫ
                fallback_multiplier = tf.reshape(fallback_multiplier_new, [B_batch, 1])
                is_missed_jump_flag = tf.reshape(is_missed_jump_flag_new, [B_batch, 1])
                correction_adaptive_val = tf.reshape(correction_adaptive_val, [B_batch])  # [B]
            else:
                # Fallback для первого шага
                correction_adaptive_val = tf.ones([B_batch], dtype=tf.float32)  # [B]

            # Объединяем детекцию: аномалия ИЛИ пропущенный скачок
            should_activate_combined = tf.cast(is_anomaly, tf.bool)

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 11: АДАПТИВНАЯ ИНФЛЯЦИЯ
            # ════════════════════════════════════════════════════════════════

            current_inflation_factor = inf_factor  # [B]
            remaining_steps = rem_steps  # [B]
            t_scalar = tf.cast(t, tf.int32)

            # Определение, нужно ли активировать inflation
            should_activate = tf.logical_and(
                should_activate_combined,
                tf.equal(remaining_steps, 0)
            )  # [B]

            # Длительность инфляции - ✅ ИСПРАВЛЕНО: выравнивание размерностей
            min_dur_val = tf.squeeze(min_duration_float_t, axis=-1)  # [B]
            max_dur_val = tf.squeeze(max_duration_float_t, axis=-1)  # [B]
            duration_f = min_dur_val + (max_dur_val - min_dur_val) * volatility_level  # [B]
            duration = tf.cast(duration_f, tf.int32)  # [B]

            # Если активация нужна, используем вычисленную длительность, иначе уменьшаем текущую
            duration = tf.where(should_activate, duration, remaining_steps)  # [B]
            duration = tf.ensure_shape(duration, [None])

            # Расчет нового фактора инфляции - ✅ ИСПРАВЛЕНО: выравнивание размерностей
            base_inflation_val = tf.squeeze(base_inflation_t, axis=-1)  # [B]
            vol_sensitivity_val = tf.squeeze(vol_sensitivity_t, axis=-1)  # [B]
            base_factor = base_inflation_val * (1.0 + vol_sensitivity_val * volatility_level)  # [B]

            # Асимметрия: увеличиваем инфляцию при падении цены
            asymmetry_factor_val = tf.squeeze(asymmetry_factor_t, axis=-1)  # [B]
            asymmetry_factor = tf.where(
                current_observation > state[:, 0],
                asymmetry_factor_val,  # [B]
                1.0 / (tf.maximum(asymmetry_factor_val, 1e-8) + 1e-8)  # [B]
            )  # [B]
            new_factor = base_factor * asymmetry_factor  # [B]

            # Активация inflation с резервной коррекцией
            fallback_mult_val = tf.squeeze(fallback_multiplier, axis=-1)  # [B]
            inflation_factor = tf.where(
                should_activate_combined,
                new_factor * fallback_mult_val,  # [B]
                current_inflation_factor  # [B]
            )  # [B]

            # Обновление времени последней аномалии
            batch_size = tf.shape(last_anom_time)[0]
            last_anom_time_updated = tf.where(
                should_activate_combined,
                tf.fill([batch_size], t_scalar),
                last_anom_time
            )  # [B]
            last_anom_time_updated = tf.ensure_shape(last_anom_time_updated, [None])

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 12: DYNAMIC INFLATION FLOOR
            # ════════════════════════════════════════════════════════════════

            current_high_inflation_steps = tf.cond(
                t > 0,
                lambda: tf.ensure_shape(high_infl_steps.read(t - 1), [None]),  # [B]
                lambda: tf.zeros([B_batch], dtype=tf.int32)  # [B]
            )
            current_high_inflation_steps = tf.ensure_shape(current_high_inflation_steps, [None])

            inflation_threshold_val = inflation_limit_val * 0.8  # [B]
            inflation_factor_flat, updated_high_inflation_steps = self._apply_inflation_limits(
                inflation_factor,  # [B]
                current_high_inflation_steps,  # [B]
                max_threshold=inflation_threshold_val,  # [B]
                max_steps=5
            )  # → inflation_factor_flat: [B], updated_high_inflation_steps: [B]

            # === НОВОЕ: ЖЕСТКИЕ ОГРАНИЧЕНИЯ НА ИНФЛЯЦИЮ ===
            max_inflation = tf.minimum(5.0, tf.maximum(3.0, inflation_factor_flat))  # жесткий cap на inflation
            inflation_factor_flat = max_inflation  # [B] - обновленный фактор инфляции

            inflation_factor_flat = tf.ensure_shape(inflation_factor_flat, [None])
            updated_high_inflation_steps = tf.ensure_shape(updated_high_inflation_steps, [None])
            # Формируем правильную размерность для инфляции
            # === АДАПТИВНЫЕ ГРАНИЦЫ ИНФЛЯЦИИ С ЗАВИСИМОСТЬЮ ОТ ВОЛАТИЛЬНОСТИ ===
            inflation_cap = 1.8 + 1.2 * volatility_level  # Базовый максимум 1.8 с адаптацией под волатильность
            inflation_floor = 1.0 - 0.2 * volatility_level  # Минимум снижается при высокой волатильности
            inflation_factor_limited = tf.clip_by_value(inflation_factor_flat, inflation_floor, inflation_cap)
            inflation_factor_for_R = tf.reshape(inflation_factor_limited, [B_batch, 1, 1])  # [B] → [B, 1, 1]

            # Дополнительное ослабление: уменьшаем влияние инфляции со временем
            time_penalty = tf.exp(-0.05 * tf.cast(t, tf.float32))  # Экспоненциальное ослабление
            inflation_factor_for_R = inflation_factor_for_R * (0.5 + 0.5 * time_penalty)
            # ===== КОНЕЦ =====

            # === НОВОЕ: МЯГКОЕ ЗАТУХАНИЕ ИНФЛЯЦИИ ===
            decay_factor = tf.pow(0.95, tf.cast(t, tf.float32)) + 0.05  # Гарантируем минимум 0.05
            inflation_factor = inflation_factor * decay_factor + 1.0 * (1 - decay_factor)

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 13: ПРИМЕНЕНИЕ ИНФЛЯЦИИ К КОВАРИАЦИИ ИЗМЕРЕНИЙ R (ДЛЯ ОБНОВЛЕНИЯ)
            # ════════════════════════════════════════════════════════════════

            # 🔑 ИСПОЛЬЗУЕМ УЖЕ ОГРАНИЧЕННЫЙ ФАКТОР inflation_factor_limited (вычислен в ЭТАПЕ 7)
            inflation_factor_for_R = tf.reshape(inflation_factor_limited, [B_batch, 1, 1])  # [B] → [B, 1, 1]
            inflation_limit_val_reshape = tf.reshape(inflation_limit_val, [B_batch, 1, 1])

            # СИММЕТРИЧНОЕ ПРИМЕНЕНИЕ ИНФЛЯЦИИ К R (только ОДИН раз!)
            R_inflated = R_t * inflation_factor_for_R
            # 🔑 ДОПОЛНИТЕЛЬНЫЙ ЖЁСТКИЙ ЛИМИТ НА ИТОГОВЫЙ R_inflated
            R_inflated = tf.clip_by_value(R_inflated, 1e-8, 8.0)  # ← глобальный лимит 8.0

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 14: ВЫЧИСЛЕНИЕ АДАПТИВНЫХ ПАРАМЕТРОВ STUDENT-T
            # ════════════════════════════════════════════════════════════════

            # ✅ ИСПРАВЛЕНО: выравнивание размерностей
            dof_base_val = tf.squeeze(dof_base_t, axis=-1)  # [B]
            dof_sensitivity_val = tf.squeeze(dof_sensitivity_t, axis=-1)  # [B]

            # Адаптация DOF к текущей волатильности
            vol_mean = tf.reduce_mean(new_volatility)
            vol_std = tf.math.reduce_std(new_volatility) + 1e-6
            vol_normalized = (new_volatility - vol_mean) / vol_std  # [B]
            dof_adaptive_val = dof_base_val - dof_sensitivity_val * tf.nn.relu(vol_normalized)  # [B]
            dof_adaptive_val = tf.clip_by_value(dof_adaptive_val, 3.0, 11.0)  # [B]
            dof_adaptive = tf.reshape(dof_adaptive_val, [B_batch, 1])  # [B] → [B, 1]

            # Асимметричные параметры
            asymmetry_pos = asymmetry_pos_t  # [B, 1]
            asymmetry_neg = asymmetry_neg_t  # [B, 1]

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 15: UKF ОБНОВЛЕНИЕ (UPDATE) СО STUDENT-T КОРРЕКЦИЕЙ
            # ════════════════════════════════════════════════════════════════

            x_upd, P_upd, innov, K = self._student_t_update(
                x_pred,  # [B, 1]
                P_pred,  # [B, 1, 1]
                tf.expand_dims(current_observation, axis=1),  # [B] → [B, 1]
                R_inflated,  # [B, 1, 1]
                tf.expand_dims(volatility_level, axis=-1),  # [B] → [B, 1]
                dof_adaptive,  # [B, 1]
                asymmetry_pos,  # [B, 1]
                asymmetry_neg   # [B, 1]
            )  # → x_upd: [B, 1], P_upd: [B, 1, 1], innov: [B, 1, 1]

            # ===== АДАПТИВНОЕ ОГРАНИЧЕНИЕ НА P_pred =====
            P_diag = tf.linalg.diag_part(P_pred)  # [B]
            # Используем адаптивное ограничение в зависимости от текущей волатильности
            current_volatility = tf.reduce_mean(tf.abs(tf.gather(Xbatch, self.feature_to_idx['log_vol_short'], axis=2)), axis=1)
            vol_adjustment = 1.0 + 2.0 * tf.nn.sigmoid(current_volatility * 5.0)  # Адаптация под волатильность
            max_rel_std = 0.02 * vol_adjustment  # Базовое 2% с адаптацией до 4%
            max_std = max_rel_std * tf.reduce_mean(tf.abs(current_observation_scalar)) + 0.01  # Минимум 0.01
            max_var = tf.square(max_std)
            P_diag_clipped = tf.minimum(P_diag, max_var)
            P_pred = tf.linalg.set_diag(P_pred, P_diag_clipped)

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 16: ЗАПИСЬ В ИСТОРИЮ
            # ════════════════════════════════════════════════════════════════

            # ✅ ЯВНОЕ УКАЗАНИЕ ФОРМ ПРИ ЗАПИСИ В TENSORARRAY
            state_scalar = tf.squeeze(x_upd, axis=-1)  # [B, 1] → [B]
            state_scalar = tf.ensure_shape(state_scalar, [None])
            innov_scalar = tf.squeeze(innov, axis=[-1, -2])  # [B, 1, 1] → [B]
            innov_scalar = tf.ensure_shape(innov_scalar, [None])

            s_hist = s_hist.write(t, state_scalar)
            i_hist = i_hist.write(t, innov_scalar)
            v_hist = v_hist.write(t, volatility_level)
            f_hist = f_hist.write(t, inflation_factor_flat)
            high_infl_steps = high_infl_steps.write(t, updated_high_inflation_steps)
            corr_adapt_hist = corr_adapt_hist.write(t, correction_adaptive_val)

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 17: ОБНОВЛЕНИЕ ОКНА ИННОВАЦИЙ
            # ════════════════════════════════════════════════════════════════

            # Сдвигаем окно: убираем первый элемент, добавляем новый
            new_innov_window = tf.concat(
                [innov_win[:, 1:], tf.reshape(innov_scalar, [B_batch, 1])],
                axis=1
            )
            new_innov_window = tf.ensure_shape(new_innov_window, [None, innov_window_size])

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 18: ВОЗВРАТ С ПРАВИЛЬНЫМИ РАЗМЕРНОСТЯМИ
            # ════════════════════════════════════════════════════════════════

            new_state = x_upd  # [B, 1]
            new_cov = P_upd  # [B, 1, 1]
            new_vol = new_volatility  # [B]
            new_inf_factor = inflation_factor_flat  # [B]
            new_rem_steps = tf.maximum(0, duration - 1)  # [B]

            return (
                t + 1,  # t
                new_state,  # state [B, 1]
                new_cov,  # cov [B, 1, 1]
                new_vol,  # vol [B]
                new_innov_window,  # innov_win [B, innov_window_size] - обновленное окно
                new_inf_factor,  # inf_factor [B]
                new_rem_steps,  # rem_steps [B]
                last_anom_time_updated,  # last_anom_time [B] - исправлено имя
                s_hist,  # states_hist
                i_hist,  # innovations_hist
                v_hist,  # volatility_levels
                f_hist,  # inflation_factors_hist
                high_infl_steps,  # high_infl_steps_hist
                corr_adapt_hist
            )

        # ════════════════════════════════════════════════════════════════
        # ИНИЦИАЛИЗАЦИЯ ПЕРЕМЕННЫХ ЦИКЛА
        # ════════════════════════════════════════════════════════════════

        loop_vars = [
            tf.constant(0, dtype=tf.int32),
            current_state,  # [B, 1]
            current_covariance,  # [B, 1, 1]
            current_volatility,  # [B]
            innov_window_init,  # ✅ [B, innov_window_size] - ДОБАВЛЕНО ОКНО ИННОВАЦИЙ
            inflation_factor_init,  # [B]
            remaining_steps_init,  # [B]
            last_anomaly_time_init,  # [B]
            states_hist,
            innovations_hist,
            volatility_levels,
            inflation_factors_hist,
            high_infl_steps_hist,
            correction_adaptive_hist,
        ]

        # ════════════════════════════════════════════════════════════════
        # ОПРЕДЕЛЕНИЕ НЕИЗМЕННЫХ ФОРМ
        # ════════════════════════════════════════════════════════════════

        shape_invariants = [
            tf.TensorShape([]),                    # t
            tf.TensorShape([None, 1]),             # state
            tf.TensorShape([None, 1, 1]),          # cov
            tf.TensorShape([None]),                # vol
            tf.TensorShape([None, innov_window_size]),  # innov_win
            tf.TensorShape([None]),                # inf_factor
            tf.TensorShape([None]),                # rem_steps
            tf.TensorShape([None]),                # last_anom_time
            tf.TensorShape(None),                  # states_hist ← изменено!
            tf.TensorShape(None),                  # innovations_hist
            tf.TensorShape(None),                  # volatility_levels
            tf.TensorShape(None),                  # inflation_factors_hist
            tf.TensorShape(None),                  # high_infl_steps_hist,
            tf.TensorShape(None),
        ]

        # ════════════════════════════════════════════════════════════════
        # ИСПОЛНЕНИЕ ЦИКЛА
        # ════════════════════════════════════════════════════════════════

        final_vars = tf.while_loop(
            cond, body, loop_vars, shape_invariants,
            maximum_iterations=T,
            parallel_iterations=1
        )

        # ════════════════════════════════════════════════════════════════
        # РАСПАКОВКА И ФОРМАТИРОВАНИЕ РЕЗУЛЬТАТОВ
        # ════════════════════════════════════════════════════════════════

        (_, final_state, final_covariance, final_volatility, final_innov_window,
         final_inflation_factor, final_remaining_steps, final_last_anom_time,
         s_hist, i_hist, v_hist, f_hist, high_infl_steps_hist,
         correction_adaptive_hist) = final_vars

        # Восстановление формы выходных данных
        states_out = tf.transpose(s_hist.stack(), [1, 0])  # [T, B] → [B, T]
        innovations_out = tf.transpose(i_hist.stack(), [1, 0])  # [T, B] → [B, T]
        volatility_out = tf.transpose(v_hist.stack(), [1, 0])  # [T, B] → [B, T]
        inflation_factors_out = tf.transpose(f_hist.stack(), [1, 0])  # [T, B] → [B, T]
        correction_adaptive_out = tf.transpose(correction_adaptive_hist.stack(), [1, 0])  # [T, B] → [B, T]

        result = (
            tf.expand_dims(states_out, axis=-1),  # [B, T] → [B, T, 1]
            tf.expand_dims(innovations_out, axis=-1),  # [B, T] → [B, T, 1]
            tf.expand_dims(volatility_out, axis=-1),  # [B, T] → [B, T, 1]
            tf.expand_dims(inflation_factors_out, axis=-1),  # [B, T] → [B, T, 1]
            final_state,  # [B, 1]
            final_covariance,  # [B, 1, 1]
            tf.expand_dims(correction_adaptive_out, axis=-1),  # [B, T, 1]
        )

        return result

    @tf.function
    def compute_loss(
        self,
        predictions: tf.Tensor,
        targets: tf.Tensor,
        volatility_levels: tf.Tensor,
        inflation_factors: tf.Tensor,
        ukf_params: Dict[str, tf.Tensor],
        calibration_loss: tf.Tensor,
        entropy_loss: tf.Tensor = 0.0,
        regime_info: Optional[Dict[str, tf.Tensor]] = None,
        training: bool = False,
    ):
        """
        Оптимизированный compute_loss:
        - Убраны Python-условия внутри @tf.function
        - Упрощены регуляризаторы
        - Убрана problematic self.last_ukf_params assignment
        """
        # 1) MSE
        mse_loss = tf.reduce_mean(tf.square(predictions - targets))

        # 2) Stability по inflation (держим около 1.0)
        inflation_clipped = tf.clip_by_value(inflation_factors - 1.0, -5.0, 5.0)
        stability_penalty = tf.constant(0.05, tf.float32) * tf.reduce_mean(tf.square(inflation_clipped))

        # 3) Spectral reg (diff UKF) — только если компонент существует
        spectral_reg = tf.constant(0.0, dtype=tf.float32)
        if hasattr(self, "diff_ukf_component") and self.diff_ukf_component is not None:
            spectrum = self.diff_ukf_component.get_spectrum_info()
            min_eig = spectrum["min_eigenvalue"]
            spectral_reg = tf.constant(0.01, tf.float32) * tf.reduce_mean(tf.nn.relu(1e-4 - min_eig))

        # 4) Regime selector reg
        selector_reg = tf.constant(0.0, dtype=tf.float32)
        if hasattr(self, "regime_selector") and self.regime_selector is not None:
            # Штраф на масштабы
            selector_reg = selector_reg + tf.constant(0.02, tf.float32) * tf.reduce_mean(
                tf.square(self.regime_selector.regime_scales - 2.5)
            )
            # Штраф на center_logits (если learnable)
            if getattr(self.regime_selector, "learnable_centers", False):
                if hasattr(self.regime_selector, "center_logits"):
                    selector_reg = selector_reg + tf.constant(0.001, tf.float32) * tf.reduce_sum(
                        tf.square(self.regime_selector.center_logits)
                    )

        # 5) Entropy penalty по режимам — ГИБРИДНЫЙ ПОДХОД v2 (ИСПРАВЛЕНО)
        entropy_penalty = tf.constant(0.0, dtype=tf.float32)
        if hasattr(self, "regime_selector") and self.regime_selector is not None:
            if regime_info is None:
                current_vol = tf.reshape(volatility_levels[:, -1, :], [-1])
                regime_info = self.regime_selector.assign_soft_regimes(current_vol)

            soft_weights = regime_info["soft_weights"]  # [B, 3]

            # 5.1) Энтропия распределения режимов — ОСТАВЛЯЕМ (уникальная логика compute_loss)
            current_entropy = tf.reduce_mean(regime_info["entropy"])
            max_entropy = tf.math.log(3.0)
            normalized_entropy = current_entropy / (max_entropy + 1e-8)
            target_entropy_normalized = tf.constant(0.85, tf.float32)
            entropy_deviation = tf.nn.relu(target_entropy_normalized - normalized_entropy)

            # ✅ ОСТАВЛЯЕМ: энтропийный штраф (только здесь)
            entropy_penalty = tf.constant(35.0, tf.float32) * tf.square(entropy_deviation)

            # ❌ УДАЛЕНО: 5.2) Штраф за доминирование одного режима (>80%)
            # dominance_penalty и empty_penalty уже вычисляются в train_step как часть regime_loss
            # max_regime_weight = tf.reduce_max(tf.reduce_mean(soft_weights, axis=0))
            # dominance_penalty = tf.nn.relu(max_regime_weight - 0.80)
            # entropy_penalty = entropy_penalty + tf.constant(25.0, tf.float32) * tf.square(dominance_penalty)

            # ❌ УДАЛЕНО: 5.3) Штраф за пустые режимы (<5%)
            # min_regime_weight = tf.reduce_min(tf.reduce_mean(soft_weights, axis=0))
            # empty_penalty = tf.nn.relu(0.05 - min_regime_weight)
            # entropy_penalty = entropy_penalty + tf.constant(15.0, tf.float32) * tf.square(empty_penalty)

            # 5.4) Contrastive loss (если метод реализован) — ОСТАВЛЯЕМ (уникальная логика)
            # Removed call to undefined method get_regime_contrastive_loss

        # 6) Total loss (упрощённый)
        total_loss = (
            mse_loss +
            calibration_loss +
            0.05 * stability_penalty +          # оставили
            0.01 * spectral_reg +               # оставили (малый вес)
            selector_reg +                      # режимы
            entropy_penalty +                   # главный регуляризатор (15.0)
            (self.lambda_entropy * tf.cast(entropy_loss, tf.float32))  # LSTM энтропия
        )

        # Safety NaN/Inf
        total_loss = tf.where(tf.math.is_nan(total_loss), tf.constant(1e6, tf.float32), total_loss)
        total_loss = tf.where(tf.math.is_inf(total_loss), tf.constant(1e6, tf.float32), total_loss)

        return total_loss

    def compute_target_coverage(self, volatility_level: tf.Tensor, training: bool = True) -> tf.Tensor:
        """
        Единая точка вычисления целевого покрытия для ВСЕХ режимов работы.
        Комбинирует:
          - глобальную адаптацию по непрерывной волатильности
          - обучаемые цели покрытия для каждого режима (LOW/MID/HIGH)
        """
        # --- 1) Базовая цель по уровню волатильности ---
        base_confidence = 0.88 # ← ОДИНАКОВО для train/val
        confidence_range = 0.10
        confidence_ceil = base_confidence + confidence_range / 2.0  # ~0.89 (train), ~0.89 (val)
        confidence_floor = base_confidence - confidence_range / 2.0  # ~0.79 (train), ~0.79 (val)

        v_clipped = tf.clip_by_value(volatility_level, 0.0, 1.0)
        target_coverage_base = confidence_ceil - (confidence_ceil - confidence_floor) * v_clipped

        # --- 2) Получаем принадлежность к режимам ---
        regime_info = self.regime_selector.assign_soft_regimes(volatility_level)
        soft_weights = regime_info["soft_weights"]  # [B, K]

        # --- 3) Обучаемые цели покрытия для каждого режима ---
        # Конвертируем логиты в вероятности, но используем как коэффициенты смещения
        cov_probs = tf.nn.softmax(self.target_cov_logits)  # [3]

        # Преобразуем обратно в вероятности покрытия: p = sigmoid(logit), но проще использовать напрямую
        # Линейная комбинация: p_k = 0.7 + 0.25 * sigmoid(z_k), где z_k ~ N(0,1)
        target_cov_values = confidence_floor + (confidence_ceil - confidence_floor) * tf.sigmoid(self.target_cov_logits)

        # --- 4) Взвешенная цель по режимам ---
        target_coverage_mode = tf.reduce_sum(
            soft_weights * target_cov_values[tf.newaxis, :],  # [1, 3]
            axis=1
        )  # [B]

        # - 5) Смесь двух стратегий с обучаемым коэффициентом -
        # Moved initialization of coverage_mixing_alpha to __init__ to avoid creating variables inside tf.function
        # Преобразуем логит в значение [0,1] через sigmoid
        alpha = tf.sigmoid(self.coverage_mixing_alpha)
        target_coverage = (alpha * target_coverage_mode + (1 - alpha) * target_coverage_base)

        return tf.clip_by_value(target_coverage, confidence_floor, confidence_ceil)

    def _get_calibration_params(self, volatility_level, student_t_config=None, correction_adaptive=None, training=True):
        """
        Возвращает student_t_config, пригодный для _calibrate_confidence_interval:
        - все скаляры -> [B] (берём последний timestep)
        - regime_soft_weights -> [B, K] (берём последний timestep)
        Также возвращает target_coverage [B] и regime_info от финальной волатильности.
        """

        # -------- 1) volatility_level -> v: [B] --------
        vol = tf.convert_to_tensor(volatility_level, dtype=tf.float32)

        # Предполагаем, что первая размерность всегда батч (B).
        B = tf.shape(vol)[0]

        # Универсально: [B], [B,1], [B,T], [B,T,1] -> [B, ?, 1]
        vol_bt1 = tf.reshape(vol, [B, -1, 1])

        # Берём последний timestep -> [B]
        v = tf.squeeze(vol_bt1[:, -1, :], axis=-1)

        # -------- 2) target coverage + regime info --------
        target_coverage = self.compute_target_coverage(v, training=training)  # ожидаем [B] или [B,1]
        target_coverage = tf.clip_by_value(
            tf.reshape(tf.squeeze(target_coverage), [B]),
            0.805, 0.955
        )

        regime_info = self.regime_selector.assign_soft_regimes(v)  # soft_weights обычно [B, K]

        # -------- 3) подготовка конфига --------
        cfg = {} if student_t_config is None else dict(student_t_config)
        cfg["target_coverage"] = target_coverage

        # Helpers (без tf.cond/tf.case со слайсами по "возможным" осям)
        def _to_B(x):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            x_bt1 = tf.reshape(x, [B, -1, 1])          # [B,*] -> [B,?,1]
            return tf.squeeze(x_bt1[:, -1, :], axis=-1)  # [B]

        def _to_BK(x, K):
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            x_btk = tf.reshape(x, [B, -1, K])          # [B,K] or [B,T,K] -> [B,?,K]
            return x_btk[:, -1, :]                     # [B,K]

        # correction_adaptive может быть [B,1] или [B,T,1]
        if correction_adaptive is not None:
            cfg["correction_adaptive"] = _to_B(correction_adaptive)

        # -------- 4) схлопываем student-t параметры до [B] --------
        for k in [
            "dof_base", "dof_sensitivity",
            "asymmetry_pos", "asymmetry_neg",
            "calibration_sensitivity",
            "tail_weight_pos", "tail_weight_neg",
            "confidence_floor", "confidence_ceil",
        ]:
            if (k in cfg) and (cfg[k] is not None):
                cfg[k] = _to_B(cfg[k])

        # -------- 5) regime_scale: -> [B] --------
        if ("regime_scale" in cfg) and (cfg["regime_scale"] is not None):
            cfg["regime_scale"] = _to_B(cfg["regime_scale"])  # [B,T,1] -> [B]
        else:
            # get_regime_scales обычно даёт [B,1] (или совместимо с _to_B)
            rs = self.regime_selector.get_regime_scales(regime_info["soft_weights"])
            cfg["regime_scale"] = _to_B(rs)

        # -------- 6) regime_soft_weights: -> [B,K] --------
        if ("regime_soft_weights" in cfg) and (cfg["regime_soft_weights"] is not None):
            # Поддержка обоих вариантов названия поля с K
            if hasattr(self.regime_selector, "num_regimes"):
                K = int(self.regime_selector.num_regimes)
            else:
                K = int(self.regime_selector.numregimes)
            cfg["regime_soft_weights"] = _to_BK(cfg["regime_soft_weights"], K)
        else:
            cfg["regime_soft_weights"] = regime_info["soft_weights"]

        return cfg, target_coverage, regime_info

    @tf.function
    def _compute_calibration_loss(
        self, ci_lower, ci_upper, y_target, y_for_filtering,
        volatility_levels, target_coverage, training=False,
        regime_info=None
    ):
        """
        Вычисляет калибровочную потерю для доверительных интервалов.
        ✅ ИСПРАВЛЕНО: Shape mismatch, log-стабильность, режим-специфичные веса
        """
        # === 1) Flatten для поэлементных операций ===
        y_target_flat = tf.reshape(y_target, [-1])              # [B*T]
        ci_min_flat = tf.reshape(ci_lower, [-1])                # [B*T]
        ci_max_flat = tf.reshape(ci_upper, [-1])                # [B*T]
        batch_size = tf.shape(y_target)[0]
        
        # === 2) Scale window (волатильность данных) ===
        vol_raw = tf.math.reduce_std(y_for_filtering[:, -20:], axis=1)  # [B]
        std_floor = tf.constant(1e-3, tf.float32)
        vol = tf.maximum(vol_raw, std_floor)                            # [B]
        y_std_batch = tf.reduce_mean(vol)                               # scalar
        
        # === 3) Soft coverage surrogate ===
        margin = tf.maximum(0.05 * y_std_batch, 1e-6)
        soft_lower = tf.sigmoid((y_target_flat - ci_min_flat) / margin)
        soft_upper = tf.sigmoid((ci_max_flat - y_target_flat) / margin)
        soft_cov = soft_lower * soft_upper
        actual_coverage = tf.reduce_mean(soft_cov)
        
        # === 4) Target coverage ===
        target_coverage_mean = tf.reduce_mean(target_coverage)
        
        # === 5) Under/over penalties — АСИММЕТРИЧНЫЙ ШТРАФ ===
        under_gap = target_coverage_mean - actual_coverage
        over_gap  = actual_coverage - target_coverage_mean
        
        under_base = tf.constant(15.0, tf.float32)
        over_base  = tf.constant(8.0, tf.float32)
        
        is_under = under_gap > 0.0
        under_w = tf.where(is_under, under_base * 2.0, under_base * 0.5)
        over_w  = tf.where(is_under, over_base  * 0.5, over_base  * 1.5)
        
        under_pen = under_w * tf.square(tf.nn.relu(under_gap))
        over_pen  = over_w  * tf.square(tf.nn.relu(over_gap))
        
        # === 6) Per-regime calibration loss ✅ ИСПРАВЛЕНО: Shape consistency ===
        regime_coverage_loss = 0.0
        if regime_info is not None and 'soft_weights' in regime_info:
            soft_weights = regime_info['soft_weights']  # [B, 3] или [B*T, 3]
            
            # 🔑 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Приведение к одной размерности
            if len(soft_weights.shape) == 3:  # [B, T, 3]
                soft_weights = soft_weights[:, -1, :]  # [B, 3]
            
            covered = tf.cast(
                (y_target_flat >= ci_min_flat) & (y_target_flat <= ci_max_flat),
                tf.float32
            )  # [B*T]
            
            # 🔑 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Reshape covered для совместимости с soft_weights
            covered_reshaped = tf.reshape(covered, [batch_size, -1])  # [B, T]
            covered_mean = tf.reduce_mean(covered_reshaped, axis=1)   # [B]
            
            regime_weights = tf.constant([1.5, 2.0, 3.0], tf.float32)  # [LOW, MID, HIGH]
            target_cov_values = 0.7 + 0.25 * tf.sigmoid(self.target_cov_logits)  # [3]
            
            for k in range(3):
                regime_mask = soft_weights[:, k]  # [B]
                regime_coverage = tf.reduce_sum(covered_mean * regime_mask) / (
                    tf.reduce_sum(regime_mask) + 1e-8
                )
                regime_target = target_cov_values[k]
                regime_gap = tf.abs(regime_coverage - regime_target)
                regime_weight = regime_weights[k]
                regime_coverage_loss += regime_weight * tf.square(regime_gap)
            
            regime_coverage_loss = regime_coverage_loss * 15.0
        
        raw_calibration_loss = under_pen + over_pen + regime_coverage_loss
        
        # === 7) Width diagnostics ✅ ИСПРАВЛЕНО: Защита от log(0) ===
        width_ps = tf.maximum(ci_max_flat - ci_min_flat, 0.0)  # [B*T]
        width_ps_reshaped = tf.reshape(width_ps, [batch_size, -1])  # [B, T]
        width_mean = tf.reduce_mean(width_ps_reshaped, axis=1)  # [B]
        
        width_ratio_ps = width_mean / (vol + 1e-8)  # [B]
        
        k = tf.constant(10.0, tf.float32)
        w = tf.clip_by_value(vol_raw / (k * std_floor), 0.0, 1.0)  # [B]
        w_sum = tf.reduce_sum(w) + 1e-8
        width_ratio = tf.reduce_sum(w * width_ratio_ps) / w_sum  # ✅ СКАЛЯР
        
        avg_volatility = tf.reduce_mean(tf.clip_by_value(
            tf.reshape(volatility_levels[:, -1, :], [-1]), 0.0, 1.0
        ))
        target_width_ratio = 1.8 * (1.0 + 0.2 * avg_volatility)
        target_width_ratio = tf.clip_by_value(target_width_ratio, 1.5, 2.5)
        
        # 🔑 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Двойная защита от log(0)
        ratio_safe = tf.clip_by_value(
            (width_ratio + 1e-8) / (target_width_ratio + 1e-8),
            1e-6, 1e6
        )
        wide_err = tf.square(tf.nn.relu(tf.math.log(ratio_safe)))
        narrow_err = tf.square(tf.nn.relu(tf.math.log(1.0 / ratio_safe)))
        width_error = wide_err + 0.10 * narrow_err
        
        return raw_calibration_loss, actual_coverage, width_ratio, target_width_ratio, width_error
    
    @tf.function(jit_compile=False)
    def _calibrate_confidence_interval(
        self,
        forecast: tf.Tensor,
        stddev: tf.Tensor,
        volatility_level: tf.Tensor,
        student_t_config: Dict[str, tf.Tensor],
        innovations: Optional[tf.Tensor] = None,
        regime_assignment: Optional[tf.Tensor] = None,
        true_values: Optional[tf.Tensor] = None
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        """
        Вычисляет адаптивные доверительные интервалы (CI) с режимно-зависимым контролем ширины.
        ✅ ИСПРАВЛЕНО: log-стабильность, shape consistency, защита от NaN
        """
        batch_size = tf.shape(forecast)[0]
        
        # ===== 1) NORMALIZE INPUTS =====
        forecast = tf.reshape(tf.squeeze(forecast), [batch_size])
        stddev = tf.reshape(tf.squeeze(stddev), [batch_size])
        volatility_level = tf.reshape(tf.squeeze(volatility_level), [batch_size])
        stddev = tf.maximum(stddev, 1e-8)
        
        # ===== 2) EXTRACT PARAMS (safe for tf.function) =====
        def _as_batch_last(key, default_B):
            x = student_t_config.get(key, default_B)
            x = tf.convert_to_tensor(x, dtype=tf.float32)
            x_flat = tf.reshape(x, [-1])
            n = tf.size(x_flat)
            def _scalar_case():
                return tf.fill([batch_size], x_flat[0])
            def _batched_case():
                x2 = tf.reshape(x_flat, [batch_size, -1])
                return x2[:, -1]
            return tf.cond(tf.equal(n, 1), _scalar_case, _batched_case)
        
        target_coverage = _as_batch_last("target_coverage", tf.fill([batch_size], 0.90))
        target_coverage = tf.clip_by_value(target_coverage, 0.805, 0.955)
        
        tail_weight_pos = _as_batch_last("tail_weight_pos", tf.ones([batch_size], tf.float32))
        tail_weight_neg = _as_batch_last("tail_weight_neg", tf.ones([batch_size], tf.float32))
        asymmetry_pos   = _as_batch_last("asymmetry_pos",   tf.ones([batch_size], tf.float32))
        asymmetry_neg   = _as_batch_last("asymmetry_neg",   tf.ones([batch_size], tf.float32))
        dof_base        = _as_batch_last("dof_base",        tf.fill([batch_size], 6.0))
        dof_sensitivity = _as_batch_last("dof_sensitivity", tf.fill([batch_size], 0.5))
        regime_scale = _as_batch_last("regime_scale", tf.ones([batch_size], tf.float32))
        correction_adaptive = _as_batch_last("correction_adaptive", tf.ones([batch_size], tf.float32))
        correction_adaptive = tf.clip_by_value(correction_adaptive, 1.0, 3.0)
        
        # ===== 3) ADAPTIVE DOF =====
        v = tf.clip_by_value(volatility_level, 0.0, 1.0)
        dof_floor = 6.0 - 2.0 * v
        dof_floor = tf.clip_by_value(dof_floor, 4.0, 8.0)
        x = tf.nn.relu(v - 0.3) / 0.7
        x = tf.clip_by_value(x, 0.0, 1.0)
        decrease = dof_sensitivity * 2.0 * x
        dof_adjusted = dof_base - decrease
        dof_adjusted = tf.clip_by_value(dof_adjusted, dof_floor, 15.0)
        
        # ===== 4) t-QUANTILE APPROX (Cornish–Fisher) =====
        prob_lower = (1.0 - target_coverage) / 2.0
        prob_lower = tf.maximum(prob_lower, 0.001)
        z = tf.sqrt(2.0) * tf.math.erfinv(2.0 * (1.0 - prob_lower) - 1.0)
        nu = tf.maximum(dof_adjusted, 3.0)
        inv_nu = 1.0 / nu
        inv_nu2 = inv_nu * inv_nu
        inv_nu3 = inv_nu2 * inv_nu
        z2 = z * z
        z3 = z2 * z
        z5 = z3 * z2
        z7 = z5 * z2
        t_approx = (
            z
            + (z3 + z) * (inv_nu / 4.0)
            + (5.0 * z5 + 16.0 * z3 + 3.0 * z) * (inv_nu2 / 96.0)
            + (3.0 * z7 + 19.0 * z5 + 17.0 * z3 - 15.0 * z) * (inv_nu3 / 384.0)
        )
        z_upper = t_approx
        z_lower = -t_approx
        
        # ===== 5) BASE MARGINS =====
        margin_lower_base = stddev * tf.abs(z_lower)
        margin_upper_base = stddev * tf.abs(z_upper)
        
        # ===== 6) INNOVATIONS-BASED ASYMMETRY (optional) =====
        center_shift = tf.zeros([batch_size], dtype=tf.float32)
        tail_adj_pos = tf.ones([batch_size], dtype=tf.float32)
        tail_adj_neg = tf.ones([batch_size], dtype=tf.float32)
        
        if innovations is not None:
            innov = tf.squeeze(innovations)
            if innov.shape.rank == 1:
                innov = tf.reshape(innov, [batch_size, -1])
            pos_mask = tf.cast(innov > 0, tf.float32)
            neg_mask = tf.cast(innov <= 0, tf.float32)
            abs_innov = tf.abs(innov)
            pos_sum = tf.reduce_sum(pos_mask * abs_innov, axis=1)
            pos_count = tf.reduce_sum(pos_mask, axis=1) + 1e-8
            pos_mag = pos_sum / pos_count
            neg_sum = tf.reduce_sum(neg_mask * abs_innov, axis=1)
            neg_count = tf.reduce_sum(neg_mask, axis=1) + 1e-8
            neg_mag = neg_sum / neg_count
            asym_ratio = (pos_mag - neg_mag) / (pos_mag + neg_mag + 1e-8)
            center_shift = tf.clip_by_value(stddev * asym_ratio * 0.3, -0.3 * stddev, 0.3 * stddev)
            ratio = (pos_mag + 1e-8) / (neg_mag + 1e-8)
            ratio = tf.clip_by_value(ratio, 0.5, 2.0)
            tail_adj_pos = tf.sqrt(ratio)
            tail_adj_neg = 1.0 / tf.sqrt(ratio)
        
        # ===== 7) APPLY MULTIPLIERS =====
        margin_lower = (
            margin_lower_base *
            tail_weight_neg * asymmetry_neg *
            regime_scale * correction_adaptive *
            tail_adj_neg
        )
        margin_upper = (
            margin_upper_base *
            tail_weight_pos * asymmetry_pos *
            regime_scale * correction_adaptive *
            tail_adj_pos
        )
        
        # ===== 8) РЕЖИМНО-ЗАВИСИМЫЙ КОНТРОЛЬ ШИРИНЫ =====
        if hasattr(self, "max_width_factors_logits") and self.max_width_factors_logits is not None:
            factors_raw = tf.nn.softplus(self.max_width_factors_logits) + 1.0  # [3]
            if regime_assignment is not None:
                target_width_per_sample = tf.gather(factors_raw, regime_assignment)  # [B]
            else:
                target_width_per_sample = tf.fill([batch_size], tf.reduce_mean(factors_raw))
        else:
            target_width_per_sample = tf.fill([batch_size], 2.0)
        
        width_raw = margin_lower + margin_upper
        width_ratio = width_raw / (stddev + 1e-8)
        
        # ===== 9) ШТРАФ ЗА ШИРИНУ ✅ ИСПРАВЛЕНО: Защита от log(0) =====
        wide_threshold = target_width_per_sample * 1.2
        narrow_threshold = target_width_per_sample * 0.8
        
        # 🔑 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: Двойной clip для ratio
        wide_ratio_safe = tf.clip_by_value(
            width_ratio / (wide_threshold + 1e-8),
            1e-6, 1e6
        )
        wide_penalty = tf.where(
            width_ratio > wide_threshold,
            0.8 * tf.square(tf.math.log(wide_ratio_safe)),
            0.0
        )
        
        narrow_ratio_safe = tf.clip_by_value(
            (narrow_threshold + 1e-8) / (width_ratio + 1e-8),
            1e-6, 1e6
        )
        narrow_penalty = tf.where(
            width_ratio < narrow_threshold,
            0.5 * tf.square(tf.math.log(narrow_ratio_safe)),
            0.0
        )
        
        width_penalty_value = tf.reduce_mean(wide_penalty + narrow_penalty)
        
        # ===== 10) ФИНАЛЬНЫЕ ПРОВЕРКИ БЕЗОПАСНОСТИ =====
        min_margin = 0.25 * stddev
        margin_lower = tf.maximum(margin_lower, min_margin)
        margin_upper = tf.maximum(margin_upper, min_margin)
        
        width_is_valid = (
            tf.math.is_finite(margin_lower) &
            tf.math.is_finite(margin_upper) &
            (margin_lower >= 0.0) &
            (margin_upper >= 0.0)
        )
        margin_lower = tf.where(width_is_valid, margin_lower, min_margin)
        margin_upper = tf.where(width_is_valid, margin_upper, min_margin)
        
        # ===== 11) ВЫЧИСЛЕНИЕ ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ =====
        ci_lower = forecast - margin_lower + center_shift
        ci_upper = forecast + margin_upper + center_shift
        ci_min = tf.minimum(ci_lower, ci_upper)
        ci_max = tf.maximum(ci_lower, ci_upper)
        
        return ci_min, ci_max, target_coverage, width_penalty_value

    @tf.function
    def train_step(self, X_batch, y_for_filtering_batch, y_target_batch, regime_labels_batch,
                   initial_state, initial_covariance):
        """
        Шаг обучения (LSTM-only): LSTM -> 37 params -> (vol_context, ukf_params, inflation_config, student_t_config)
        -> adaptive_ukf_filter -> explicit predict -> CI calibration -> loss -> grads.
        
        ✅ ИСПРАВЛЕНО:
        - target_coverage_val scope (вынесен перед использованием)
        - Удалено дублирование domain penalty (оставлен только sigmoid подход)
        - Все tf.cond возвращают одинаковые типы
        - Shape consistency для всех тензоров
        - Domain penalty усилен (weight=25.0, slope=15.0, threshold=0.8)
        - Val calibration loss с tf.stop_gradient
        - Temperature clip обновлён до [0.3, 10.0]
        """
        B = tf.shape(X_batch)[0]
        
        # Safety: счетчик может отсутствовать если не создали в __init__
        if not hasattr(self, "_step_counter"):
            self._step_counter = tf.Variable(0, trainable=False, dtype=tf.int64, name="step_counter")
        
        with tf.device(self.device):
            with tf.GradientTape(persistent=False) as tape:
                
                # 0) Debug helpers (минимально; без Python-ветвлений по Tensor)
                if not hasattr(self, "grad_debug_enabled"):
                    self.grad_debug_enabled = True
                if not hasattr(self, "grad_debug_every"):
                    self.grad_debug_every = 20
                if not hasattr(self, "grad_debug_step0"):
                    self.grad_debug_step0 = 0
                if not hasattr(self, "grad_debug_max_vars"):
                    self.grad_debug_max_vars = 40
                
                def _is_debug_step():
                    step = self._step_counter
                    cond1 = tf.cast(self.debug_mode, tf.bool)
                    cond2 = tf.cast(self.grad_debug_enabled, tf.bool)
                    cond3 = step >= self.grad_debug_step0
                    cond4 = tf.equal(tf.math.floormod(step, self.grad_debug_every), 0)
                    
                    # ← НОВОЕ: Всегда логировать на эпохе 10 (критический переход)
                    epoch_10_start = 200  # Примерный step для начала эпохи 10
                    epoch_10_end = 250    # Примерный step для конца эпохи 10
                    cond5 = (step >= epoch_10_start) & (step <= epoch_10_end)
                    
                    return tf.logical_or(
                        tf.logical_and(tf.logical_and(cond1, cond2), tf.logical_and(cond3, cond4)),
                        cond5
                    )
                
                # 1) FORWARD PASS LSTM
                lstm_outputs = self.model(X_batch, training=True)
                params_output = lstm_outputs["params"]   # [B, T, 37]
                h_lstm2 = lstm_outputs["h_lstm2"]        # [B, T, 128]
                
                # 2) ENTROPY REG
                entropy_loss = self.entropy_regularizer.compute_entropy_loss(h_lstm2)
                
                # 3) PROCESS LSTM OUTPUTS (единственный источник параметров)
                vol_context, ukf_params, inflation_config, student_t_config = self.process_lstm_output(params_output)
                
                # 4) ADAPTIVE UKF FILTER
                results = self.adaptive_ukf_filter(
                    X_batch,
                    y_for_filtering_batch,
                    vol_context,
                    ukf_params,
                    inflation_config,
                    student_t_config,
                    initial_state,
                    initial_covariance
                )
                
                x_filtered = results[0]                  # [B, T, 1]
                innovations = results[1]                 # [B, T, 1]
                volatility_levels = results[2]           # [B, T, 1]
                inflation_factors = results[3]           # [B, T, 1]
                final_state = results[4]                 # [B, 1]
                final_covariance = results[5]            # [B, 1, 1]
                correction_adaptive_hist = results[6]    # [B, T, 1]
                
                normalized_innovations = tf.abs(innovations[:, -10:, :])  # [B, 10, 1]
                
                # 5) EXPLICIT ONE-STEP PREDICT (t+1)
                final_volatility = tf.reshape(volatility_levels[:, -1, :], [-1])  # [B]
                t_last = tf.shape(ukf_params["q_base"])[1] - 1
                
                q_base_final = tf.gather(ukf_params["q_base"], t_last, axis=1)                 # [B,1]
                q_sensitivity_final = tf.gather(ukf_params["q_sensitivity"], t_last, axis=1)   # [B,1]
                q_floor_final = tf.gather(ukf_params["q_floor"], t_last, axis=1)               # [B,1]
                
                relax_base_final = tf.gather(ukf_params["relax_base"], t_last, axis=1)         # [B,1]
                relax_sensitivity_final = tf.gather(ukf_params["relax_sensitivity"], t_last, axis=1)  # [B,1]
                alpha_base_final = tf.gather(ukf_params["alpha_base"], t_last, axis=1)         # [B,1]
                alpha_sensitivity_final = tf.gather(ukf_params["alpha_sensitivity"], t_last, axis=1)  # [B,1]
                kappa_base_final = tf.gather(ukf_params["kappa_base"], t_last, axis=1)         # [B,1]
                kappa_sensitivity_final = tf.gather(ukf_params["kappa_sensitivity"], t_last, axis=1)  # [B,1]
                
                inf_factor_final = tf.gather(inflation_factors, t_last, axis=1)                # [B,1]
                correction_adaptive = correction_adaptive_hist[:, -1, :]  # [B,1]
                
                # --- НАЧАЛО supervised regime classification ---
                regime_info = self.regime_selector.assign_soft_regimes(final_volatility)
                soft_weights = regime_info["soft_weights"]  # [B, K]
                logits = regime_info["logits"]         # [B, K]

                regime_assignment = regime_info.get("regime_assignment", None)
                
                # Используем переданный аргумент — НЕ ПЕРЕЗАПИСЫВАЕМ!
                # regime_labels_batch уже доступен → просто используем
                
                # 1. Вычисляем веса классов
                unique, _, counts = tf.unique_with_counts(regime_labels_batch)
                class_weights = tf.math.reciprocal(tf.cast(counts, tf.float32) + 1e-6)
                class_weights /= tf.reduce_sum(class_weights)
                class_weights *= 3.0  # Нормализация на 3 класса
                
                # 2. Применяем веса к каждому образцу
                sample_weights = tf.gather(class_weights, regime_labels_batch)
                
                # 3. Вычисляем ВЗВЕШЕННУЮ перекрестную энтропию
                per_sample_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=regime_labels_batch,
                    logits=logits
                )
                weighted_ce = per_sample_ce * sample_weights
                regime_ce_loss = tf.reduce_mean(weighted_ce)  # ✅ ВЕСА ПРИМЕНЕНЫ!
                
                # 4. Остальной код без изменений
                rprog = tf.cast(self._step_counter, tf.float32) / 200.0
                lambda_regime_base = 2.0
                lambda_regime = lambda_regime_base * (1.0 - tf.exp(-5.0 * rprog))
                
                separation_loss = self.regime_selector.get_center_separation_loss()
                regime_loss = (
                    lambda_regime * regime_ce_loss +
                    0.1 * separation_loss +
                    0.5 * self.regime_selector.get_regime_entropy_loss(soft_weights)  # ✅ Добавлен entropy loss
                )
                # --- КОНЕЦ regime classification ---
                
                forecast, std_dev, pred_dbg = self._explicit_predict_next_step(
                    final_state,
                    final_covariance,
                    final_volatility,
                    q_base_final, q_sensitivity_final, q_floor_final,
                    inf_factor_final,
                    relax_base_final, relax_sensitivity_final,
                    alpha_base_final, alpha_sensitivity_final,
                    kappa_base_final, kappa_sensitivity_final,
                    regime_assignment=regime_assignment  # ← ДОБАВЛЕНО
                )
                
                # === 6) CALIBRATION PARAMS (только из LSTM + обязательные поля) ===
                student_t_config, target_coverage, regime_info = self._get_calibration_params(
                    final_volatility,
                    student_t_config=student_t_config,
                    correction_adaptive=correction_adaptive,
                    training=True
                )
                target_coverage_mean = tf.reduce_mean(target_coverage)
                
                # 🔑 ГИБРИДНЫЙ ПОДХОД v9: Диагностика смещения прогноза
                # ✅ ИСПРАВЛЕНО: Перемещено ПОСЛЕ создания regime_assignment
                train_bias = tf.reduce_mean(forecast - y_target_batch)
                train_bias_abs = tf.reduce_mean(tf.abs(forecast - y_target_batch))
                
                # Логирование bias каждые 50 шагов
                should_log_bias = tf.equal(tf.math.floormod(self._step_counter, 50), 0)
                
                def _log_bias():
                    # ✅ ТЕПЕРЬ regime_assignment доступен
                    if regime_assignment is not None:
                        bias_per_regime = tf.math.unsorted_segment_mean(
                            forecast - y_target_batch,
                            regime_assignment,
                            num_segments=3
                        )
                    else:
                        bias_per_regime = tf.zeros([3], dtype=tf.float32)
                    
                    tf.print(
                        "[BIAS DEBUG] step=", self._step_counter,
                        "| train_bias=", train_bias,
                        "| train_bias_abs=", train_bias_abs,
                        "| bias_per_regime (L/M/H)=", bias_per_regime
                    )
                    return tf.constant(0)
                
                tf.cond(should_log_bias, _log_bias, lambda: tf.constant(0))
                
                # === 7) CALIBRATE CI ===
                ci_lower, ci_upper, _, width_regularization = self._calibrate_confidence_interval(
                    forecast, std_dev, final_volatility, student_t_config,
                    innovations=innovations[:, -10:, :],
                    regime_assignment=regime_assignment,
                    true_values=y_target_batch
                )
                ci_min = tf.minimum(ci_lower, ci_upper)
                ci_max = tf.maximum(ci_lower, ci_upper)
                
                # === DIAG: batch mix + per-regime calibration (LOSS-CONSISTENT) ===
                def _print_regime_diag():
                    def _pct(x1d, q):
                        xs = tf.sort(x1d)
                        n = tf.shape(xs)[0]
                        idx = tf.cast(tf.math.floor((q / 100.0) * tf.cast(n - 1, tf.float32)), tf.int32)
                        idx = tf.clip_by_value(idx, 0, n - 1)
                        return xs[idx]
                    
                    B_diag = tf.shape(y_for_filtering_batch)[0]
                    
                    # 1) std окна как в loss + floor
                    vol_raw = tf.math.reduce_std(y_for_filtering_batch[:, -20:], axis=1)  # [B]
                    std_floor = tf.constant(1e-3, tf.float32)                              # как в _compute_calibration_loss
                    vol = tf.maximum(vol_raw, std_floor)                                   # [B]
                    
                    ystd_p10 = _pct(vol, 10.0)
                    ystd_p50 = _pct(vol, 50.0)
                    ystd_p90 = _pct(vol, 90.0)
                    
                    # 2) covered per-sample (hard coverage, для DIAG)
                    y_flat = tf.reshape(y_target_batch, [-1])                              # [B]
                    ci_min_flat = tf.reshape(ci_min, [-1])                                 # [B]
                    ci_max_flat = tf.reshape(ci_max, [-1])                                 # [B]
                    covered = tf.cast(
                        (y_flat >= ci_min_flat) & (y_flat <= ci_max_flat),
                        tf.float32
                    )                                                                      # [B]
                    
                    # 3) widthRatio per-sample (unweighted) + веса как в _compute_calibration_loss
                    width_ps = tf.maximum(ci_max_flat - ci_min_flat, 0.0)                  # [B]
                    wr_ps = width_ps / vol                                                 # [B]  (unweighted per-sample)
                    
                    k = tf.constant(10.0, tf.float32)                                      # MUST match _compute_calibration_loss
                    w = tf.clip_by_value(vol_raw / (k * std_floor), 0.0, 1.0)              # [B]
                    w_sum = tf.reduce_sum(w) + 1e-8
                    
                    wr_unweighted = tf.reduce_mean(wr_ps)
                    wr_weighted = tf.reduce_sum(w * wr_ps) / w_sum
                    
                    floor_frac = tf.reduce_mean(tf.cast(vol_raw <= (k * std_floor), tf.float32))
                    w_mean = tf.reduce_mean(w)
                    
                    # 4) режимы датасета
                    ds_regimes = tf.reshape(tf.cast(regime_labels_batch, tf.int32), [-1])  # [B], 0..2
                    
                    def _safe_wmean(x, mask, w_local):
                        m = tf.cast(mask, tf.float32)
                        ww = w_local * m
                        denom = tf.reduce_sum(ww) + 1e-8
                        return tf.math.divide_no_nan(tf.reduce_sum(x * ww), denom)
                    
                    # coverage per regime (взвешенно теми же w — можно оставить так или вернуться к невзвешенному)
                    cov_low  = _safe_wmean(covered, ds_regimes == 0, w)
                    cov_mid  = _safe_wmean(covered, ds_regimes == 1, w)
                    cov_high = _safe_wmean(covered, ds_regimes == 2, w)
                    
                    # widthRatio per regime (взвешенно, как в лоссе)
                    wr_low  = _safe_wmean(wr_ps, ds_regimes == 0, w)
                    wr_mid  = _safe_wmean(wr_ps, ds_regimes == 1, w)
                    wr_high = _safe_wmean(wr_ps, ds_regimes == 2, w)
                    
                    # 5) pred regime + confusion matrix (ds x pred)
                    if ("regime_assignment" in regime_info) and (regime_info["regime_assignment"] is not None):
                        pred_reg = tf.reshape(tf.cast(regime_info["regime_assignment"], tf.int32), [-1])  # [B]
                    else:
                        soft_w_reg = tf.cast(regime_info["soft_weights"], tf.float32)                     # [B,3]
                        pred_reg = tf.argmax(soft_w_reg, axis=-1, output_type=tf.int32)                  # [B]
                    
                    cm_idx = ds_regimes * 3 + pred_reg                                                   # [B]
                    cm_flat = tf.math.bincount(cm_idx, minlength=9, maxlength=9, dtype=tf.int32)
                    cm = tf.reshape(cm_flat, [3, 3])
                    
                    # 6) ds_counts / ds_pcts
                    ds_counts = tf.math.bincount(ds_regimes, minlength=3, maxlength=3, dtype=tf.int32)
                    ds_pcts = tf.cast(ds_counts, tf.float32) / tf.cast(B_diag, tf.float32)
                    
                    tf.print(
                        "\n[REGIME DIAG][TRAIN] step", self._step_counter,
                        "| ds_counts", ds_counts,
                        "| ds_pcts", ds_pcts,
                        "| ystd(p10/p50/p90)", ystd_p10, ystd_p50, ystd_p90,
                        "| cov(low/mid/high)", [cov_low, cov_mid, cov_high],
                        "| widthRatio(low/mid/high)", [wr_low, wr_mid, wr_high],
                        "| cm(ds x pred)=\n", cm,
                        "| wr_unweighted", wr_unweighted,
                        "| wr_weighted", wr_weighted,
                        "| floor_frac", floor_frac,
                        "| w_mean", w_mean,
                    )
                    return tf.constant(0)
                
                _ = tf.cond(_is_debug_step(), _print_regime_diag, lambda: tf.constant(0))
                
                # 8) CALIBRATION LOSS (raw -> normalized -> clipped/weighted)
                base_calibration_weight = 0.18  # ← 18% от MSE (БЫЛО 0.12)
                warmup_steps = 400  # ← ↑ 250 → 400 (более плавный warmup)
                progress = tf.cast(self._step_counter, tf.float32) / tf.cast(warmup_steps, tf.float32)
                progress = tf.clip_by_value(progress, 0.0, 1.0)
                current_weight = base_calibration_weight * (1.0 - tf.exp(-2.0 * progress))  # ✅ Медленнее (-2.0 вместо -3.0)
                
                # ✅ ИСПРАВЛЕНО: tf.cond требует одинаковых типов возврата из обеих веток
                should_print = tf.equal(tf.math.floormod(self._step_counter, 50), 0)
                
                def _print_dbg():
                    tf.print(
                        '[WARMUP DBG] step=', self._step_counter,
                        '| progress=', progress,
                        '| calib_weight=', base_calibration_weight * (1.0 - tf.exp(-2.0 * progress))  # ✅ Унифицировано на -2.0
                    )
                    return tf.constant(0)  # ← Явно возвращаем tf.constant(0)
                
                def _no_print():
                    return tf.constant(0)  # ← Тоже tf.constant(0)
                
                tf.cond(should_print, _print_dbg, _no_print)  # ✅ Теперь типы совпадают (оба int32)
                
                mse_loss_for_normalization = tf.reduce_mean(tf.square(forecast - y_target_batch))
                
                raw_calibration_loss, actual_coverage, width_ratio, target_width_ratio, width_error = \
                    self._compute_calibration_loss(
                        ci_min, ci_max, y_target_batch, y_for_filtering_batch,
                        volatility_levels, target_coverage, training=True,
                        regime_info=regime_info  # ← ДОБАВЛЕНО
                    )
                
                # ✅ Width-only сигнал с warmup — УСИЛЕННЫЙ РЕЖИМНО-ЗАВИСИМЫЙ ПОДХОД
                width_warmup_steps = tf.constant(150.0, tf.float32)  # ← 150 (БЫЛО 300)
                wprog = tf.cast(self._step_counter, tf.float32) / width_warmup_steps
                wprog = tf.clip_by_value(wprog, 0.0, 1.0)
                
                lambda_width_base = tf.constant(0.10, tf.float32)    # ← 0.10 (БЫЛО 0.08)
                lambda_width = lambda_width_base * (1.0 - tf.exp(-5.0 * wprog))
                
                # === ДОБАВЬТЕ СЮДА УСЛОВНЫЙ ЛОГ ===
                debug_freq = 100
                is_debug_step = tf.equal(tf.math.floormod(self._step_counter, debug_freq), 0)
                
                def _log_debug():
                    tf.print(
                        "🔍 TRAIN DIAG | step", self._step_counter,
                        "| wr:", width_ratio,
                        "| cov:", actual_coverage,
                        "| tgt:", target_coverage_mean,
                        "| w_err:", width_error,
                        "| l_w:", lambda_width
                    )
                    return tf.constant(0)
                
                def _no_log():
                    return tf.constant(0)
                
                tf.cond(is_debug_step, _log_debug, _no_log)
                
                # ✅ НОРМАЛИЗАЦИЯ ОТНОСИТЕЛЬНО MSE ДЛЯ СТАБИЛЬНОСТИ
                mse_scale = tf.stop_gradient(mse_loss_for_normalization)
                scale = tf.stop_gradient(mse_scale / (raw_calibration_loss + 1e-8))
                calibration_loss_normalized = raw_calibration_loss * scale
                # ✅ ОГРАНИЧЕНИЕ ВЕСА КАЛИБРОВКИ (не более 50% от MSE)
                max_calib_weight = 0.5 * mse_scale
                x = current_weight * calibration_loss_normalized
                calibration_loss_clipped = tf.minimum(x, max_calib_weight)
                
                # 🔑 ГИБРИДНЫЙ ПОДХОД v8: НОРМАЛИЗОВАННЫЙ штраф за train/val coverage gap
                # ✅ ИСПРАВЛЕНО: Вес нормализован относительно MSE для стабильности
                if hasattr(self, '_prev_val_coverage') and self._prev_val_coverage is not None:
                    val_cov_estimate = tf.stop_gradient(self._prev_val_coverage)
                    train_val_gap = tf.abs(actual_coverage - val_cov_estimate)
                    # ✅ НОРМАЛИЗОВАННЫЙ ВЕС (0.5 × MSE вместо фиксированного 200.0)
                    gap_penalty = tf.constant(0.5, tf.float32) * mse_loss_for_normalization * tf.square(train_val_gap)
                    # ✅ Защита от NaN/Inf
                    gap_penalty = tf.where(
                        tf.math.is_finite(gap_penalty),
                        gap_penalty,
                        tf.constant(0.0, tf.float32)
                    )
                    calibration_loss_clipped = calibration_loss_clipped + gap_penalty
                    
                    # Логирование gap каждые 50 шагов
                    should_log_gap = tf.equal(tf.math.floormod(self._step_counter, 50), 0)
                    def _log_gap():
                        tf.print(
                            "[COV GAP DEBUG] step=", self._step_counter,
                            "| train_cov=", actual_coverage,
                            "| val_cov=", val_cov_estimate,
                            "| gap=", train_val_gap,
                            "| penalty=", gap_penalty
                        )
                        return tf.constant(0)
                    tf.cond(should_log_gap, _log_gap, lambda: tf.constant(0))
                
                # 9) TOTAL LOSS
                loss = self.compute_loss(
                    forecast,
                    y_target_batch,
                    volatility_levels,
                    inflation_factors,
                    ukf_params,
                    calibration_loss_clipped,
                    entropy_loss,
                    regime_info=regime_info,
                    training=True
                )
                
                # ✅ Улучшенная регуляризация regime_scales
                scales = self.regime_selector.regime_scales
                target_scales = tf.constant([2.96, 4.44, 6.16], dtype=tf.float32)
                
                # 1. Квадратичная регуляризация к целевым значениям
                scale_reg1 = 0.05 * tf.reduce_mean(tf.square(scales - target_scales)) 
                
                # 2. Регуляризация отношений между масштабами (сохранение пропорций)
                scale_ratios = scales[1:] / scales[:-1]
                target_ratios = target_scales[1:] / target_scales[:-1]
                ratio_reg = 0.05 * tf.reduce_mean(tf.square(scale_ratios - target_ratios))
                
                # 3. L2-регуляризация для стабильности абсолютных значений
                l2_reg = 1e-4 * tf.reduce_sum(tf.square(scales))
                
                # Общий регуляризационный терм
                scale_reg = scale_reg1 + ratio_reg + l2_reg
                
                loss = loss + regime_loss + scale_reg
                
                # ✅ Разбиваем на компоненты
                base_width_loss = lambda_width * tf.cast(width_error, tf.float32)
                high_regime_penalty_loss = 0.0
                ci_regularization_loss = 0.5 * width_regularization
                
                if regime_assignment is not None:
                    is_high_regime = tf.cast(tf.equal(regime_assignment, 2), tf.float32)
                    
                    # Штраф за слишком широкие интервалы в высоковолатильном режиме (оставляем как было)
                    wide_high_penalty = is_high_regime * tf.nn.relu(width_ratio - 4.0)
                    
                    # НОВЫЙ: Усиленный штраф за слишком узкие интервалы в высоковолатильном режиме
                    narrow_threshold_high = 0.7  # более высокий порог для высоковолатильного режима
                    narrow_high_penalty = is_high_regime * tf.nn.relu(narrow_threshold_high - width_ratio) * 2.0
                    
                    # Объединенный штраф для высоковолатильного режима
                    high_width_penalty = wide_high_penalty + narrow_high_penalty
                    high_regime_penalty_loss = 0.1 * high_width_penalty
                
                # Суммируем
                width_loss = base_width_loss + high_regime_penalty_loss + ci_regularization_loss
                loss = loss + width_loss
                
                # ✅ ПУНКТ 7: L2-регуляризация для max_width_factors_logits
                # Предотвращает переобучение параметров ширины ДИ (Train width_ratio ~3.4x vs Val ~2.4x)
                l2_width_reg = tf.constant(0.0, tf.float32)
                if hasattr(self, "max_width_factors_logits") and self.max_width_factors_logits is not None:
                    l2_width_reg = 1e-4 * tf.reduce_sum(tf.square(self.max_width_factors_logits))
                    loss = loss + l2_width_reg
                # ✅ МОНИТОРИНГ L2 регуляризации
                should_log_l2 = tf.equal(tf.math.floormod(self._step_counter, 100), 0)
                def _log_l2():
                    tf.print("  L2 width reg:", l2_width_reg)
                    return tf.constant(0)
                tf.cond(should_log_l2, _log_l2, lambda: tf.constant(0))
                
                # 10) TRAINABLE VARS (LSTM-only + diff_ukf + regime_selector + max_width_factor_logit)
                trainable_vars = []
                if self.model is not None:
                    trainable_vars.extend(self.model.trainable_variables)
                
                if getattr(self, "use_diff_ukf", False) and hasattr(self, "diff_ukf_component") and self.diff_ukf_component is not None:
                    trainable_vars.extend(self.diff_ukf_component.trainable_variables)
                
                if getattr(self, "regime_selector", None) is not None:
                    trainable_vars.extend([self.regime_selector.regime_scales, self.regime_selector.temperature])
                    if getattr(self.regime_selector, "learnable_centers", False):
                        trainable_vars.append(self.regime_selector.center_logits)
                
                if hasattr(self, "max_width_factors_logits") and isinstance(self.max_width_factors_logits, tf.Variable):
                    trainable_vars.append(self.max_width_factors_logits)
                
                if hasattr(self, "target_cov_logits") and isinstance(self.target_cov_logits, tf.Variable):
                    trainable_vars.append(self.target_cov_logits)
            
            # 11) GRADS + OPTIMIZER STEP (вне tape scope)
            use_lso = isinstance(self._optimizer, tf.keras.mixed_precision.LossScaleOptimizer)
            
            if use_lso:
                scaled_total_loss = self._optimizer.get_scaled_loss(loss)
                scaled_grads = tape.gradient(scaled_total_loss, trainable_vars)
                gradients = self._optimizer.get_unscaled_gradients(scaled_grads)
            else:
                gradients = tape.gradient(loss, trainable_vars)
            
            # optional: мониторинг None grads (редко)
            do_monitor = tf.logical_and(
                tf.cast(self.debug_mode, tf.bool),
                tf.equal(tf.math.floormod(self._step_counter, 20), 0)
            )
            if do_monitor:
                none_total = tf.reduce_sum(tf.cast([g is None for g in gradients], tf.int32))
                tf.print("[GRAD DEBUG] step", self._step_counter, "| None total:", none_total, "/", len(trainable_vars))
            
            # === ДИАГНОСТИКА ГРАДИЕНТОВ: max_width_factors_logits (по режимам) ===
            if tf.equal(tf.math.floormod(self._step_counter, 50), 0):
                try:
                    var_name = 'max_width_factors_logits'
                    target_var = None
                    grad = None
                    var_idx = -1
                    
                    for i, var in enumerate(trainable_vars):
                        if 'max_width_factors_logits' in var.name:
                            target_var = var
                            var_idx = i
                            break
                    
                    if target_var is None:
                        tf.print(
                            "[GRAD DEBUG]", "step=", self._step_counter,
                            "|", var_name + ": NOT FOUND in trainable_vars ❌"
                        )
                    else:
                        grad = gradients[var_idx]
                        if grad is None:
                            tf.print(
                                "[GRAD DEBUG]", "step=", self._step_counter,
                                "|", var_name + ": GRAD IS None ❌"
                            )
                        else:
                            # Логируем значения по режимам
                            factors_raw = target_var
                            factors_actual = tf.nn.softplus(factors_raw) + 1.0  # [3]
                            
                            tf.print(
                                "[GRAD DEBUG]", "step=", self._step_counter,
                                "|", var_name + ":",
                                "raw_logits (L/M/H):", tf.round(factors_raw * 100) / 100,
                                "| actual_factors (L/M/H):", tf.round(factors_actual * 100) / 100,
                                "| grad_mean:", tf.reduce_mean(grad),
                                "| grad_abs_max:", tf.reduce_max(tf.abs(grad)),
                                "| grad_norm:", tf.norm(grad)
                            )
                except Exception as e:
                    tf.print(
                        "[GRAD DEBUG]", "step=", self._step_counter,
                        "| ERROR:", str(e)
                    )
            
            # ✅ Gradient scaling для regime_scales (усилить градиенты на 50%)
            scaled_grads = []
            for i, grad in enumerate(gradients):
                if grad is not None:
                    var_name = trainable_vars[i].name
                    # 🔑 Усиленные градиенты для калибровочных параметров
                    if 'regime_scales' in var_name:
                        scaled_grads.append(grad * 3.0)  # ✅ ↑1.5→3.0
                    elif 'temperature' in var_name:
                        # ✅ ИСПРАВЛЕНО: фиксированное масштабирование (без Python-if)
                        # Адаптивное масштабирование удалено — не работает внутри @tf.function
                        scaled_grads.append(grad * 1.5)  # ← ✅ Базовое масштабирование
                    elif 'forecast_bias_correction' in var_name:
                        scaled_grads.append(grad * 2.0)  # ✅ Новый: bias correction быстрее
                    else:
                        scaled_grads.append(grad)
                else:
                    scaled_grads.append(grad)
            
            clipped_grads, global_norm = tf.clip_by_global_norm(scaled_grads, 1.0)
            self._optimizer.apply_gradients(zip(clipped_grads, trainable_vars))
            
            # Гарантировать диапазон температуры
            self.regime_selector.temperature.assign(
                tf.clip_by_value(self.regime_selector.temperature, 0.3, 10.0)  # ✅ Согласовано с __init__
            )
            
            # 🔑 ГИБРИДНЫЙ ПОДХОД v6: ЯВНЫЙ CLIP ТЕМПЕРАТУРЫ (критично!)
            if hasattr(self, "regime_selector") and self.regime_selector is not None:
                # 🔑 НОВОЕ: Логирование для отладки
                if tf.equal(tf.math.floormod(self._step_counter, 50), 0):
                    tf.print(
                        "[TEMP DEBUG] step=", self._step_counter,
                        "| temperature=", self.regime_selector.temperature,  # ← ✅ Без .numpy()
                        "| regime_scales=", self.regime_selector.regime_scales  # ← ✅ Без .numpy()
                    )
            
            # step counter increment
            self._step_counter.assign_add(1)
            
            # 12) METRICS
            # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: ЯВНАЯ СКАЛЯРИЗАЦИЯ ВСЕХ МЕТРИК
            mse_loss = tf.reduce_mean(tf.square(forecast - y_target_batch))  # ✅ Уже скаляр
            avg_volatility = tf.reduce_mean(final_volatility)  # ✅ Уже скаляр
            
            regime_soft_weights = tf.reduce_mean(regime_info["soft_weights"], axis=0)  # [3]
            regime_entropy = tf.reduce_mean(regime_info["entropy"])  # ✅ Скаляр
            
            q_current = tf.reduce_mean(q_base_final)
            r_current = tf.reduce_mean(tf.gather(ukf_params["r_base"], t_last, axis=1))
            qr_ratio = q_current / (r_current + 1e-8)
            
            avg_inflation = tf.reduce_mean(inflation_factors[:, -1, :])
            
            dynamic_threshold, inflation_anomaly_ratio = compute_adaptive_threshold(
                inflation_factors,
                final_volatility,
                self.threshold_ema,
                target_anomaly_ratio=0.35
            )
            
            spectrum_info = self.diff_ukf_component.get_spectrum_info()
            min_eigenvalue = spectrum_info["min_eigenvalue"]
            
            # ✅ ЯВНАЯ СКАЛЯРИЗАЦИЯ ЧЕРЕЗ tf.reduce_mean
            metrics = {
                "total_loss": tf.reduce_mean(loss),  # ✅ Принудительный редьюс
                "mse_loss": tf.reduce_mean(mse_loss),  # ✅ Дублируем для безопасности
                "entropy_loss": tf.reduce_mean(entropy_loss) if entropy_loss.shape.rank > 0 else entropy_loss,
                "avg_volatility": tf.reduce_mean(avg_volatility),
                "avg_inflation": tf.reduce_mean(avg_inflation),
                "global_norm": tf.reduce_mean(global_norm) if hasattr(global_norm, 'shape') and global_norm.shape.rank > 0 else global_norm,
                "qr_ratio": tf.reduce_mean(qr_ratio),
                "q_value": tf.reduce_mean(q_current),
                "r_value": tf.reduce_mean(r_current),
                "regime_low_weight": regime_soft_weights[0],
                "regime_mid_weight": regime_soft_weights[1],
                "regime_high_weight": regime_soft_weights[2],
                "regime_entropy": tf.reduce_mean(regime_entropy),
                "inflation_anomaly_ratio": tf.reduce_mean(inflation_anomaly_ratio),
                "ukf_min_eigenvalue": tf.reduce_mean(min_eigenvalue),
                "coverage_ratio": tf.reduce_mean(actual_coverage),  # ✅ Скаляризация
                "train_val_coverage_gap": tf.reduce_mean(train_val_gap),  # ✅ НОВАЯ
                "gap_penalty": tf.reduce_mean(gap_penalty),  # ✅ НОВАЯ
                "target_coverage": tf.reduce_mean(target_coverage_mean),  # ✅ Скаляризация
                "ci_width_vs_stddev": tf.reduce_mean(width_ratio),  # ✅ Скаляризация
                "calibration_error": tf.reduce_mean(tf.abs(actual_coverage - target_coverage_mean)),
                "calib_weight": tf.reduce_mean(current_weight),
                "calib_clipped": tf.reduce_mean(calibration_loss_clipped),  # ✅ Скаляризация
                "calib_raw": tf.reduce_mean(raw_calibration_loss),  # ✅ Скаляризация
                "calib_norm": tf.reduce_mean(calibration_loss_normalized),  # ✅ Скаляризация
                "target_width_ratio": tf.reduce_mean(target_width_ratio),
                "width_error": tf.reduce_mean(width_error),  # ✅ Скаляризация
                "width_loss": tf.reduce_mean(width_loss),  # ✅ Скаляризация
                "lambda_width": tf.reduce_mean(lambda_width),
                "width_loss_base": tf.reduce_mean(base_width_loss),  # ✅ Скаляризация
                "width_loss_high_regime": tf.reduce_mean(high_regime_penalty_loss),  # ✅ Скаляризация
                "width_loss_ci_regul": tf.reduce_mean(ci_regularization_loss),  # ✅ Скаляризация
            }
            
            # === ОБНОВЛЕНИЕ ПРЕДЫДУЩЕГО ПОКРЫТИЯ с экспоненциальным сглаживанием ===
            alpha = 0.1  # коэффициент сглаживания
            prev_cov = self._prev_train_coverage  # ← ✅ Без .read_value()
            new_cov = alpha * actual_coverage + (1.0 - alpha) * prev_cov
            self._prev_train_coverage.assign(new_cov)
            
            entropy_stats = self.entropy_regularizer.get_entropy_stats(h_lstm2)
            self.regime_selector.update_history(final_volatility)
            
            if tf.equal(tf.math.floormod(self._step_counter, 50), 0):
                centers_val = self.regime_selector.get_centers()
                soft_mean = tf.reduce_mean(soft_weights, axis=0)
                temperature_val = self.regime_selector.temperature
                scales_val = self.regime_selector.regime_scales
                
                tf.print("[REGIME DEBUG]", self._step_counter)
                tf.print("  Centers:", centers_val)
                tf.print("  Soft weights (mean):", soft_mean)
                tf.print("  Temperature:", temperature_val)
                tf.print("  Regime scales:", scales_val)
                tf.print("  Regime CE Loss:", regime_ce_loss)
                tf.print("  Separation Loss:", separation_loss)
                
                # === ДОБАВЛЕНО: диагностика max_width_factors_logits ===
                if hasattr(self, "max_width_factors_logits"):
                    factors = tf.nn.softplus(self.max_width_factors_logits) + 1.0
                    tf.print("  Max width factor per regime (LOW/MID/HIGH):", factors)
                
                # === ДОБАВЛЕНО: обучаемые целевые покрытия по режимам ===
                if hasattr(self, "target_cov_logits") and self.target_cov_logits is not None:
                    target_cov_vals = 0.7 + 0.25 * tf.sigmoid(self.target_cov_logits)  # [3]
                    tf.print(
                        "  Target coverage (LOW/MID/HIGH):",
                        tf.round(target_cov_vals * 1000) / 1000
                    )
                    tf.print(
                        "  Raw logits (L/M/H):",
                        tf.round(self.target_cov_logits * 1000) / 1000
                    )
                else:
                    tf.print("  Target coverage mode: FIXED")
            
            return (
                loss,
                metrics,
                final_state,
                final_covariance,
                forecast,
                std_dev,
                volatility_levels,
                regime_info,
                final_volatility,
                entropy_stats,
                normalized_innovations,
            )

    def _explicit_predict_next_step(self, final_state, final_covariance, current_volatility,
        q_base_final, q_sensitivity_final, q_floor_final,
        inf_factor,
        relax_base_final, relax_sensitivity_final,
        alpha_base_final, alpha_sensitivity_final,
        kappa_base_final, kappa_sensitivity_final,
        regime_assignment=None):
        """Предсказание следующего шага с симметричным применением инфляции"""

        batch_size = tf.shape(final_state)[0]
        current_vol_scalar = tf.reshape(tf.squeeze(current_volatility), [batch_size])

        # Преобразование параметров в [B]
        q_base_final = tf.squeeze(q_base_final)
        q_sensitivity_final = tf.squeeze(q_sensitivity_final)
        q_floor_final = tf.squeeze(q_floor_final)
        relax_base_final = tf.squeeze(relax_base_final)
        relax_sensitivity_final = tf.squeeze(relax_sensitivity_final)
        alpha_base_final = tf.squeeze(alpha_base_final)
        alpha_sensitivity_final = tf.squeeze(alpha_sensitivity_final)
        kappa_base_final = tf.squeeze(kappa_base_final)
        kappa_sensitivity_final = tf.squeeze(kappa_sensitivity_final)

        inf_factor = tf.squeeze(inf_factor)  # [B]
        # FIX: инфляция Q не должна быть ~3 на стандартизированной шкале
        # --- FIX SHAPES for inf_factor and Q_t ---
        inf_factor = tf.squeeze(inf_factor)
        # гарантируем строго [B]
        inf_factor = tf.reshape(inf_factor, [batch_size])
        inf_factor = tf.clip_by_value(inf_factor, 0.9, 1.2)

        # Вычисление Q с инфляцией (q_val тоже строго [B])
        q_val = q_base_final * (1.0 + q_sensitivity_final * current_vol_scalar)
        q_val = tf.squeeze(q_val)
        q_val = tf.reshape(q_val, [batch_size])
        q_val = tf.maximum(q_val, q_floor_final)
        q_val = tf.maximum(q_val, 1e-8)

        # Сначала [B]*[B] -> [B], потом reshape -> [B,1,1] (без broadcasting)
        Q_unclipped = q_val * inf_factor                      # [B]
        Q_t = tf.reshape(Q_unclipped, [batch_size, 1, 1])     # [B,1,1]

        Q_max = tf.cast(self.Q_max_pred, tf.float32)
        Q_t = tf.clip_by_value(Q_t, 1e-8, Q_max)

        # DEBUG p95 без TFP: сортируем и берём элемент с индексом ceil(0.95*N)-1
        Q_unclipped_mean = tf.reduce_mean(Q_unclipped)
        q_sorted = tf.sort(Q_unclipped)                     # [B]
        n = tf.shape(q_sorted)[0]
        idx = tf.cast(tf.math.ceil(0.95 * tf.cast(n, tf.float32)) - 1.0, tf.int32)
        idx = tf.clip_by_value(idx, 0, n - 1)
        Q_unclipped_p95 = q_sorted[idx]

        # --- END FIX ---
        q_clip_hi_frac = tf.reduce_mean(tf.cast(Q_t[:, 0, 0] >= Q_max - 1e-6, tf.float32))
        q_val_mean = tf.reduce_mean(q_val)
        q_val_max = tf.reduce_max(q_val)
        Q_mean = tf.reduce_mean(Q_t[:, 0, 0])

        # --- UKF adaptive params ---
        vol_for_ukf = tf.math.tanh(current_vol_scalar)  # (0..∞) -> (0..1)

        # 1) raw (до клипа)
        relax_factor_raw = relax_base_final * (1.0 + relax_sensitivity_final * vol_for_ukf)

        # 2) реальные лимиты (подбери; стартовые значения такие)
        relax_min_clip = 0.9
        relax_max_clip = 2.5   # начни с 2.5, потом подстроишь по логам

        # 3) то, что реально идёт в predict()
        relax_factor = tf.clip_by_value(relax_factor_raw, relax_min_clip, relax_max_clip)

        # 4) статистика raw и доли клипа (именно по raw!)
        relax_raw_mean = tf.reduce_mean(relax_factor_raw)
        relax_raw_min  = tf.reduce_min(relax_factor_raw)
        relax_raw_max  = tf.reduce_max(relax_factor_raw)

        relaxcliphifrac = tf.reduce_mean(tf.cast(relax_factor_raw > relax_max_clip, tf.float32))
        relaxcliplofrac = tf.reduce_mean(tf.cast(relax_factor_raw < relax_min_clip, tf.float32))

        alpha_t      = alpha_base_final * (1.0 + alpha_sensitivity_final * vol_for_ukf)
        kappa_t      = kappa_base_final * (1.0 + kappa_sensitivity_final * vol_for_ukf)

        # PREDICT шаг
        x_pred, P_pred = self.diff_ukf_component.predict(
            final_state,      # [B, 1]
            final_covariance, # [B, 1, 1] ← ДОБАВЛЕНО: ТЕКУЩАЯ КОВАРИАЦИЯ!
            Q_t,              # [B, 1, 1]
            relax_factor=relax_factor,
            alpha_t=alpha_t,
            kappa_t=kappa_t
        )
        pmax = tf.cast(self.diff_ukf_component.max_P, tf.float32)
        p_clip_hi_frac = tf.reduce_mean(tf.cast(P_pred[:,0,0] >= pmax - 1e-6, tf.float32))

        forecast_var = P_pred[:, 0, 0]
        std_dev = tf.sqrt(tf.maximum(forecast_var, 1e-8))

        # if self.debug_mode and self._step_counter < 20:
        #     tf.print(
        #         "PRED dbg:",
        #         "P_filt_mean", tf.reduce_mean(final_covariance[:, 0, 0]),
        #         "q_val_mean", tf.reduce_mean(q_val),
        #         "inf_mean", tf.reduce_mean(inf_factor),
        #         "Q_mean", tf.reduce_mean(Q_t[:, 0, 0]),
        #         "relax_mean", relax_raw_mean,
        #         "relax_min",  relax_raw_min,
        #         "relax_max",  relax_raw_max,
        #         "P_pred_mean", tf.reduce_mean(P_pred[:, 0, 0]),
        #         "std_mean", tf.reduce_mean(std_dev),
        #         "q_clip_hi_frac", q_clip_hi_frac,
        #         "p_clip_hi_frac", p_clip_hi_frac,
        #         "relax_clip_hi_frac", relaxcliphifrac,
        #         "relax_clip_lo_frac", relaxcliplofrac,
        #         "relax_base_mean", tf.reduce_mean(relax_base_final),
        #         "relax_sens_mean", tf.reduce_mean(relax_sensitivity_final),
        #         "vol_mean", tf.reduce_mean(current_vol_scalar),
        #         "vol_for_ukf_mean", tf.reduce_mean(vol_for_ukf),
        #         "Q_unclipped_mean", Q_unclipped_mean,
        #         "Q_unclipped_p95", Q_unclipped_p95,
        #     )

        forecast_value = tf.squeeze(x_pred, axis=-1)
        
        # ✅ ПРИМЕНИТЬ коррекцию смещения по режимам
        if hasattr(self, 'forecast_bias_correction') and regime_assignment is not None:
            regime_bias = tf.gather(self.forecast_bias_correction, regime_assignment)
            forecast_value = forecast_value + regime_bias
    
        std_dev_value = tf.squeeze(std_dev)

        pred_dbg = {
            "q_val_mean": q_val_mean,
            "q_val_max": q_val_max,
            "Q_mean": Q_mean,
            "q_clip_hi_frac": q_clip_hi_frac,
            "Q_max": self.Q_max_pred,
            "Q_unclipped_mean": Q_unclipped_mean,
            "Q_unclipped_p95": Q_unclipped_p95,
        }

        return forecast_value, std_dev_value, pred_dbg

    def _create_innovation_window(self, innovations_hist, t, B, window_size=20):
        """Эффективное создание окна истории инноваций для анализа

        Args:
            innovations_hist: TensorArray с историей инноваций формы [t_current, B, 1]
            t: текущий временной шаг (скаляр)
            B: размер батча (скаляр)
            window_size: размер окна для анализа (по умолчанию 20)

        Returns:
            window: тензор формы [B, window_size] с окном истории инноваций
        """
        if t < window_size:
            # Заполняем нулями начало, если недостаточно истории
            zeros = tf.zeros([window_size - t, B, 1], dtype=tf.float32)
            if t > 0:
                # Собираем существующую историю инноваций
                recent = innovations_hist.gather(tf.range(0, t))
                window = tf.concat([zeros, recent], axis=0)
            else:
                # Для первого шага (t=0) возвращаем полностью нулевое окно
                window = zeros
        else:
            # Берем последние window_size значений из истории
            start_idx = t - window_size
            window = innovations_hist.gather(tf.range(start_idx, t))

        # Преобразуем в правильную форму [B, window_size]
        # 1. Убираем последнее измерение (axis=2) -> [window_size, B]
        # 2. Транспонируем -> [B, window_size]
        window = tf.transpose(tf.squeeze(window, axis=2), perm=[1, 0])
        return window

    def _fallback_inflation_correction(
        self, t, x_pred_prev, x_upd_curr, P_upd_curr,
        inflation_factor_prev,
        base_inflation_t, vol_sensitivity_t, decay_rate_t,
        anomaly_threshold_t, min_duration_float_t, max_duration_float_t,
        asymmetry_factor_t, memory_decay_t, inflation_limit_t,
        vol_current
    ):
        """..."""
        B = tf.shape(x_pred_prev)[0]

        # ✅ ИСПРАВКА: Убедитесь, что vol_current одномерный [B]
        if len(vol_current.shape) > 1:
            vol_current = vol_current[:, -1] if vol_current.shape[-1] != 1 else tf.squeeze(vol_current, -1)
        vol_current = tf.squeeze(vol_current)  # Гарантия [B]

        # ✅ Ошибка предсказания в сигмах
        sigma_upd = tf.sqrt(tf.maximum(P_upd_curr, 1e-8))  # [B, 1, 1]
        sigma_upd_scalar = tf.squeeze(sigma_upd, [-2, -1])  # [B]
        prediction_error = tf.abs(x_pred_prev[:, 0] - x_upd_curr[:, 0]) / (sigma_upd_scalar + 1e-8)  # [B]

        # ✅ Если ошибка > 3σ → пропущен скачок
        missed_jump_threshold = 3.0
        is_missed_jump = prediction_error > missed_jump_threshold  # [B]

        # ✅ Коррекция инфляции для пропущенных скачков
        severity = tf.clip_by_value(
            prediction_error / missed_jump_threshold, 1.0, 3.0  # [B]
        )
        correction_adaptive = tf.where(
            is_missed_jump,
            1.5 * severity,        # При пропуске: коррекция 1.5-4.5x
            tf.ones_like(severity)  # Нет пропуска: коррекция=1.0
        )  # [B]

        # ✅ Волатильность-зависимая коррекция
        vol_correction = 0.9 + 0.2 * tf.nn.sigmoid(vol_current - 1.0)  # [B]
        inflation_correction = correction_adaptive * vol_correction  # [B]
        inflation_correction = tf.reshape(inflation_correction, [B, 1])  # [B, 1]

        # ✅ Флаг пропущенного скачка для совместимости
        is_missed_jump_flag = tf.cast(is_missed_jump, tf.float32)  # [B]
        is_missed_jump_flag = tf.reshape(is_missed_jump_flag, [B, 1])  # [B, 1]

        return inflation_correction, is_missed_jump_flag, correction_adaptive

    @tf.function
    def val_step(self, X_batch, y_for_filtering_batch, y_target_batch, 
                 regime_labels_batch=None,  # ← СДЕЛАНО ОПЦИОНАЛЬНЫМ
                 initial_state=None, initial_covariance=None):
        """
        Шаг валидации для адаптивной UKF с контекстной волатильностью.
        
        ✅ ИСПРАВЛЕНО:
        - regime_labels_batch сделан опциональным
        - compute_adaptive_threshold НЕ вызывается (не модифицирует состояние)
        - Все tf.cond возвращают одинаковые типы
        - Shape consistency для всех тензоров
        - Нет модификации состояния модели во время валидации
        """
        B = tf.shape(X_batch)[0]
        
        with tf.device(self.device):
            # 1) LSTM forward pass (training=False для валидации)
            lstm_outputs = self.model(X_batch, training=False)
            params_output = lstm_outputs['params']  # [B, T, 37]
            h_lstm2 = lstm_outputs['h_lstm2']       # [B, T, 128]
    
            # 2) Энтропийная регуляризация
            entropy_loss = self.entropy_regularizer.compute_entropy_loss(h_lstm2)
    
            # 3) Обработка выходов LSTM
            vol_context, ukf_params, inflation_config, student_t_config = self.process_lstm_output(params_output)
    
            # 4) Адаптивная UKF фильтрация
            # ✅ Инициализация состояния если не передано
            if initial_state is None:
                initial_state = tf.tile(
                    tf.reshape(self._last_state, [1, self.state_dim]), 
                    [B, 1]
                )
            if initial_covariance is None:
                initial_covariance = tf.tile(
                    tf.reshape(self._last_P, [1, self.state_dim, self.state_dim]), 
                    [B, 1, 1]
                )
            
            results = self.adaptive_ukf_filter(
                X_batch,
                y_for_filtering_batch,
                vol_context,
                ukf_params,
                inflation_config,
                student_t_config,
                initial_state,
                initial_covariance
            )
    
            # Распаковка результатов
            x_filtered = results[0]                # [B, T, 1]
            innovations = results[1]               # [B, T, 1]
            volatility_levels = results[2]         # [B, T, 1]
            inflation_factors = results[3]         # [B, T, 1]
            final_state = results[4]               # [B, 1]
            final_covariance = results[5]          # [B, 1, 1]
            correction_adaptive_hist = results[6]  # [B, T, 1]
    
            # Финальная волатильность
            final_volatility = tf.reshape(volatility_levels[:, -1, :], [-1])  # [B]
    
            # 5) Явный predict на следующий шаг
            t_last = tf.shape(ukf_params['q_base'])[1] - 1
    
            q_base_final = tf.gather(ukf_params['q_base'], t_last, axis=1)                 # [B, 1]
            q_sensitivity_final = tf.gather(ukf_params['q_sensitivity'], t_last, axis=1)   # [B, 1]
            q_floor_final = tf.gather(ukf_params['q_floor'], t_last, axis=1)               # [B, 1]
            r_base_final = tf.gather(ukf_params['r_base'], t_last, axis=1)                 # [B, 1]
            r_sensitivity_final = tf.gather(ukf_params['r_sensitivity'], t_last, axis=1)   # [B, 1]
            r_floor_final = tf.gather(ukf_params['r_floor'], t_last, axis=1)               # [B, 1]
            relax_base_final = tf.gather(ukf_params['relax_base'], t_last, axis=1)         # [B, 1]
            relax_sensitivity_final = tf.gather(ukf_params['relax_sensitivity'], t_last, axis=1)  # [B, 1]
            alpha_base_final = tf.gather(ukf_params['alpha_base'], t_last, axis=1)         # [B, 1]
            alpha_sensitivity_final = tf.gather(ukf_params['alpha_sensitivity'], t_last, axis=1)  # [B, 1]
            kappa_base_final = tf.gather(ukf_params['kappa_base'], t_last, axis=1)         # [B, 1]
            kappa_sensitivity_final = tf.gather(ukf_params['kappa_sensitivity'], t_last, axis=1)  # [B, 1]
            inf_factor_final = tf.gather(inflation_factors, t_last, axis=1)                # [B, 1]
            correction_adaptive = correction_adaptive_hist[:, -1, :]                       # [B, 1]
    
            # 6) Доверительные интервалы (LSTM-only): НЕ затираем student_t_config из LSTM
            student_t_config, target_coverage, regime_info = self._get_calibration_params(
                final_volatility,
                student_t_config=student_t_config,          # <-- берём конфиг из LSTM
                correction_adaptive=correction_adaptive     # <-- добавляем correction
            )
            target_coverage_mean = tf.reduce_mean(target_coverage)

            regime_assignment = regime_info.get("regime_assignment", None)
            
            forecast, std_dev, _ = self._explicit_predict_next_step(
                final_state,
                final_covariance,
                final_volatility,
                q_base_final, q_sensitivity_final, q_floor_final,
                inf_factor_final,
                relax_base_final, relax_sensitivity_final,
                alpha_base_final, alpha_sensitivity_final,
                kappa_base_final, kappa_sensitivity_final,
                regime_assignment=regime_assignment
            )
            
    
            ci_lower, ci_upper, _, width_penalty_from_ci = self._calibrate_confidence_interval(
                forecast, std_dev, final_volatility, student_t_config,
                innovations=innovations[:, -10:, :],
                regime_assignment=regime_info.get('regime_assignment', None),
                true_values=y_target_batch
            )
    
            ci_min = tf.minimum(ci_lower, ci_upper)
            ci_max = tf.maximum(ci_lower, ci_upper)
    
            # === DIAG: batch mix + per-regime calibration (AFTER ci_min/ci_max) ===
            # ✅ ПРОВЕРКА: regime_labels_batch доступен
            has_regime_labels = regime_labels_batch is not None
            
            diag_on = tf.logical_and(
                tf.cast(self.debug_mode, tf.bool),
                tf.equal(tf.math.floormod(self._step_counter, 42), 0)
            )
    
            def _masked_mean(x, mask_bool):
                mask = tf.cast(mask_bool, tf.float32)
                return tf.reduce_sum(x * mask) / (tf.reduce_sum(mask) + 1e-8)
    
            def _approx_q(x, q01):  # q01 in [0,1]
                xs = tf.sort(tf.reshape(x, [-1]))
                n = tf.shape(xs)[0]
                idx = tf.cast(tf.round(q01 * tf.cast(n - 1, tf.float32)), tf.int32)
                return xs[tf.clip_by_value(idx, 0, n - 1)]
    
            def _print_diag():
                if has_regime_labels:
                    # A) Mix батча по режимам ДАТАСЕТА
                    rl = tf.cast(tf.reshape(regime_labels_batch, [-1]), tf.int32)  # [B]
                    counts = tf.math.bincount(rl, minlength=3, maxlength=3, dtype=tf.int32)
                    counts_f = tf.cast(counts, tf.float32)
                    pcts = counts_f / (tf.reduce_sum(counts_f) + 1e-8)
    
                    # B) Размах масштаба внутри батча (per-sample std по y_for_filtering)
                    vol_window = tf.math.reduce_std(y_for_filtering_batch[:, -20:], axis=1)  # [B]
                    v_p10 = _approx_q(vol_window, 0.10)
                    v_p50 = _approx_q(vol_window, 0.50)
                    v_p90 = _approx_q(vol_window, 0.90)
    
                    # C) Coverage и widthRatio по режимам датасета
                    y = tf.reshape(y_target_batch, [-1])   # [B]
                    L = tf.reshape(ci_min, [-1])          # [B]
                    U = tf.reshape(ci_max, [-1])          # [B]
                    covered = tf.cast((y >= L) & (y <= U), tf.float32)  # [B]
    
                    width = tf.maximum(U - L, 0.0)        # [B]
                    width_ratio_ps = width / (vol_window + 1e-8)  # [B]
    
                    covs = []
                    wrs = []
                    for k in range(3):
                        m = (rl == k)
                        covs.append(_masked_mean(covered, m))
                        wrs.append(_masked_mean(width_ratio_ps, m))
    
                    # D) Согласованность режимов: датасет vs selector
                    pred_reg = tf.argmax(regime_info["soft_weights"], axis=1, output_type=tf.int32)  # [B]
                    cm = tf.math.confusion_matrix(rl, pred_reg, num_classes=3, dtype=tf.int32)
    
                    tf.print(
                        "\n[REGIME DIAG][VAL] step", self._step_counter,
                        "| ds_counts", counts, "| ds_pcts", pcts,
                        "| ystd(p10/p50/p90)", v_p10, v_p50, v_p90,
                        "| cov(low/mid/high)", covs,
                        "| widthRatio(low/mid/high)", wrs,
                        "| cm(ds x pred)=\n", cm
                    )
                else:
                    # Упрощённая диагностика без режимов
                    vol_window = tf.math.reduce_std(y_for_filtering_batch[:, -20:], axis=1)  # [B]
                    y = tf.reshape(y_target_batch, [-1])   # [B]
                    L = tf.reshape(ci_min, [-1])          # [B]
                    U = tf.reshape(ci_max, [-1])          # [B]
                    covered = tf.cast((y >= L) & (y <= U), tf.float32)  # [B]
                    width = tf.maximum(U - L, 0.0)        # [B]
                    width_ratio = tf.reduce_mean(width) / (tf.reduce_mean(vol_window) + 1e-8)
                    
                    tf.print(
                        "\n[REGIME DIAG][VAL] step", self._step_counter,
                        "| coverage:", tf.reduce_mean(covered),
                        "| width_ratio:", width_ratio
                    )
                return tf.constant(0, dtype=tf.int32)
    
            # ВАЖНО: tf.cond, чтобы не было Python-if по Tensor внутри @tf.function
            _ = tf.cond(diag_on, _print_diag, lambda: tf.constant(0, dtype=tf.int32))
    
            # 7) Калибровочный loss (валидация)
            mse_loss_for_normalization = tf.reduce_mean(tf.square(forecast - y_target_batch))
    
            raw_calibration_loss, actual_coverage, width_ratio, target_width_ratio, width_error = \
                self._compute_calibration_loss(
                    ci_min, ci_max, y_target_batch, y_for_filtering_batch,
                    volatility_levels, target_coverage, training=False,
                    regime_info=regime_info  # ← ДОБАВЛЕНО
                )
            
            scale = tf.stop_gradient(mse_loss_for_normalization / (raw_calibration_loss + 1e-8))
            calibration_loss_normalized = raw_calibration_loss * scale
    
            # На валидации делаем фиксированный вес (для отчёта и консистентности)
            # Используем ТОТ ЖЕ warmup как в train_step для согласованности
            warmup_steps = 400  # ← Параметр из train_step
            progress = tf.cast(self._step_counter, tf.float32) / tf.cast(warmup_steps, tf.float32)
            progress = tf.clip_by_value(progress, 0.0, 1.0)
            base_calibration_weight = 0.05  # ← Фиксированное значение для val_step
            current_weight = base_calibration_weight * (1.0 - tf.exp(-2.0 * progress))  # ✅ Медленнее
    
            calibration_loss = calibration_loss_normalized
    
            # 8) Total loss
            # ✅ calibration_loss используется для отчётности (не обнуляем)
            calibration_loss_for_val = calibration_loss
    
            loss = self.compute_loss(
                forecast,
                y_target_batch,
                volatility_levels,
                inflation_factors,
                ukf_params,
                calibration_loss_for_val,
                entropy_loss,
                regime_info=regime_info,
                training=False
            )
    
            # ✅ Width-only сигнал (не клипится calibration clip'ом)
            lambda_width_base = tf.constant(0.10, tf.float32)    # ← Как в train_step
            width_warmup_steps = tf.constant(150.0, tf.float32)  # ← Как в train_step
            wprog = tf.cast(self._step_counter, tf.float32) / width_warmup_steps
            wprog = tf.clip_by_value(wprog, 0.0, 1.0)
            lambda_width = lambda_width_base * (1.0 - tf.exp(-5.0 * wprog))
    
            width_loss = lambda_width * tf.cast(width_error, tf.float32)
    
            # ✅ Добавляем width_loss в общий loss на валидации
            loss = loss + width_loss
    
            # 9) Метрики
            mse_loss = tf.reduce_mean(tf.square(forecast - y_target_batch))
            avg_volatility = tf.reduce_mean(final_volatility)
            avg_inflation = tf.reduce_mean(inflation_factors[:, -1, :])
            forecast_std = tf.reduce_mean(std_dev)
    
            regime_soft_weights = tf.reduce_mean(regime_info['soft_weights'], axis=0)
            regime_entropy = tf.reduce_mean(regime_info['entropy'])
    
            q_val = tf.reduce_mean(q_base_final)
            r_val = tf.reduce_mean(r_base_final)
            qr_ratio_val = q_val / (r_val + 1e-8)
    
            # ✅ КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: НЕ вызывать compute_adaptive_threshold на валидации
            # Эта функция модифицирует self.threshold_ema что недопустимо при валидации
            inflation_anomaly_ratio = tf.constant(0.0, tf.float32)  # ← placeholder для val
    
            spectrum_info_val = self.diff_ukf_component.get_spectrum_info()
            min_eigenvalue_val = spectrum_info_val['min_eigenvalue']
    
            ci_width_vs_stddev = width_ratio  # единый источник истины
    
            # ✅ ЯВНАЯ СКАЛЯРИЗАЦИЯ ВСЕХ МЕТРИК
            metrics = {
                'total_loss': tf.reduce_mean(loss),
                'mse_loss': tf.reduce_mean(mse_loss),
                'entropy_loss': tf.reduce_mean(entropy_loss) if entropy_loss.shape.rank > 0 else entropy_loss,
    
                'coverage_ratio': tf.reduce_mean(actual_coverage),
                'target_coverage': tf.reduce_mean(target_coverage_mean),
                'ci_width_vs_stddev': tf.reduce_mean(ci_width_vs_stddev),
                'target_width_ratio': tf.reduce_mean(target_width_ratio),
                'calibration_error': tf.reduce_mean(tf.abs(actual_coverage - target_coverage_mean)),
    
                'avg_volatility': tf.reduce_mean(avg_volatility),
                'avg_inflation': tf.reduce_mean(avg_inflation),
                'forecast_std': tf.reduce_mean(forecast_std),
    
                'qr_ratio': tf.reduce_mean(qr_ratio_val),
                'q_value': tf.reduce_mean(q_val),
                'r_value': tf.reduce_mean(r_val),
    
                'regime_low_weight': regime_soft_weights[0],
                'regime_mid_weight': regime_soft_weights[1],
                'regime_high_weight': regime_soft_weights[2],
                'regime_entropy': tf.reduce_mean(regime_entropy),
    
                'inflation_anomaly_ratio': inflation_anomaly_ratio,  # ← placeholder для val
                'ukf_min_eigenvalue': tf.reduce_mean(min_eigenvalue_val),
    
                # --- Calibration diagnostics for epoch report ---
                'calib_weight': tf.reduce_mean(current_weight),
                'calib_clipped': tf.reduce_mean(calibration_loss),
                'calib_raw': tf.reduce_mean(raw_calibration_loss),
                'calib_norm': tf.reduce_mean(calibration_loss_normalized),
                'width_error': tf.reduce_mean(width_error),
                'lambda_width': tf.reduce_mean(lambda_width),
            }
    
            # === ЛОГИРОВАНИЕ ДИАГНОСТИКИ ===
            def _log_diagnostics():
                tf.print(
                    "\n[VAL DIAGNOSTICS] Step", self._step_counter,
                    "| MSE:", mse_loss_for_normalization,
                    "| Raw calibration:", raw_calibration_loss,
                    "| Normalized calibration:", calibration_loss_normalized,
                    "| Clipped calibration:", calibration_loss,
                    "| Ratio (calib/mse):", calibration_loss / (mse_loss_for_normalization + 1e-8),
                    "| actual_coverage:", actual_coverage,
                    "| target_coverage_mean:", target_coverage_mean
                )
                return tf.constant(0, dtype=tf.int32)
    
            def _no_log():
                return tf.constant(0, dtype=tf.int32)
    
            should_log = tf.logical_and(
                tf.cast(self.debug_mode, tf.bool),
                tf.equal(tf.math.floormod(self._step_counter, 100), 0)
            )
    
            tf.cond(should_log, _log_diagnostics, _no_log)
    
        # ✅ Возвращаем все необходимые значения для fit()
        return (
            loss, 
            metrics, 
            final_state, 
            final_covariance, 
            forecast, 
            std_dev, 
            ci_min, 
            ci_max, 
            target_coverage
        )

    def get_lr_scheduler(self, epoch, totalepochs=50, warmupepochs=8, baselr=1e-4, minlr=1e-5, warmup_type='exponential', gamma=2.0):
        """Улучшенный планировщик learning rate с явным указанием устройства"""
        with tf.device(self.device):
            epoch = tf.cast(epoch, tf.float32)
            totalepochs = tf.cast(totalepochs, tf.float32)
            warmupepochs = tf.cast(warmupepochs, tf.float32)
            baselr = tf.cast(baselr, tf.float32)
            minlr = tf.cast(minlr, tf.float32)
            gamma = tf.cast(gamma, tf.float32)

            if epoch < warmupepochs:
                # Экспоненциальный warmup - плавное начало обучения
                if warmup_type == 'exponential':
                    factor = tf.pow((epoch + 1) / warmupepochs, gamma)
                    return minlr + (baselr - minlr) * factor
                # Линейный warmup
                elif warmup_type == 'linear':
                    return baselr * (epoch + 1) / warmupepochs
                # Cosine warmup
                else:  # 'cosine'
                    progress = (epoch + 1) / warmupepochs
                    return minlr + 0.5 * (baselr - minlr) * (1.0 - math.cos(math.pi * progress))
            else:
                # Cosine decay с плавным переходом после warmup
                progress = tf.minimum(1.0, (epoch - warmupepochs) / (totalepochs - warmupepochs))
                return minlr + 0.5 * (baselr - minlr) * (1.0 + math.cos(math.pi * progress))

    def prepare_features(self, df: pd.DataFrame, mode='batch', include_ground_truth=False) -> pd.DataFrame:
        """
        Подготовка признаков с фокусом на многошкальные оценки волатильности
        для поддержки контекстной адаптации UKF параметров.

        Args:
            df: DataFrame с OHLC данными
            mode: 'batch' для обучения/валидации, 'online' для онлайн-прогнозирования
            include_ground_truth: включать ли ground truth значение для следующего шага

        Returns:
            DataFrame с вычисленными признаками
        """
        # Проверка минимального количества данных
        if len(df) < self.min_history_for_features:
            raise ValueError(
                f"Недостаточно данных для расчета признаков. Требуется минимум {self.min_history_for_features} точек")

        # Проверка обязательных колонок
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")

        # Расчет St_comp через EMD декомпозицию
        st_comp, _ = get_emd_components(df['Close'].values)

        # Создание DataFrame с признаками
        features_df = pd.DataFrame(index=df.index)
        features_df['St_comp'] = st_comp

        # Динамические окна для вычисления волатильности
        span_short = self.vol_window_short  # 36 по умолчанию
        span_medium = self.vol_window_long // 2  # ~75
        span_long = self.vol_window_long  # 150 по умолчанию
        perc_window = self.rolling_window_percentile  # 100 по умолчанию

        # 1. БАЗОВЫЕ ПРИЗНАКИ
        features_df['level'] = features_df['St_comp']

        rolling_std = features_df['St_comp'].rolling(window=20, min_periods=5).std()
        features_df['st_comp_diff'] = np.sinh(features_df['St_comp'].diff() / (rolling_std + 1e-8))

        # 2. МНОГОШКАЛЬНЫЕ ОЦЕНКИ ВОЛАТИЛЬНОСТИ
        # Короткосрочная волатильность (EWMA)
        features_df['vol_short'] = features_df['St_comp'].ewm(
            span=span_short,
            min_periods=max(3, span_short // 5),
            adjust=False
        ).std()
        features_df['log_vol_short'] = np.log(features_df['vol_short'].clip(lower=1e-8) + 1e-8)

        # Среднесрочная волатильность
        features_df['vol_medium'] = features_df['St_comp'].ewm(
            span=span_medium,
            min_periods=max(5, span_medium // 5),
            adjust=False
        ).std().clip(lower=1e-8)
        features_df['rel_vol_short_medium'] = features_df['vol_short'] / features_df['vol_medium']

        # Долгосрочная волатильность
        features_df['vol_long'] = features_df['St_comp'].ewm(
            span=span_long,
            min_periods=max(10, span_long // 5),
            adjust=False
        ).std().clip(lower=1e-8)
        features_df['rel_vol_short_long'] = features_df['vol_short'] / features_df['vol_long']

        # 3. ПРОИЗВОДНЫЕ ПО ВОЛАТИЛЬНОСТИ
        features_df['vol_accel_short'] = features_df['vol_short'].pct_change().fillna(0)
        features_df['vol_accel_rel'] = features_df['rel_vol_short_long'].pct_change().fillna(0)
        features_df['norm_vol_accel'] = features_df['vol_accel_rel'] / features_df['vol_long']

        # 4. АБСОЛЮТНО БЕЗОПАСНЫЙ РАСЧЕТ ДОХОДНОСТЕЙ
        price_change = features_df['St_comp'] - features_df['St_comp'].shift(1)
        prev_price = features_df['St_comp'].shift(1)

        # Защита от деления на ноль и отрицательных цен
        valid_prev_price = prev_price.clip(lower=1e-8)
        ratio = price_change / valid_prev_price

        # Защита от недопустимых значений для логарифма
        ratio_safe = np.clip(ratio, -0.99, 10.0)  # Ограничиваем реалистичными значениями
        returns = np.log1p(ratio_safe)  # log(1+x) более стабилен для малых x

        # Явная обработка оставшихся некорректных значений
        returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
        returns = np.clip(returns, -10.0, 10.0)
        returns_series = pd.Series(returns, index=features_df.index)  # ✅ ВАЖНО: создаем pandas Series с индексом

        # 5. ЭНТРОПИЙНЫЕ ПРИЗНАКИ
        def safe_entropy(x):
            """Безопасное вычисление энтропии с защитой от всех граничных случаев"""
            if len(x) < 5:
                return 0.0

            # Явная обработка NaN и бесконечностей
            x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

            # Если все значения одинаковые - энтропия 0
            if np.allclose(x_clean, x_clean[0], atol=1e-8):
                return 0.0

            # Нормализация
            mean_val = np.mean(x_clean)
            std_val = np.std(x_clean) + 1e-8
            x_norm = (x_clean - mean_val) / std_val

            # Защита от неконечных значений после нормализации
            x_norm = np.nan_to_num(x_norm, nan=0.0, posinf=0.0, neginf=0.0)

            try:
                # Вычисление гистограммы с защитой
                hist, _ = np.histogram(x_norm, bins=5, range=(-3, 3), density=True)
                hist = hist[hist > 0]
                if len(hist) == 0:
                    return 0.0
                entropy = -np.sum(hist * np.log(hist + 1e-8))
                return np.clip(entropy, 0.0, 10.0)
            except Exception:
                return 0.0

        # Короткосрочная энтропия
        features_df['entropy_short'] = returns_series.rolling(
            window=span_short,
            min_periods=5
        ).apply(safe_entropy, raw=True).fillna(0.0)

        # Долгосрочная энтропия
        features_df['entropy_long'] = returns_series.rolling(
            window=span_long,
            min_periods=10
        ).apply(safe_entropy, raw=True).fillna(0.0)

        # Относительная энтропия
        features_df['entropy_long_safe'] = features_df['entropy_long'].clip(lower=1e-8)
        features_df['rel_entropy'] = features_df['entropy_short'] / features_df['entropy_long_safe']

        # 6. ДОПОЛНИТЕЛЬНЫЕ СТАТИСТИЧЕСКИЕ ПРИЗНАКИ
        # Статистические характеристики доходностей
        features_df['skew'] = returns_series.rolling(
            window=span_short,
            min_periods=5
        ).skew().fillna(0)
        features_df['kurtosis'] = returns_series.rolling(
            window=span_short,
            min_periods=5
        ).kurt().fillna(0)

        # Процентильная позиция
        features_df['q5_long'] = features_df['St_comp'].rolling(
            window=perc_window,
            min_periods=10
        ).quantile(0.05).ffill().fillna(0.0)
        features_df['q95_long'] = features_df['St_comp'].rolling(
            window=perc_window,
            min_periods=10
        ).quantile(0.95).ffill().fillna(0.0)

        # Безопасное вычисление знаменателя
        denom = (features_df['q95_long'] - features_df['q5_long']).clip(lower=1e-8)
        features_df['percentile_pos'] = (features_df['St_comp'] - features_df['q5_long']) / denom

        # Безопасное преобразование Фишера
        percentile_pos = np.clip(features_df['percentile_pos'].values, 1e-6, 1 - 1e-6)
        fisher_arg = percentile_pos / (1 - percentile_pos + 1e-8)
        fisher_arg = np.clip(fisher_arg, 1e-6, 1e6)  # Защита от экстремальных значений
        features_df['percentile_pos_fisher'] = 0.5 * np.log(fisher_arg + 1e-8)

        # 7. ВОЛАТИЛЬНОСТЬ ПО РАЗЛИЧНЫМ МЕТОДИКАМ
        # Безопасные вычисления с явной обработкой ошибок
        def safe_volatility_calc(func, *args, **kwargs):
            try:
                result = func(*args, **kwargs)
                return np.nan_to_num(result.values, nan=0.0, posinf=0.0, neginf=0.0)
            except Exception:
                return np.zeros(len(features_df))

        features_df['yz'] = safe_volatility_calc(yang_zhang, df, window=span_short, trading_periods=2190, clean=False)
        features_df['gc'] = safe_volatility_calc(garman_klass, df, window=span_short, trading_periods=2190, clean=False)
        features_df['p'] = safe_volatility_calc(parkinson, df, window=span_short, trading_periods=2190, clean=False)
        features_df['rs'] = safe_volatility_calc(rogers_satchell, df, window=span_short, trading_periods=2190, clean=False)
        features_df['ht'] = safe_volatility_calc(hodges_tompkins, df, window=span_short, trading_periods=2190, clean=False)

        # 8. ПРИЗНАКИ АСИММЕТРИИ И ЭНЕРГИИ
        diff_comp = features_df['St_comp'].diff().fillna(0)

        # Энергия колебаний
        features_df['energy'] = (diff_comp ** 2).ewm(
            span=span_short,
            min_periods=5,
            adjust=False
        ).mean().clip(lower=0.0)

        # Амплитуда колебаний
        rolling_max = features_df['St_comp'].rolling(window=3, min_periods=2).max().ffill()
        rolling_min = features_df['St_comp'].rolling(window=3, min_periods=2).min().bfill()
        features_df['amplitude'] = ((rolling_max - rolling_min) / 2).fillna(0).clip(lower=0.0)

        # Импульс экстремальных позиций
        pos_diff = diff_comp.where(diff_comp > 0, 0)
        features_df['extreme_pos_momentum'] = pos_diff.rolling(
            window=span_short,
            min_periods=5
        ).mean().fillna(0)

        # Отношение асимметрий
        up_movements = diff_comp.where(diff_comp > 0, 0)
        down_movements = diff_comp.where(diff_comp < 0, 0).abs()

        up_std = up_movements.rolling(window=span_short, min_periods=10).std().fillna(0)
        down_std = down_movements.rolling(window=span_short, min_periods=10).std().fillna(0)

        # Безопасное вычисление отношения
        denominator = down_std.clip(lower=1e-8)
        features_df['asymmetry_ratio'] = (up_std + 1e-8) / denominator

        # Вес хвостовых событий
        abs_diff = diff_comp.abs()
        tail90 = abs_diff.rolling(window=span_short, min_periods=5).quantile(0.9).fillna(0)
        tail50 = abs_diff.rolling(window=span_short, min_periods=5).quantile(0.5).fillna(0)
        tail50_safe = tail50.clip(lower=1e-8)
        features_df['tail_weight_indicator'] = (tail90 + 1e-8) / tail50_safe

        # 9. СКОРОСТЬ И УСКОРЕНИЕ
        features_df['velocity'] = diff_comp
        features_df['acceleration'] = features_df['velocity'].diff().fillna(0)

        # 10. ЧИСЛОВАЯ СТАБИЛЬНОСТЬ
        features_df = features_df.replace([np.inf, -np.inf], np.nan)

        # 11. ФИНАЛЬНАЯ ОБРАБОТКА
        # Удаление вспомогательных колонок
        cols_to_drop = ['q5_long', 'q95_long', 'St_comp', 'entropy_long_safe']
        for col in cols_to_drop:
            if col in features_df.columns:
                features_df = features_df.drop(col, axis=1)

        # Финальная очистка
        features_df = features_df.ffill().bfill()
        features_df = features_df.fillna(0.0)

        # Защита от экстремальных значений
        for col in features_df.columns:
            if 'entropy' in col or 'vol' in col or 'ratio' in col:
                features_df[col] = np.clip(features_df[col], -10.0, 10.0)
            elif 'fisher' in col:
                features_df[col] = np.clip(features_df[col], -5.0, 5.0)

        # Проверка на оставшиеся некорректные значения
        features_df = features_df.replace([np.inf, -np.inf], np.nan)
        features_df = features_df.fillna(0.0)

        # Возврат результатов в зависимости от режима
        if mode == 'online':
            if include_ground_truth:
                recent_data = features_df.tail(self.seq_len + 1)
                ground_truth_value = recent_data.iloc[-1]['level']
                features_for_batch = recent_data.head(self.seq_len).copy()
                features_for_batch['ground_truth_level'] = ground_truth_value
                return features_for_batch.astype(np.float32)
            else:
                return features_df.tail(self.seq_len).astype(np.float32)

        return features_df.astype(np.float32)

    def _scale_features(self, features_df):
        """✅ ОПТИМАЛЬНАЯ ВЕРСИЯ: Применение скейлеров с сохранением группировки + СОГЛАСОВАННОЕ МАСШТАБИРОВАНИЕ 'level'"""
        scaled_df = pd.DataFrame(index=features_df.index)

        # ✅ ГРУППИРОВКА ИСПОЛЬЗУЕТСЯ ЗДЕСЬ ДЛЯ ПРИМЕНЕНИЯ
        # 1. Признаки для групповых скейлеров
        for group_name, features in self.scale_groups.items():
            if group_name == 'none':  # ← ПРОПУСКАЕМ группу 'none' здесь — обработаем отдельно
                continue

            if group_name in self.feature_scalers and self.feature_scalers[group_name] is not None:
                valid_features = [f for f in features if f in features_df.columns]
                if valid_features:
                    scaled_values = self.feature_scalers[group_name].transform(features_df[valid_features].values)
                    for i, col in enumerate(valid_features):
                        scaled_df[col] = scaled_values[:, i]
                    # print(f"  ✅ {group_name} скейлер применен к {len(valid_features)} признакам")

        # 2. Признаки без масштабирования ('none' группа)
        if 'none' in self.scale_groups:
            for col in self.scale_groups['none']:
                if col in features_df.columns:
                    scaled_df[col] = features_df[col].values
                    # 🔑 ЧИСЛЕННАЯ СТАБИЛИЗАЦИЯ (согласовано с обучением!)
                    if col == 'asymmetry_ratio':
                        scaled_df[col] = np.clip(scaled_df[col], 0.1, 3.0)
                    elif col == 'percentile_pos_fisher':
                        scaled_df[col] = np.clip(scaled_df[col], -5.0, 5.0)
                    # print(f"  ✅ {col}: семантика сохранена (без масштабирования)")

        # ════════════════════════════════════════════════════════════════
        # ✅ ВСТАВИТЬ СЮДА (после группы 'none', перед missing_cols)
        # ════════════════════════════════════════════════════════════════
        # ✅ Гарантируем, что 'level' в X_seq масштабируется ТОЧНО ТАК ЖЕ, как в y_filter
        if 'level' in features_df.columns and 'Y' in self.feature_scalers:
            # Принудительно применяем 'Y' скейлер ко ВСЕМ значениям 'level'
            level_values = features_df[['level']].values
            level_scaled = self.feature_scalers['Y'].transform(level_values)
            scaled_df['level'] = level_scaled.flatten()
        # ════════════════════════════════════════════════════════════════

        # 3. 🔑 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: СОГЛАСОВАННОЕ МАСШТАБИРОВАНИЕ 'level' ЧЕРЕЗ 'Y' СКЕЙЛЕР
        # При обучении 'level' в y_filter/y_target масштабируется через 'Y' (PowerTransformer)
        # Здесь применяем тот же скейлер к 'level' в X_seq для согласованности
        if 'level' in features_df.columns and 'Y' in self.feature_scalers and self.feature_scalers['Y'] is not None:
            # Сначала убедимся, что 'level' уже в scaled_df (из группового скейлера)
            if 'level' not in scaled_df.columns:
                # Если не попал в групповые скейлеры — берём исходное значение
                level_values = features_df[['level']].values
            else:
                level_values = scaled_df[['level']].values

            # 🔑 ПРИМЕНЯЕМ 'Y' СКЕЙЛЕР (PowerTransformer) — СОГЛАСОВАНО С ОБУЧЕНИЕМ!
            level_scaled = self.feature_scalers['Y'].transform(level_values)
            scaled_df['level'] = level_scaled.flatten()
            # print(f"  ✅ 'level' масштабирован через 'Y' скейлер (PowerTransformer) для согласованности с обучением")
        elif 'level' in features_df.columns:
            # Fallback: если 'Y' скейлер отсутствует — оставляем как есть (но это ошибка!)
            warnings.warn(
                "⚠️  Скейлер 'Y' отсутствует! 'level' не масштабирован согласованно с обучением. "
                "Это приведёт к несогласованности между обучением и инференсом."
            )

        # 4. Обработка пропущенных признаков
        missing_cols = [col for col in self.feature_columns if col not in scaled_df.columns]
        if missing_cols:
            print(f"  ⚠️ Найдены нераспределенные признаки: {', '.join(missing_cols)}")
            for col in missing_cols:
                if col in features_df.columns:
                    # Стратегия по умолчанию - RobustScaler
                    if 'robust' in self.feature_scalers and self.feature_scalers['robust'] is not None:
                        scaled_df[col] = self.feature_scalers['robust'].transform(features_df[[col]].values)[:, 0]
                    else:
                        scaled_df[col] = features_df[col].values

        # 5. 🔑 ФИНАЛЬНАЯ ОЧИСТКА ОТ НЕКОРРЕКТНЫХ ЗНАЧЕНИЙ (согласовано с обучением!)
        scaled_df = scaled_df.replace([np.inf, -np.inf], np.nan)
        scaled_df = scaled_df.fillna(0.0)

        return scaled_df

    def fit(self, train_features, val_features, epochs=50, min_epochs=5,
            patience=15, save_best_weights=True, early_stopping=True, log_dir=None):
        """Оптимизированный метод обучения для адаптивной UKF с контекстной волатильностью"""
        print("=" * 80)
        print("🚀 ОБУЧЕНИЕ LSTM-АДАПТИВНОЙ UKF МОДЕЛИ С КОНТЕКСТНОЙ ВОЛАТИЛЬНОСТЬЮ")
        print("=" * 80)

        # 0. Сброс отслеживания лучших весов
        self.reset_best_weights_tracking()
        print("🔄 Отслеживание лучших весов сброшено")

        # 1. ПОДГОТОВКА ДАТАСЕТОВ — УМНАЯ ЛОГИКА С АВТОМАТИЧЕСКИМ КЭШИРОВАНИЕМ
        print("\n📊 Подготовка оптимизированных датасетов...")

        # === КЛЮЧЕВАЯ ПРОВЕРКА: есть ли честно подготовленные данные? ===
        if hasattr(self, '_honest_preparation') and self._honest_preparation is not None:
            # Используем предварительно обработанные данные БЕЗ утечки будущего
            train_data = self._honest_preparation['train_data']
            val_data = self._honest_preparation['val_data']
            preparator = self._honest_preparation.get('preparator')

            if preparator is not None:
                # Создаём оптимизированные tf.data.Dataset напрямую
                train_ds, val_ds = preparator.create_tf_datasets(train_data, val_data, batch_size=64)
        else:
            raise RuntimeError(
                "❌ Устаревшая логика подготовки данных (_prepare_datasets) УДАЛЕНА из-за утечки будущего!\n"
                "Используйте:\n"
                " train_data, val_data, test_data = model.prepare_or_load_honest_datasets(full_df, './cache/my_data')\n"
                "   model.fit_from_prepared(train_data, val_data, test_data, epochs=50)\n"
            )

        log_interval = max(1, len(train_ds) // 10)
        print("✅ Оптимизированные датасеты успешно подготовлены")
        # 2. Инициализация модели на GPU
        print("\n🔧 Инициализация LSTM модели на GPU...")
        for X_batch, y_for_filtering_batch, y_target_batch, regime_labels_batch in train_ds.take(1):
            input_shape = (int(X_batch.shape[1]), int(X_batch.shape[2]))
            print(f"✅ Определена форма входа: {input_shape}")
            with tf.device('/GPU:0' if self._gpu_available else '/CPU:0'):
                self.model = self._build_model(input_shape, training=True)
            print(f"✅ LSTM модель инициализирована с формой входа: {input_shape}")
            break

        # 3. Инициализация оптимизатора
        print("\n✅ Инициализация оптимизатора с Loss Scale...")
        base_lr = 5e-4
        base_optimizer = tf.keras.optimizers.Adam(
                learning_rate=base_lr,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                amsgrad=False
            )
        current_policy = tf.keras.mixed_precision.global_policy()
        if current_policy.name == 'mixed_float16':
            self._optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
            print("✅ Loss Scale Optimizer инициализирован для mixed precision")
        else:
            self._optimizer = base_optimizer
            print("✅ Стандартный оптимизатор инициализирован (без mixed precision)")

        # 4. История обучения
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
            'best_epoch': 0,
            'best_val_loss': float('inf'),
            'patience_counter': 0
        }

        # 5. Основной цикл обучения
        print("\n🔥 НАЧАЛО ОБУЧЕНИЯ")
        print("=" * 80)

        try:
            for epoch in range(epochs):
                epoch_start_time = datetime.datetime.now()

                # Обновление learning rate
                current_lr = self.get_lr_scheduler(
                    epoch=epoch,
                    baselr=5e-4,
                    minlr=1e-5,
                    warmupepochs=5,  # ← ↑ 3→5 (более плавный warmup LR)
                    warmup_type='exponential',
                    gamma=1.5
                )
                self._optimizer.learning_rate.assign(current_lr)
                print(f"\n{'─' * 80}")
                print(f"📅 EPOCH {epoch + 1}/{epochs} | LR: {current_lr:.2e} | Начало: {epoch_start_time.strftime('%H:%M:%S')}")

                # ===== ОБУЧЕНИЕ =====
                print("\n📈 ОБУЧЕНИЕ...")
                epoch_losses = []
                epoch_mse_losses = []
                epoch_coverage_ratios = []
                epoch_volatility_levels = []
                train_metrics = []  # Список для хранения всех метрик по шагам обучения

                # ДОБАВЛЕНО: сбор всех нормализованных инноваций за эпоху
                all_normalized_innov = []

                for batch_idx, (X_batch, y_for_filtering_batch, y_target_batch, regime_labels_batch) in enumerate(train_ds):
                    batch_size = tf.shape(X_batch)[0]
                    current_state_size = tf.shape(self._last_state)[0]

                    # === ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ UKF С АДАПТИВНОЙ ДИСПЕРСИЕЙ ===
                    if batch_idx == 0 and epoch == 0:
                        # Адаптивная инициализация с учетом волатильности данных
                        window_std = tf.math.reduce_std(y_for_filtering_batch[:, :20], axis=1)
                        initial_variance = tf.maximum(window_std ** 2, 0.01)  # Минимум 0.01
                        initial_variance = tf.minimum(initial_variance, 0.1)   # Максимум 0.1

                        # ✅ ИНИЦИАЛИЗАЦИЯ КАК СКАЛЯРА [1]
                        initial_state_val = tf.reduce_mean(y_for_filtering_batch[:, 0])  # Скаляр

                        # Создаём переменные с формой [1] и [1, 1, 1]
                        self._last_state = tf.Variable(
                            tf.reshape(initial_state_val, [1]),  # ← Явно [1]
                            trainable=False,
                            name='last_state'
                        )

                        # Для ковариации: усредняем дисперсию → [1, 1, 1]
                        initial_cov_value = tf.reshape(
                            tf.reduce_mean(initial_variance),
                            [1, self.state_dim, self.state_dim]  # ← Явно [1, 1, 1]
                        )
                        self._last_P = tf.Variable(
                            initial_cov_value,
                            trainable=False,
                            name='last_P'
                        )

                        # Флаг инициализации состояния
                        self._state_initialized = tf.Variable(False, trainable=False, name='state_initialized', dtype=tf.bool)

                        # Счетчик шагов (используется в train_step)
                        self._step_counter = tf.Variable(0, trainable=False, name='step_counter', dtype=tf.int64)

                        # Время последней аномалии (если adaptive inflation использует это состояние)
                        self._last_anomaly_time = tf.Variable(-100, trainable=False, name='last_anomaly_time', dtype=tf.int64)

                        self._state_initialized.assign(False)
                        print(f"✅ Переменные состояния инициализированы: state={self._last_state.shape}, P={self._last_P.shape}")
                        print(f"   Начальная дисперсия: {tf.reduce_mean(initial_variance):.6f}")

                    # ВСЕГДА используем сохранённое состояние (оно всегда [1]), расширяем до размера батча
                    should_use_saved_state = tf.logical_and(self._state_initialized, self._step_counter > 0)

                    def use_saved_state():
                        # ✅ РАСШИРЕНИЕ СКАЛЯРА [1] → [B, 1] ЧЕРЕЗ TILE
                        return (
                            tf.tile(tf.reshape(self._last_state, [1, self.state_dim]), [batch_size, 1]),
                            tf.tile(tf.reshape(self._last_P, [1, self.state_dim, self.state_dim]), [batch_size, 1, 1])
                        )

                    def initialize_from_data():
                        # Инициализация только если состояние ещё не инициализировано
                        base_value = y_for_filtering_batch[:, 0]
                        initial_state = tf.reshape(base_value, [batch_size, self.state_dim])
                        window_std = tf.math.reduce_std(y_for_filtering_batch[:, :10], axis=1)
                        initial_variance = tf.maximum(window_std ** 2, 0.01)
                        initial_covariance = tf.reshape(initial_variance, [batch_size, self.state_dim, self.state_dim])
                        return (initial_state, initial_covariance)

                    # Используем tf.cond для графового режима
                    initial_state, initial_covariance = tf.cond(
                        should_use_saved_state,
                        use_saved_state,
                        initialize_from_data
                    )

                    # Шаг обучения
                    results = self.train_step(
                        X_batch, y_for_filtering_batch, y_target_batch, regime_labels_batch,
                        initial_state, initial_covariance
                    )
                    loss, metrics, final_state, final_covariance, forecast, std_devs, volatility_levels, \
                    regime_info, vol_final, entropy_stats, batch_normalized_innov = results
                    
                    # 🔴 ДОБАВИТЬ ЭТОТ БЛОК — ЛОГ ПО КАЖДОМУ БАТЧУ
                    def _log_batch_loss():
                        tf.print("[BATCH LOSS DEBUG] step=", self._step_counter,
                                 "| raw_loss=", loss,
                                 "| is_nan=", tf.math.is_nan(loss),
                                 "| is_inf=", tf.math.is_inf(loss))
                        return tf.constant(0)
                    
                    # Скалярное условие: есть ли хотя бы один NaN или Inf в loss?
                    has_nan_or_inf = tf.reduce_any(tf.math.is_nan(loss)) | tf.reduce_any(tf.math.is_inf(loss))
                    
                    _ = tf.cond(
                        has_nan_or_inf,
                        _log_batch_loss,
                        lambda: tf.constant(0)
                    )

                    # ДОБАВЛЕНО: сбор инноваций
                    all_normalized_innov.append(batch_normalized_innov.numpy())

                    # Преобразуем тензоры в числовые значения для логирования
                    entropy_stats_np = {
                        'entropy_mean': entropy_stats['entropy_mean'].numpy(),
                        'entropy_std': entropy_stats['entropy_std'].numpy(),
                        'entropy_min': entropy_stats['entropy_min'].numpy(),
                        'entropy_max': entropy_stats['entropy_max'].numpy(),
                    }

                    # ===== СОХРАНЕНИЕ СОСТОЯНИЯ ДЛЯ СЛЕДУЮЩЕГО ШАГА =====
                    # ✅ УСРЕДНЕНИЕ ПО БАТЧУ ДО СКАЛЯРА [1]
                    self._last_state.assign(
                        tf.reduce_mean(tf.squeeze(final_state, axis=1), keepdims=True)  # [B] → [1]
                    )
                    self._last_P.assign(
                        tf.reduce_mean(final_covariance, axis=0, keepdims=True)  # [B, 1, 1] → [1, 1, 1]
                    )
                    # ✅ ИСПРАВЛЕНО: Обновление _last_volatility с использованием vol_final, а не final_state
                    self._last_volatility.assign(tf.reduce_mean(vol_final, keepdims=True))  # [B] → [1]
                    self._state_initialized.assign(True)

                    # ДОПОЛНИТЕЛЬНО: принудительно применяем статистику истории для адаптации центров
                    if self._step_counter % 10 == 0:  # каждые 10 шагов
                        self.regime_selector.get_centers()  # вызов для обновления адаптивных центров

                    # Агрегация метрик
                    epoch_losses.append(loss)
                    epoch_mse_losses.append(metrics['mse_loss'])
                    epoch_volatility_levels.append(metrics['avg_volatility'])
                    train_metrics.append(metrics)  # Сохраняем все метрики для детального анализа

                    # Прогресс-бар
                    if batch_idx % 20 == 0:
                        progress = (batch_idx + 1) / len(train_ds) * 100
                        entropy_val = entropy_stats_np['entropy_mean']  # Используем предварительно вычисленную статистику

                        # Мягкие веса (вероятности режимов)
                        # --- Извлечение средних весов режимов ---
                        soft_weights = regime_info['soft_weights']
                        soft_weights_mean = tf.reduce_mean(soft_weights, axis=0).numpy().tolist()
                        low_val, mid_val, high_val = soft_weights_mean

                        regime_dist = f"LOW: {low_val:.1%} | MID: {mid_val:.1%} | HIGH: {high_val:.1%}"

                        # --- Конвертация тензоров в числа ДО f-строки ---
                        def safe_scalar(t):
                            return float(np.array(t).flatten()[0])

                        loss_val = safe_scalar(loss)
                        progress_val = safe_scalar(progress)

                        # --- Логирование прогресса ---
                        print(f"\r   Batch {batch_idx+1}/{len(train_ds)} | "
                              f"Progress: {progress_val:.1f}% | loss={loss_val:.6f} | "
                              f"LSTM_Ent:{entropy_val:.1f} | "
                              f"Regimes(LOW/MID/HIGH): {int(low_val*100)}%/{int(mid_val*100)}%/{int(high_val*100)}%",
                              end='', flush=True)
                print()  # Новая строка после прогресс-бара

                # Средние метрики по эпохе
                train_loss_avg = tf.reduce_mean(epoch_losses)
                train_mse_avg = tf.reduce_mean(epoch_mse_losses)
                train_volatility_avg = tf.reduce_mean(tf.stack(epoch_volatility_levels))

                # ===== ВАЛИДАЦИЯ =====
                print("\n📉 ВАЛИДАЦИЯ...")

                # ✅ КРИТИЧЕСКИ ВАЖНО: СОХРАНЯЕМ тренировочное состояние ПЕРЕД валидацией
                train_state_tf = tf.identity(self._last_state)
                train_P_tf = tf.identity(self._last_P)

                print(f"   💾 Сохранено тренировочное состояние (для восстановления после валидации)")

                val_losses = []
                val_mse_losses = []
                self.all_val_covered = []
                val_volatility_levels = []
                val_metrics = []  # Список для хранения всех метрик по шагам валидации
                for batch_idx, (X_val_batch, y_val_for_filtering_batch, y_val_target_batch, regime_labels_batch) in enumerate(val_ds):
                    B_val = tf.shape(X_val_batch)[0]

                    # === ФУНКЦИИ ДЛЯ ВЫБОРА СОСТОЯНИЯ ДЛЯ ВАЛИДАЦИИ ===
                    def initialize_val_from_data():
                        base_value = y_val_for_filtering_batch[:, 0]
                        initial_state = tf.reshape(base_value, [B_val, self.state_dim])
                        initial_variance = tf.math.reduce_variance(y_val_for_filtering_batch, axis=1) + 1e-6
                        initial_covariance = tf.reshape(initial_variance, [B_val, self.state_dim, self.state_dim])
                        initial_covariance = tf.maximum(initial_covariance, 1e-8)
                        return initial_state, initial_covariance

                    initial_state_val, initial_covariance_val = initialize_val_from_data()

                    # Шаг валидации
                    results_val = self.val_step(
                        X_val_batch, y_val_for_filtering_batch, y_val_target_batch, regime_labels_batch,
                        initial_state_val, initial_covariance_val
                    )
                    val_loss, metrics_val, final_state_val, final_covariance_val, forecast_val, std_devs_val, ci_lower_val, ci_upper_val, target_coverage_val = results_val

                    # Сохраняем прогнозы и ДИ для диагностики
                    if not hasattr(self, 'all_forecasts'):
                        self.all_forecasts = []
                        self.all_ci_lowers = []
                        self.all_ci_uppers = []
                        self.all_actuals = []
                        self.all_target_coverages = []

                    self.all_forecasts.extend(forecast_val.numpy().flatten())
                    self.all_ci_lowers.extend(ci_lower_val.numpy().flatten())
                    self.all_ci_uppers.extend(ci_upper_val.numpy().flatten())
                    self.all_actuals.extend(y_val_target_batch.numpy().flatten())
                    self.all_target_coverages.append(target_coverage_val.numpy())

                    # Агрегация метрик
                    val_losses.append(val_loss)
                    val_mse_losses.append(metrics_val['mse_loss'])

                    # Накопляем флаги покрытия для корректного усреднения по всем точкам
                    y_target_flat = tf.reshape(y_val_target_batch, [-1])
                    ci_min_flat = tf.reshape(ci_lower_val, [-1])
                    ci_max_flat = tf.reshape(ci_upper_val, [-1])
                    covered_flat = tf.cast((y_target_flat >= ci_min_flat) & (y_target_flat <= ci_max_flat), tf.float32)
                    self.all_val_covered.extend(covered_flat.numpy().flatten())

                    val_volatility_levels.append(metrics_val['avg_volatility'])
                    val_metrics.append(metrics_val)  # Сохраняем все метрики для детального анализа

                    # Прогресс-бар
                    if batch_idx % 10 == 0:
                        progress = (batch_idx + 1) / len(val_ds) * 100
                        print(f"\r   Batch {batch_idx+1}/{len(val_ds)} | "
                              f"Progress: {progress:.1f}% | val_loss={val_loss:.6f}", end='')
                print()  # Новая строка после прогресс-бара

                # ✅ КРИТИЧЕСКИ ВАЖНО: ВОССТАНАВЛИВАЕМ тренировочное состояние ПОСЛЕ валидации
                self._last_state.assign(train_state_tf)
                self._last_P.assign(train_P_tf)
                print(f"   🔁 Восстановлено тренировочное состояние (чистая валидация без утечки)")

                # Средние метрики валидации
                val_loss_avg = tf.reduce_mean(val_losses)
                val_mse_avg = tf.reduce_mean(val_mse_losses)
                val_volatility_avg = tf.reduce_mean(tf.stack(val_volatility_levels))

                # 🔑 ГИБРИДНЫЙ ПОДХОД v2: Сохраняем val coverage для domain adaptation penalty
                if hasattr(self, 'all_val_covered') and len(self.all_val_covered) > 0:
                    self._prev_val_coverage = tf.constant(
                        np.mean(self.all_val_covered), 
                        dtype=tf.float32
                    )
                else:
                    if not hasattr(self, '_prev_val_coverage'):
                        self._prev_val_coverage = tf.constant(0.85, dtype=tf.float32)  # default target
                
                # ===== ОБРАБОТКА РЕЗУЛЬТАТОВ =====
                epoch_end_time = datetime.datetime.now()
                epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()

                if all_normalized_innov:
                    all_normalized_innov = np.concatenate(all_normalized_innov, axis=0)
                    all_normalized_innov = np.abs(all_normalized_innov).flatten()  # делаем одномерным

                # Генерация и вывод детального отчета - Пока временно отключили
                epoch_report = self.generate_epoch_report(epoch, train_metrics, val_metrics, all_normalized_innov)
                print(epoch_report)

                if self.debug_mode:
                    print("\n[DEBUG EPOCH END] Parameter snapshots:")
                    print(f"  d_raw = {self.diff_ukf_component.spec_param.d_raw.numpy():.6f} "
                          f"→ P = {tf.nn.softplus(self.diff_ukf_component.spec_param.d_raw).numpy() + 0.01:.6f}")
                    print(f"  regime_scales = {self.regime_selector.regime_scales.numpy()}")

                # ===== СОХРАНЕНИЕ ЛУЧШИХ ВЕСОВ И РАННЯЯ ОСТАНОВКА =====
                current_val_loss = val_mse_avg.numpy()
                if save_best_weights and not np.isnan(current_val_loss) and not np.isinf(current_val_loss):
                    if current_val_loss < self.best_val_loss:
                        self.best_val_loss = current_val_loss
                        self.best_epoch = epoch
                        self.best_weights_dict = self.get_current_weights()
                        self.best_scalers = {
                            'feature_scalers': self.feature_scalers.copy() if hasattr(self, 'feature_scalers') and self.feature_scalers is not None else None,
                            'y_scaler': self.feature_scalers.get('Y') if hasattr(self, 'feature_scalers') and self.feature_scalers is not None else None
                        } if hasattr(self, 'feature_scalers') and self.feature_scalers is not None else None
                        self.patience_counter = 0
                        print(f"\n🌟 УЛУЧШЕНИЕ! Новые лучшие веса: {self.best_val_loss:.6f} (эпоха {self.best_epoch + 1})")
                    else:
                        self.patience_counter += 1
                else:
                    self.patience_counter += 1

                # Очищаем накопитель для следующей эпохи
                if hasattr(self, 'all_val_covered'):
                    self.all_val_covered = []

                # Проверка условий ранней остановки
                should_stop = False
                if early_stopping and epoch >= min_epochs and patience is not None:
                    if self.patience_counter >= patience:
                        print(f"\n🛑 EARLY STOPPING: нет улучшения в течение {self.patience_counter} эпох")
                        should_stop = True
                    # ✅ НОВОЕ: АДАПТИВНЫЙ Early stopping по coverage gap
                    if epoch > 15:
                        train_cov = np.mean([m['coverage_ratio'] for m in train_metrics])
                        val_cov = np.mean([m['coverage_ratio'] for m in val_metrics])
                        coverage_gap = abs(train_cov - val_cov)
                        # ✅ Адаптивный порог на основе волатильности
                        avg_volatility = np.mean([m['avg_volatility'] for m in val_metrics])
                        adaptive_gap_threshold = 0.08 + 0.10 * float(avg_volatility)  # 8-18pp
                        if coverage_gap > adaptive_gap_threshold:
                            print(f"⚠️ Early stopping: чрезмерный разрыв покрытия ({coverage_gap:.2%} > {adaptive_gap_threshold:.2%})")
                            should_stop = True

                # Сохранение истории
                history['train_loss'].append(train_loss_avg.numpy())
                history['val_loss'].append(val_loss_avg.numpy())
                history['learning_rates'].append(current_lr.numpy())
                history['patience_counter'] = self.patience_counter

                if should_stop:
                    if self.best_weights_dict is not None:
                        print(f"\n📥 Загружаем лучшие веса с эпохи {self.best_epoch + 1}")
                        self.load_best_weights()
                    break

                # Сохранение модели каждые 10 эпох
                if (epoch + 1) % 10 == 0 and save_best_weights:
                    # Формируем полный путь с использованием save_dir
                    save_path = os.path.join(self.save_dir, f"model_epoch_{epoch+1}")
                    self.save(save_path)
                    print(f"\n💾 Промежуточная модель сохранена: {os.path.abspath(save_path)}")

        except Exception as e:
            print(f"❌ КРИТИЧЕСКАЯ ОШИБКА в эпохе {epoch+1}: {str(e)}")
            import traceback
            traceback.print_exc()
            # Восстановление лучших весов
            if self.best_weights_dict is not None:
                print("🔄 Восстановление лучших весов")
                self.load_best_weights()
            raise

        # ===== ЗАВЕРШЕНИЕ ОБУЧЕНИЯ =====
        if self.best_weights_dict is not None:
            print(f"\n📥 Загружаем лучшие веса с эпохи {self.best_epoch + 1} (val_loss: {self.best_val_loss:.6f})")
            self.load_best_weights()

        print(f"\n{'=' * 80}")
        print(f"🎉 ОБУЧЕНИЕ ЗАВЕРШЕНО!")
        print(f"{'=' * 80}")
        print(f"✅ Лучшая валидационная потеря: {self.best_val_loss:.6f} (эпоха {self.best_epoch + 1})")
        print(f"   Общая длительность: {datetime.datetime.now().strftime('%H:%M:%S')}")
        print("=" * 80)

        return history

    def evaluate_coverage(self, ci_lower, ci_upper, y_actual):
        covered = (y_actual >= ci_lower) & (y_actual <= ci_upper)
        coverage = np.mean(covered)

        # ===== ДОБАВИТЬ: Диагностические проверки =====
        print(f"\n{'='*50}")
        print(f"📊 АНАЛИЗ ПОКРЫТИЯ ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ")
        print(f"{'='*50}")

        if coverage >= 0.99:
            print(f"🚨 КРИТИЧНО: Покрытие = {coverage:.1%} (слишком высокое!)")
            print(f"   → Проверьте adaptive_ukf_filter (ограничение P)")
            print(f"   → Проверьте _student_t_update (ограничение tail_weight)")
        elif coverage < 0.70:
            print(f"⚠️ ПРЕДУПРЕЖДЕНИЕ: Покрытие = {coverage:.1%} (слишком низкое!)")

        ci_width = np.mean(ci_upper - ci_lower)
        y_std = np.std(y_actual)
        width_ratio = ci_width / (y_std + 1e-8)

        status_indicator = '⚠️' if width_ratio > 3.0 or width_ratio < 0.5 else '✅'
        print(f"{status_indicator} Отношение ширины ДИ к std: {width_ratio:.2f}")
        if width_ratio > 3.0:
            print(f"   → СЛИШКОМ ШИРОКИЕ интервалы (width_ratio > 3.0)")
        elif width_ratio < 0.5:
            print(f"   → СЛИШКОМ УЗКИЕ интервалы (width_ratio < 0.5)")

        print(f"📈 Покрытие: {coverage:.2%}")
        print(f"   Количество попаданий: {np.sum(covered)} / {len(covered)}")
        status = '✅ ХОРОШО' if 0.80 <= coverage <= 0.95 else '❌ ПРОБЛЕМА'
        print(f"🎯 Качество калибровки: {status}")
        print(f"{'='*50}\n")
        # ===== КОНЕЦ =====

        return {
            'coverage': coverage,
            'ci_width': ci_width,
            'width_ratio': width_ratio,
            'is_valid': (0.80 <= coverage <= 0.95)
        }

    def generate_epoch_report(self, epoch, train_metrics, val_metrics, all_normalized_innov=None):
        """
        Компактный и информативный отчёт по эпохе.
        Ориентирован на калибровку: coverage vs target и ширина ДИ, плюс базовые метрики.
        """
        import numpy as np

        def _mean(key, metrics_list, default=np.nan):
            vals = []
            for m in metrics_list:
                if key not in m:
                    continue
                v = m[key]
                # v может быть tf.Tensor (eager) или python float
                try:
                    v = float(v.numpy())
                except Exception:
                    try:
                        v = float(v)
                    except Exception:
                        continue
                if not np.isnan(v) and not np.isinf(v):
                    vals.append(v)
            return float(np.mean(vals)) if vals else float(default)

        def _fmt_pct(x):
            return f"{100.0 * x:.2f}%"

        def _cov_status(cov, target, tol=0.03):
            gap = cov - target
            if abs(gap) <= tol:
                return "✅", "В цели", gap
            if gap < -tol:
                return "⚠️", "Низкое", gap
            return "⚠️", "Избыточное", gap

        def _width_status(width_ratio, target_width_ratio=None):
            # target_width_ratio пока может отсутствовать; тогда даём простую эвристику
            if target_width_ratio is None or np.isnan(target_width_ratio):
                if width_ratio <= 5.0:
                    return "✅", "Ок", np.nan
                return "⚠️", "Слишком широкие", np.nan

            mult = width_ratio / (target_width_ratio + 1e-8)
            if mult <= 1.25:
                return "✅", "Ок", mult
            return "⚠️", "Слишком широкие", mult

        # === БАЗОВЫЕ МЕТРИКИ ===
        tr_loss = _mean("total_loss", train_metrics)
        tr_mse = _mean("mse_loss", train_metrics)
        va_loss = _mean("total_loss", val_metrics)
        va_mse = _mean("mse_loss", val_metrics)

        # === КАЛИБРОВКА ===
        tr_cov = _mean("coverage_ratio", train_metrics)
        tr_tc = _mean("target_coverage", train_metrics)
        tr_w = _mean("ci_width_vs_stddev", train_metrics)

        va_cov = _mean("coverage_ratio", val_metrics)
        va_tc = _mean("target_coverage", val_metrics)
        va_w = _mean("ci_width_vs_stddev", val_metrics)

        # Если ты добавишь эти ключи в metrics — отчёт станет ещё полезнее.
        # (Если ключей нет — останется nan и строчки просто будут менее информативны.)
        tr_cw = _mean("calib_weight", train_metrics)
        tr_calib_clip = _mean("calib_clipped", train_metrics)
        tr_calib_raw = _mean("calib_raw", train_metrics)
        tr_calib_norm = _mean("calib_norm", train_metrics)

        # Опционально: если начнёшь возвращать target_width_ratio из _compute_calibration_loss
        # и добавишь в metrics.
        tr_tw = _mean("target_width_ratio", train_metrics)
        va_tw = _mean("target_width_ratio", val_metrics)

        # === WIDTH-ONLY (если добавишь ключи в metrics) ===
        tr_we = _mean("width_error", train_metrics)
        tr_wl = _mean("width_loss", train_metrics)
        tr_lw = _mean("lambda_width", train_metrics)

        va_we = _mean("width_error", val_metrics)
        va_wl = _mean("width_loss", val_metrics)
        va_lw = _mean("lambda_width", val_metrics)

        # Статусы
        tr_cov_s, tr_cov_msg, tr_cov_gap = _cov_status(tr_cov, tr_tc)
        va_cov_s, va_cov_msg, va_cov_gap = _cov_status(va_cov, va_tc)

        tr_w_s, tr_w_msg, tr_w_mult = _width_status(tr_w, tr_tw if not np.isnan(tr_tw) else None)
        va_w_s, va_w_msg, va_w_mult = _width_status(va_w, va_tw if not np.isnan(va_tw) else None)

        tr_overall = "✅" if (tr_cov_s == "✅" and tr_w_s == "✅") else "⚠️"
        va_overall = "✅" if (va_cov_s == "✅" and va_w_s == "✅") else "⚠️"

        # Инновации (опционально, очень коротко)
        innov_line = ""
        if all_normalized_innov is not None:
            try:
                arr = np.asarray(all_normalized_innov).astype(np.float32).reshape(-1)
                arr = arr[np.isfinite(arr)]
                if arr.size > 0:
                    innov_line = f"   Innovations | abs mean: {arr.mean():.3f} | p95: {np.quantile(arr, 0.95):.3f}\n"
            except Exception:
                innov_line = ""

        # === Сборка отчёта ===
        parts = [
            f"\n{'='*80}\n",
            f"📊 ОТЧЕТ ПО ЭПОХЕ {epoch+1}\n",
            f"{'='*80}\n",
            "📈 БАЗОВЫЕ МЕТРИКИ:\n",
            f"   TRAIN → Loss: {tr_loss:.6f} | MSE: {tr_mse:.6f}\n",
            f"   VAL   → Loss: {va_loss:.6f} | MSE: {va_mse:.6f}\n",
            "\n",
            f"📊 КАЛИБРОВКА ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ (TRAIN): {tr_overall}\n",
            f"   • Покрытие: {_fmt_pct(tr_cov)} (цель: {_fmt_pct(tr_tc)}) | gap: {100.0*tr_cov_gap:+.2f} п.п. → {tr_cov_s} {tr_cov_msg}\n",
            f"   • Ширина ДИ / масштаб данных: {tr_w:.2f}x → {tr_w_s} {tr_w_msg}",
            (f" (×{tr_w_mult:.2f} от цели)\n" if np.isfinite(tr_w_mult) else "\n"),
            "\n",
            f"📊 КАЛИБРОВКА ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ (VAL): {va_overall}\n",
            f"   • Покрытие: {_fmt_pct(va_cov)} (цель: {_fmt_pct(va_tc)}) | gap: {100.0*va_cov_gap:+.2f} п.п. → {va_cov_s} {va_cov_msg}\n",
            f"   • Ширина ДИ / масштаб данных: {va_w:.2f}x → {va_w_s} {va_w_msg}",
            (f" (×{va_w_mult:.2f} от цели)\n" if np.isfinite(va_w_mult) else "\n"),
        ]

        report = "".join(parts)

        # Доп. инфа по калибровке (если ты добавил ключи)
        # Печатаем только если есть хоть что-то не nan, чтобы не шуметь.
        if any(np.isfinite(x) for x in [tr_cw, tr_calib_clip, tr_calib_raw, tr_calib_norm]):
            report += "\n🧪 ВЕС/КЛИП КАЛИБРОВКИ (TRAIN, среднее по батчам):\n"
            if np.isfinite(tr_cw):
                report += f"   • calib_weight: {tr_cw:.6f}\n"
            if np.isfinite(tr_calib_clip):
                report += f"   • calib_clipped: {tr_calib_clip:.6e}\n"
            if np.isfinite(tr_calib_norm):
                report += f"   • calib_norm: {tr_calib_norm:.6f}\n"
            if np.isfinite(tr_calib_raw):
                report += f"   • calib_raw: {tr_calib_raw:.6f}\n"

        # Доп. инфа по width-only (TRAIN/VAL)
        if any(np.isfinite(x) for x in [tr_we, tr_wl, tr_lw, va_we, va_wl, va_lw]):
            report += "\n📏 WIDTH-ONLY (среднее по батчам):\n"

            if np.isfinite(tr_we) or np.isfinite(tr_wl) or np.isfinite(tr_lw):
                report += "   TRAIN:\n"
                if np.isfinite(tr_we):
                    report += f"     • width_error: {tr_we:.6f}\n"
                if np.isfinite(tr_wl):
                    report += f"     • width_loss: {tr_wl:.6e}\n"
                if np.isfinite(tr_lw):
                    report += f"     • lambda_width: {tr_lw:.6f}\n"

            if np.isfinite(va_we) or np.isfinite(va_wl) or np.isfinite(va_lw):
                report += "   VAL:\n"
                if np.isfinite(va_we):
                    report += f"     • width_error: {va_we:.6f}\n"
                if np.isfinite(va_wl):
                    report += f"     • width_loss: {va_wl:.6e}\n"
                if np.isfinite(va_lw):
                    report += f"     • lambda_width: {va_lw:.6f}\n"

        if innov_line:
            report += "\n🔎 НОРМАЛИЗОВАННЫЕ ИННОВАЦИИ:\n" + innov_line

        report += f"{'='*80}\n"
        return report

    def inverse_transform_target(self, scaled_values: np.ndarray) -> np.ndarray:
        """
        ✅ Восстановление целевого значения из масштабированного пространства

        Пример:
            scaled_pred = np.array([-0.15])  # Прогноз модели (масштабированный)
            original_pred = model.inverse_transform_target(scaled_pred)
            # Результат: [99.2] - восстановленное значение в исходном пространстве
        """

        if self.feature_scalers is None or 'Y' not in self.feature_scalers:
            raise ValueError("Скейлеры не загружены!")

        # Преобразуем в NumPy если нужно
        if isinstance(scaled_values, tf.Tensor):
            scaled_values = scaled_values.numpy()

        # Преобразуем в 2D для sklearn [n_samples, 1]
        if scaled_values.ndim == 1:
            scaled_values = scaled_values.reshape(-1, 1)

        # Обратное преобразование
        original_values = self.feature_scalers['Y'].inverse_transform(scaled_values)

        return original_values.flatten()

    def save(self, path: str):
        """Сохранение модели и полного состояния в *_state.pkl (без отдельных файлов selector/ukf)."""
        import os
        import pickle
        import io
        import joblib
        import datetime
        import numpy as np
        import tensorflow as tf

        if not path:
            raise ValueError("Path не может быть пустым")

        dir_path = os.path.dirname(path)
        filename = os.path.basename(path)
        if not dir_path:
            dir_path = self.save_dir if hasattr(self, 'save_dir') else './model_checkpoints'
            os.makedirs(dir_path, exist_ok=True)
            full_path = os.path.join(dir_path, filename)
        else:
            os.makedirs(dir_path, exist_ok=True)
            full_path = path

        print(f"📁 Полный путь для сохранения: {os.path.abspath(full_path)}")

        def _mean_scalar(x) -> float:
            arr = np.asarray(x)
            if arr.ndim == 0:
                val = float(arr.item())
            else:
                val = float(np.mean(arr.reshape(-1)).item())
            # Защита от некорректных значений
            if not np.isfinite(val):
                return 0.0
            return val

        # --- 1) LSTM ---
        if self.model is not None:
            lstm_keras_path = f"{full_path}_lstm.keras"
            self.model.save(lstm_keras_path)
            print(f"✅ LSTM модель сохранена: {lstm_keras_path}")
        else:
            print("⚠️ self.model is None — сохранён будет только state/metadata")

        # --- 2) metadata ---
        metadata = {
            'version': '1.0.0',
            'state_dim': int(getattr(self, 'state_dim', 1)),
            'seq_len': int(getattr(self, 'seq_len', 72)),
            'feature_columns': getattr(self, 'feature_columns', None),
            'vol_window_short': int(getattr(self, 'vol_window_short', 36)),
            'vol_window_long': int(getattr(self, 'vol_window_long', 150)),
            'rolling_window_percentile': int(getattr(self, 'rolling_window_percentile', 100)),
            'emd_window': int(getattr(self, 'emd_window', 350)),
            'min_history_for_features': int(getattr(self, 'min_history_for_features', 350)),
            'num_modes': int(getattr(self, 'num_modes', 1)),
            'use_diff_ukf': bool(getattr(self, 'use_diff_ukf', False)),
            'saved_at': str(datetime.datetime.now()),
            'architecture': 'single_state_pkl_with_selector_and_diffukf'
        }
        metadata_path = f"{full_path}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ Метаданные сохранены: {metadata_path}")

        # --- 3) canonical filter state ---
        last_state_scalar = _mean_scalar(self._last_state.numpy()) if hasattr(self, '_last_state') else 0.0

        P = np.asarray(self._last_P.numpy(), dtype=np.float32) if hasattr(self, '_last_P') else np.asarray([[[0.1]]], dtype=np.float32)
        # Всегда приводим к [[p11]] для портативности
        if P.ndim >= 2:
            p11 = float(np.mean(P[:, 0, 0]).item()) if P.ndim == 3 else float(np.mean(P.reshape(-1)).item())
        else:
            p11 = float(np.asarray(P).item())
        # Сохраняем как [[p11]]
        last_P_val = [[p11]]

        last_vol_scalar = 0.1
        if hasattr(self, '_last_volatility') and self._last_volatility is not None:
            last_vol_scalar = _mean_scalar(self._last_volatility.numpy())

        # --- 4) diff ukf state (embedded) ---
        diff_ukf_state = None
        if bool(getattr(self, 'use_diff_ukf', False)) and hasattr(self, 'diff_ukf_component'):
            try:
                diff_ukf_state = {
                    'd_raw': np.asarray(self.diff_ukf_component.spec_param.d_raw.numpy()).astype(np.float32)
                }
            except Exception:
                diff_ukf_state = None

        # --- 5) regime selector state (embedded) ---
        regime_selector_state = None
        if hasattr(self, 'regime_selector') and self.regime_selector is not None:
            rs = self.regime_selector
            regime_selector_state = {
                'regime_scales': np.asarray(rs.regime_scales.numpy(), dtype=np.float32) if hasattr(rs, 'regime_scales') else None,
                'temperature': float(np.asarray(rs.temperature.numpy()).item()) if hasattr(rs, 'temperature') else None,
                'history': (
                    np.asarray(rs._vol_history.numpy(), dtype=np.float32) if hasattr(rs, '_vol_history')
                    else (np.asarray(rs.vol_history.numpy(), dtype=np.float32) if hasattr(rs, 'vol_history') else None)
                ),
                'learnable_centers': bool(getattr(rs, 'learnable_centers', False))
            }
            if hasattr(rs, 'center_logits'):
                regime_selector_state['center_logits'] = np.asarray(rs.center_logits.numpy(), dtype=np.float32)

        # --- 6) scalers serialize ---
        def _safe_serialize_scaler(scaler):
            if scaler is None:
                return None
            try:
                buffer = io.BytesIO()
                joblib.dump(scaler, buffer)
                return {'type': 'joblib', 'data': buffer.getvalue(), 'class_name': scaler.__class__.__name__}
            except Exception:
                params = {}
                for attr in ['scale_', 'mean_', 'var_', 'n_samples_seen_', 'center_', 'scale', 'quantile_range']:
                    if hasattr(scaler, attr):
                        value = getattr(scaler, attr)
                        params[attr] = value.tolist() if isinstance(value, np.ndarray) else value
                return {'type': 'manual', 'class_name': scaler.__class__.__name__, 'params': params}

        feature_scalers_blob = None
        if hasattr(self, 'feature_scalers') and self.feature_scalers is not None:
            feature_scalers_blob = {
                'robust': _safe_serialize_scaler(self.feature_scalers.get('robust')),
                'standard': _safe_serialize_scaler(self.feature_scalers.get('standard')),
                'minmax': _safe_serialize_scaler(self.feature_scalers.get('minmax')),
                'Y': _safe_serialize_scaler(self.feature_scalers.get('Y')),
            }
            print("✅ Скейлеры признаков сохранены")

        # --- 7) best_weights_dict embedded (optional but requested) ---
        best_weights_dict_blob = None
        if hasattr(self, 'best_weights_dict') and self.best_weights_dict is not None:
            # ВАЖНО: best_weights_dict уже numpy/list/float-ориентирован (из get_current_weights),
            # поэтому его можно pickle'ить напрямую.
            best_weights_dict_blob = self.best_weights_dict

        # --- 8) state.pkl ---
        model_state = {
            # canonical online/filter state
            '_last_state': float(last_state_scalar),   # float
            '_last_P': [[float(p11)]],                 # portable [1,1]
            '_last_volatility': float(last_vol_scalar),

            '_state_initialized': bool(self._state_initialized.numpy()) if hasattr(self, '_state_initialized') else False,
            '_step_counter': int(self._step_counter.numpy()) if hasattr(self, '_step_counter') else 0,
            '_last_anomaly_time': int(self._last_anomaly_time.numpy()) if hasattr(self, '_last_anomaly_time') else -100,

            # anomaly buffer
            'anomaly_buffer': np.asarray(self.anomaly_buffer.value().numpy(), dtype=np.float32) if hasattr(self, 'anomaly_buffer') else None,
            'buffer_index': int(self.buffer_index.numpy()) if hasattr(self, 'buffer_index') else 0,
            'anomaly_buffer_size': int(getattr(self, 'anomaly_buffer_size', 100)),

            # scalar params (оставляем как часть архитектуры)
            'max_width_factors_logits': self.max_width_factors_logits.numpy() if hasattr(self, 'max_width_factors_logits') else np.array([np.log(1.5)] * 3),
            'lambda_entropy': self.lambda_entropy.numpy() if hasattr(self, 'lambda_entropy') else 0.02,
            'threshold_ema': self.threshold_ema.numpy() if hasattr(self, 'threshold_ema') else 3.0,

            # embedded components
            'diff_ukf_state': diff_ukf_state,
            'regime_selector_state': regime_selector_state,

            # scalers/groups
            'feature_scalers': feature_scalers_blob,
            'scale_groups': self.scale_groups if hasattr(self, 'scale_groups') else None,
            'best_scalers': self.best_scalers if hasattr(self, 'best_scalers') else None,

            # tracking
            'best_val_loss': float(getattr(self, 'best_val_loss', float('inf'))),
            'best_epoch': int(getattr(self, 'best_epoch', 0)),
            'patience_counter': int(getattr(self, 'patience_counter', 0)),

            # requested: embed best snapshot too
            'best_weights_dict': best_weights_dict_blob,

            # flags
            'use_diff_ukf': bool(getattr(self, 'use_diff_ukf', False)),
            'num_modes': int(getattr(self, 'num_modes', 1)),
        }

        state_path = f"{full_path}_state.pkl"
        with open(state_path, 'wb') as f:
            pickle.dump(model_state, f)
        print(f"✅ Состояние модели сохранено: {state_path}")

        print("\n" + "=" * 60)
        print(f"✅ МОДЕЛЬ ПОЛНОСТЬЮ СОХРАНЕНА: {full_path}")
        print(f"   Версия: {metadata['version']}")
        print(f"   Лучшая эпоха: {model_state['best_epoch']}, Val Loss: {model_state['best_val_loss']:.6f}")
        print("=" * 60)

    def load(self, path: str):
        """Загрузка модели и полного состояния из *_state.pkl (встроены selector и diff_ukf)."""
        import os
        import pickle
        import joblib
        import io
        import numpy as np
        import tensorflow as tf
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer

        print("\n" + "=" * 60)
        print("📥 ЗАГРУЗКА МОДЕЛИ")
        print("=" * 60)

        def _safe_deserialize_scaler(data):
            if data is None:
                return None
            try:
                if data.get('type') == 'joblib':
                    return joblib.load(io.BytesIO(data['data']))
                if data.get('type') == 'manual':
                    cls_name = data.get('class_name', 'StandardScaler')
                    cls_map = {
                        'RobustScaler': RobustScaler,
                        'StandardScaler': StandardScaler,
                        'MinMaxScaler': MinMaxScaler,
                        'PowerTransformer': PowerTransformer
                    }
                    scaler = cls_map.get(cls_name, StandardScaler)()
                    for k, v in data.get('params', {}).items():
                        if hasattr(scaler, k):
                            setattr(scaler, k, np.array(v) if isinstance(v, list) else v)
                    return scaler
            except Exception as e:
                print(f"   ⚠️ Не удалось десериализовать скейлер: {str(e)}")
                return None
            return None

        # 0) optimizer policy: only self._optimizer exists, do not init here
        if hasattr(self, '_optimizer'):
            self._optimizer = None

        # === 1) metadata (required) ===
        metadata_path = f"{path}_metadata.pkl"
        if not os.path.exists(metadata_path):
            raise RuntimeError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: файл метаданных не найден: {metadata_path}")

        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        print(f"✅ Метаданные загружены: версия {metadata.get('version', 'N/A')}")

        # === 2) restore base params ===
        for param_name, default_value in [
            ('seq_len', 72), ('state_dim', 1), ('num_modes', 1),
            ('vol_window_short', 36), ('vol_window_long', 150),
            ('rolling_window_percentile', 100),
            ('emd_window', 350),
            ('min_history_for_features', 350),
            ('use_diff_ukf', True)
        ]:
            setattr(self, param_name, metadata.get(param_name, default_value))

        self.feature_columns = metadata.get('feature_columns', self._default_feature_columns())
        print("✅ Базовые параметры восстановлены")

        # === 3) LSTM (required) ===
        keras_path = f"{path}_lstm.keras"
        if not os.path.exists(keras_path):
            raise RuntimeError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: файл LSTM модели не найден: {keras_path}")

        if self.model is None:
            self.model = tf.keras.models.load_model(
                keras_path,
                custom_objects={'MultiHeadAttention': tf.keras.layers.MultiHeadAttention}
            )
        print(f"✅ LSTM модель загружена: {keras_path}")

        # === 4) state.pkl (required) ===
        state_path = f"{path}_state.pkl"
        if not os.path.exists(state_path):
            raise RuntimeError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: файл состояния не найден: {state_path}")

        with open(state_path, 'rb') as f:
            model_state = pickle.load(f)
        print("✅ Состояние модели загружено")

        # === 5) restore canonical filter state ===
        for k in ['_last_state', '_last_P', '_state_initialized', '_step_counter', '_last_anomaly_time']:
            if k not in model_state:
                raise RuntimeError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: отсутствует '{k}' в state.pkl")

        # _last_state -> Variable([x])
        last_state_val = float(np.asarray(model_state['_last_state']).item())
        new_state = tf.constant([last_state_val], dtype=tf.float32)
        if hasattr(self, '_last_state') and isinstance(self._last_state, tf.Variable):
            self._last_state.assign(new_state)
        else:
            self._last_state = tf.Variable(new_state, trainable=False, dtype=tf.float32, name='_last_state')

        # _last_P stored [[p11]] -> Variable([[[p11]]])
        P_saved = np.asarray(model_state['_last_P'], dtype=np.float32)
        p11 = float(P_saved.reshape(-1)[0].item()) if P_saved.size > 0 else 0.1
        new_P = tf.constant([[[p11]]], dtype=tf.float32)
        if hasattr(self, '_last_P') and isinstance(self._last_P, tf.Variable):
            self._last_P.assign(new_P)
        else:
            self._last_P = tf.Variable(new_P, trainable=False, dtype=tf.float32, name='_last_P')

        # _last_volatility (optional but we use it)
        last_vol_val = float(np.asarray(model_state.get('_last_volatility', 0.1)).item())
        new_vol = tf.constant([last_vol_val], dtype=tf.float32)
        if hasattr(self, '_last_volatility') and isinstance(self._last_volatility, tf.Variable):
            self._last_volatility.assign(new_vol)
        else:
            self._last_volatility = tf.Variable(new_vol, trainable=False, dtype=tf.float32, name='last_volatility')

        # flags/timers
        if hasattr(self, '_state_initialized'):
            self._state_initialized.assign(bool(model_state['_state_initialized']))
        if hasattr(self, '_step_counter'):
            self._step_counter.assign(int(model_state['_step_counter']))
        if hasattr(self, '_last_anomaly_time'):
            self._last_anomaly_time.assign(int(model_state['_last_anomaly_time']))

        print("✅ Состояние фильтра/online восстановлено")

        # === 5.1) restore anomaly buffer (optional) ===
        if hasattr(self, 'anomaly_buffer') and model_state.get('anomaly_buffer') is not None:
            buf = np.asarray(model_state['anomaly_buffer'], dtype=np.float32).reshape(-1)
            target_n = int(model_state.get('anomaly_buffer_size', getattr(self, 'anomaly_buffer_size', buf.shape[0])))
            if buf.shape[0] != target_n:
                if buf.shape[0] > target_n:
                    buf = buf[:target_n]
                else:
                    buf = np.concatenate([buf, np.zeros([target_n - buf.shape[0]], dtype=np.float32)], axis=0)
            self.anomaly_buffer.assign(buf)
            print("✅ anomaly_buffer восстановлен")

        if hasattr(self, 'buffer_index'):
            self.buffer_index.assign(int(model_state.get('buffer_index', 0)))
            print("✅ buffer_index восстановлен")

        # === 6) restore scalar params ===
        scalar_params = ['max_width_factors_logits', 'lambda_entropy', 'threshold_ema']

        for p in scalar_params:
            if p in model_state and hasattr(self, p):
                getattr(self, p).assign(model_state[p])
        print("✅ Скалярные параметры восстановлены")

        # === 7) restore embedded regime selector ===
        selector_state = model_state.get('regime_selector_state', None)
        if selector_state is not None:
            if not hasattr(self, 'regime_selector') or self.regime_selector is None:
                raise RuntimeError("❌ regime_selector отсутствует в объекте (не инициализирован)")
            rs = self.regime_selector

            if selector_state.get('regime_scales') is not None and hasattr(rs, 'regime_scales'):
                rs.regime_scales.assign(tf.constant(selector_state['regime_scales'], dtype=tf.float32))
            if selector_state.get('temperature') is not None and hasattr(rs, 'temperature'):
                rs.temperature.assign(tf.constant(selector_state['temperature'], dtype=tf.float32))

            hist = selector_state.get('history', None)
            if hist is not None:
                hist_t = tf.constant(hist, dtype=tf.float32)
                if hasattr(rs, '_vol_history'):
                    rs._vol_history.assign(hist_t)
                elif hasattr(rs, 'vol_history'):
                    rs.vol_history.assign(hist_t)

            # centers
            rs.learnable_centers = bool(selector_state.get('learnable_centers', getattr(rs, 'learnable_centers', False)))
            if rs.learnable_centers and 'center_logits' in selector_state and hasattr(rs, 'center_logits'):
                rs.center_logits.assign(tf.constant(selector_state['center_logits'], dtype=tf.float32))

            print("✅ regime_selector восстановлен")
        else:
            print("⚠️ regime_selector_state отсутствует в state.pkl (будут использованы текущие значения)")

        # === 8) restore embedded diff ukf ===
        diff_state = model_state.get('diff_ukf_state', None)
        if diff_state is not None and bool(getattr(self, 'use_diff_ukf', False)) and hasattr(self, 'diff_ukf_component'):
            if isinstance(diff_state, dict) and 'd_raw' in diff_state and diff_state['d_raw'] is not None:
                try:
                    self.diff_ukf_component.spec_param.d_raw.assign(diff_state['d_raw'])
                    print("✅ diff_ukf d_raw восстановлен")
                except Exception as e:
                    print(f"⚠️ Не удалось восстановить diff_ukf d_raw: {str(e)}")

        # === 9) restore scalers/groups (your current design treats scalers as critical) ===
        if model_state.get('feature_scalers') is None:
            raise RuntimeError(
                "❌ КРИТИЧЕСКАЯ ОШИБКА: state.pkl не содержит feature_scalers.\n"
                "   Пересохраните модель через model.save(path)."
            )

        self.feature_scalers = {k: _safe_deserialize_scaler(v) for k, v in model_state['feature_scalers'].items()}
        if self.feature_scalers.get('Y') is None:
            raise RuntimeError("❌ КРИТИЧЕСКАЯ ОШИБКА: скейлер 'Y' не восстановлен")
        print("✅ Скейлеры восстановлены (включая Y)")

        if model_state.get('scale_groups') is not None:
            self.scale_groups = model_state['scale_groups']
            print("✅ scale_groups восстановлены")
        if model_state.get('best_scalers') is not None:
            self.best_scalers = model_state['best_scalers']
            print("✅ best_scalers восстановлены")

        # === 10) tracking + embedded best snapshot ===
        self.best_val_loss = float(model_state.get('best_val_loss', float('inf')))
        self.best_epoch = int(model_state.get('best_epoch', 0))
        self.patience_counter = int(model_state.get('patience_counter', 0))

        if 'best_weights_dict' in model_state:
            self.best_weights_dict = model_state['best_weights_dict']
            if self.best_weights_dict is not None:
                print("✅ best_weights_dict восстановлен (можно продолжать early stopping без потери лучшего состояния)")
            else:
                print("ℹ️ best_weights_dict в файле пуст (None)")

        print("\n" + "=" * 60)
        print("✅ МОДЕЛЬ УСПЕШНО ЗАГРУЖЕНА")
        try:
            print(f"   step_counter: {int(self._step_counter.numpy())}")
            print(f"   state_initialized: {bool(self._state_initialized.numpy())}")
            print(f"   _last_state shape: {self._last_state.shape}")
            print(f"   _last_P shape: {self._last_P.shape}")
            print(f"   _last_volatility: {float(self._last_volatility.numpy().reshape(-1)[0]):.6f}")
        except Exception:
            pass
        print("=" * 60)
        return self

    def get_current_weights(self) -> Dict[str, Any]:
        """
        Сохранение ПОЛНОГО состояния модели с усреднением по батчу для независимости от размера батча.
        Все состояния фильтра сохраняются как скаляры/матрицы без размерности батча.
        """

        def _mean_scalar(x) -> float:
            arr = np.asarray(x)
            if arr.ndim == 0:
                return float(arr.item())
            return float(np.mean(arr.reshape(-1)).item())

        # === UKF: last_state ===
        last_state_val = _mean_scalar(self._last_state.numpy())

        # === UKF: last_P -> [[p11]] ===
        last_P_np = np.asarray(self._last_P.numpy())
        if last_P_np.ndim == 3:
            P_mean = np.mean(last_P_np, axis=0).reshape(1, 1)
            last_P_val = P_mean.tolist()
        elif last_P_np.ndim == 2:
            last_P_val = last_P_np.reshape(1, 1).tolist()
        elif last_P_np.ndim == 0:
            last_P_val = [[float(last_P_np.item())]]
        else:
            last_P_val = [[float(np.mean(last_P_np.reshape(-1)).item())]]

        # === NEW: last_volatility (scalar) ===
        if hasattr(self, "_last_volatility") and self._last_volatility is not None:
            last_volatility_val = _mean_scalar(self._last_volatility.numpy())
        else:
            last_volatility_val = 0.1  # дефолт для online_predict

        # === anomaly buffer ===
        anomaly_buffer_val = self.anomaly_buffer.value().numpy() if hasattr(self, "anomaly_buffer") else None
        buffer_index_val = int(self.buffer_index.numpy()) if hasattr(self, "buffer_index") else 0
        anomaly_buffer_size_val = int(getattr(self, "anomaly_buffer_size", 100))

        # === diff ukf state ===
        diff_ukf_state = {"d_raw": float(np.log(0.1))}
        if getattr(self, "use_diff_ukf", False) and hasattr(self, "diff_ukf_component"):
            if hasattr(self.diff_ukf_component, "spec_param") and hasattr(self.diff_ukf_component.spec_param, "d_raw"):
                diff_ukf_state["d_raw"] = _mean_scalar(self.diff_ukf_component.spec_param.d_raw.numpy())

        # === regime selector state ===
        regime_selector_state = {
            "regime_scales": np.array([2.96, 4.44, 6.16], dtype=np.float32),
            "temperature": 0.8,  # ← Оставлено 0.8 (согласовано с __init__)
            "history": np.zeros([1, 100], dtype=np.float32),
            "center_logits": np.log([0.12, 0.35, 0.75]),
        }
        if hasattr(self, "regime_selector") and self.regime_selector is not None:
            rs = self.regime_selector
            if hasattr(rs, "regime_scales"):
                regime_selector_state["regime_scales"] = np.asarray(rs.regime_scales.numpy(), dtype=np.float32)
            if hasattr(rs, "temperature"):
                regime_selector_state["temperature"] = float(np.asarray(rs.temperature.numpy()).item())
            if hasattr(rs, "_vol_history"):
                regime_selector_state["history"] = np.asarray(rs._vol_history.numpy(), dtype=np.float32)
            elif hasattr(rs, "vol_history"):
                regime_selector_state["history"] = np.asarray(rs.vol_history.numpy(), dtype=np.float32)
            if hasattr(rs, "center_logits") and getattr(rs, "learnable_centers", False):
                regime_selector_state["center_logits"] = np.asarray(rs.center_logits.numpy(), dtype=np.float32)

        state_dict: Dict[str, Any] = {
            # === LSTM ===
            "lstm_weights": self.model.get_weights() if self.model is not None else None,

            # === UKF state ===
            "_last_state": last_state_val,      # float
            "_last_P": last_P_val,              # [[p11]]
            "_last_volatility": last_volatility_val,  # float (NEW)

            "_state_initialized": bool(self._state_initialized.numpy()) if hasattr(self, "_state_initialized") else False,
            "_step_counter": int(self._step_counter.numpy()) if hasattr(self, "_step_counter") else 0,
            "_last_anomaly_time": int(self._last_anomaly_time.numpy()) if hasattr(self, "_last_anomaly_time") else -100,

            # === anomaly detector buffer ===
            "anomaly_buffer": np.asarray(anomaly_buffer_val, dtype=np.float32) if anomaly_buffer_val is not None else None,
            "buffer_index": buffer_index_val,
            "anomaly_buffer_size": anomaly_buffer_size_val,

            # === trainable/important params ===
            "max_width_factors_logits": self.max_width_factors_logits.numpy() if hasattr(self, "max_width_factors_logits") else np.array([np.log(1.5)] * 3),

            # === diff ukf ===
            "diff_ukf_state": diff_ukf_state,

            # === regime selector ===
            "regime_selector_state": regime_selector_state,

            # === scalers ===
            "feature_scalers": self.feature_scalers.copy() if hasattr(self, "feature_scalers") and self.feature_scalers is not None else None,
            "best_scalers": self.best_scalers.copy() if hasattr(self, "best_scalers") and self.best_scalers is not None else None,
            "scale_groups": self.scale_groups.copy() if hasattr(self, "scale_groups") and self.scale_groups is not None else None,

            # === tracking ===
            "best_val_loss": float(getattr(self, "best_val_loss", float("inf"))),
            "best_epoch": int(getattr(self, "best_epoch", 0)),
            "patience_counter": int(getattr(self, "patience_counter", 0)),

            # === flags/meta ===
            "use_diff_ukf": bool(getattr(self, "use_diff_ukf", False)),
            "num_modes": int(getattr(self, "num_modes", 1)),
            "state_dim": int(getattr(self, "state_dim", 1)),
            "seq_len": int(getattr(self, "seq_len", 1)),

            # === regularization helpers ===
            "lambda_entropy": self.lambda_entropy.numpy() if hasattr(self, "lambda_entropy") else 0.02,
            "threshold_ema": self.threshold_ema.numpy() if hasattr(self, "threshold_ema") else 3.0,

            # === coverage mixing parameter ===
            "coverage_mixing_alpha": self.coverage_mixing_alpha.numpy() if hasattr(self, "coverage_mixing_alpha") else 0.405,  # log(0.6/(1-0.6))
        }

        print("✅ Полное состояние модели сохранено (усреднённое по батчу)")
        print(f"   • _last_state: скаляр = {state_dict['_last_state']:.6f}")
        print(f"   • _last_P: форма = {[1, 1]}")
        print(f"   • _last_volatility: скаляр = {state_dict['_last_volatility']:.6f}")
        return state_dict

    def load_best_weights(self) -> bool:
        """
        Загрузка лучших весов из self.best_weights_dict: восстановление полного состояния модели.
        """
        if self.best_weights_dict is None:
            print("⚠️  Нет лучших весов для загрузки (best_weights_dict is None)")
            return False

        try:
            print("\n" + "=" * 80)
            print("📥 ЗАГРУЗКА ЛУЧШИХ ВЕСОВ — ВОССТАНОВЛЕНИЕ ПОЛНОГО СОСТОЯНИЯ")
            print("=" * 80)

            d = self.best_weights_dict

            # --- 0) validate minimal structure ---
            required_keys = ["lstm_weights", "_last_state", "_last_P", "regime_selector_state"]
            missing = [k for k in required_keys if k not in d or d[k] is None]
            if missing:
                print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: отсутствуют обязательные компоненты: {missing}")
                return False

            # --- 1) LSTM weights ---
            if self.model is not None and d.get("lstm_weights") is not None:
                self.model.set_weights(d["lstm_weights"])
                print("✅ LSTM веса загружены")

            # --- 1.1) Coverage mixing parameter ---
            if "coverage_mixing_alpha" in d and d["coverage_mixing_alpha"] is not None:
                if hasattr(self, "coverage_mixing_alpha"):
                    self.coverage_mixing_alpha.assign(float(d["coverage_mixing_alpha"]))
                    print(f"✅ coverage_mixing_alpha: {float(d['coverage_mixing_alpha']):.6f}")
                else:
                    # Initialize if it doesn't exist yet
                    initial_logit = d["coverage_mixing_alpha"]
                    self.coverage_mixing_alpha = tf.Variable(
                        initial_value=float(initial_logit),
                        trainable=True,
                        dtype=tf.float32,
                        name='coverage_mixing_alpha'
                    )
                    print(f"➕ coverage_mixing_alpha (новый): {float(d['coverage_mixing_alpha']):.6f}")

            # --- 2) UKF filter state ---
            print("\n🔧 Восстановление состояния фильтра UKF...")

            # _last_state
            saved_state = d.get("_last_state", 0.0)
            new_state = tf.constant([float(np.asarray(saved_state).item())], dtype=tf.float32)
            if hasattr(self, "_last_state") and isinstance(self._last_state, tf.Variable):
                self._last_state.assign(new_state)
            else:
                self._last_state = tf.Variable(new_state, trainable=False, dtype=tf.float32, name="_last_state")
            print(f"   ✅ _last_state: {float(new_state.numpy()[0]):.6f}")

            # _last_P
            saved_P = d.get("_last_P", [[0.1]])
            saved_P_np = np.asarray(saved_P, dtype=np.float32)
            if saved_P_np.ndim == 0:
                P11 = float(saved_P_np.item())
            else:
                P11 = float(saved_P_np.reshape(-1)[0])
            new_P = tf.constant([[[P11]]], dtype=tf.float32)  # [1,1,1]
            if hasattr(self, "_last_P") and isinstance(self._last_P, tf.Variable):
                self._last_P.assign(new_P)
            else:
                self._last_P = tf.Variable(new_P, trainable=False, dtype=tf.float32, name="_last_P")
            print("   ✅ _last_P: восстановлен как [1, 1, 1]")

            # NEW: _last_volatility
            saved_vol = d.get("_last_volatility", 0.1)
            new_vol = tf.constant([float(np.asarray(saved_vol).item())], dtype=tf.float32)  # shape [1]
            if hasattr(self, "_last_volatility") and isinstance(self._last_volatility, tf.Variable):
                self._last_volatility.assign(new_vol)
            else:
                # создаём, чтобы online_predict не делал lazy-init по дефолту
                self._last_volatility = tf.Variable(new_vol, trainable=False, dtype=tf.float32, name="last_volatility")
            print(f"   ✅ _last_volatility: {float(new_vol.numpy()[0]):.6f}")

            # --- 2.1) Restore state flags/timers ---
            for k in ["_state_initialized", "_step_counter", "_last_anomaly_time"]:
                if k in d and d[k] is not None and hasattr(self, k):
                    try:
                        v = getattr(self, k)
                        if isinstance(v, tf.Variable):
                            v.assign(d[k])
                        else:
                            setattr(self, k, d[k])
                        print(f"   ✅ {k} восстановлен")
                    except Exception as e:
                        print(f"   ⚠️ Ошибка при восстановлении {k}: {str(e)}")

            # --- 3) anomaly buffer ---
            print("\n🔧 Восстановление состояния детектора аномалий...")

            if "anomaly_buffer_size" in d and d["anomaly_buffer_size"] is not None:
                try:
                    self.anomaly_buffer_size = int(d["anomaly_buffer_size"])
                except Exception:
                    pass

            if "anomaly_buffer" in d and d["anomaly_buffer"] is not None and hasattr(self, "anomaly_buffer"):
                try:
                    buf = np.asarray(d["anomaly_buffer"], dtype=np.float32).reshape(-1)
                    target_n = int(getattr(self, "anomaly_buffer_size", buf.shape[0]))
                    if buf.shape[0] != target_n:
                        if buf.shape[0] > target_n:
                            buf = buf[:target_n]
                        else:
                            buf = np.concatenate([buf, np.zeros([target_n - buf.shape[0]], dtype=np.float32)], axis=0)
                    self.anomaly_buffer.assign(buf)
                    print("   ✅ anomaly_buffer восстановлен")
                except Exception as e:
                    print(f"   ⚠️ Ошибка при восстановлении anomaly_buffer: {str(e)}")

            if "buffer_index" in d and d["buffer_index"] is not None and hasattr(self, "buffer_index"):
                try:
                    self.buffer_index.assign(int(d["buffer_index"]))
                    print("   ✅ buffer_index восстановлен")
                except Exception as e:
                    print(f"   ⚠️ Ошибка при восстановлении buffer_index: {str(e)}")

            # --- 4) restore trainable/important params ---
            print("\n🔧 Восстановление обучаемых/важных параметров...")

            restore_map = [
                ("max_width_factors_logits", "max_width_factors_logits"),
                ("lambda_entropy", "lambda_entropy"),
                ("threshold_ema", "threshold_ema"),
            ]

            for attr_name, key in restore_map:
                if key in d and d[key] is not None and hasattr(self, attr_name):
                    try:
                        var = getattr(self, attr_name)
                        if isinstance(var, tf.Variable):
                            var.assign(d[key])
                        else:
                            setattr(self, attr_name, d[key])
                        print(f"   ✅ {attr_name} восстановлен")
                    except Exception as e:
                        print(f"   ⚠️ Ошибка при восстановлении {attr_name}: {str(e)}")

            # --- 5) restore diff ukf ---
            print("\n🔧 Восстановление состояния дифференцируемого UKF...")
            if getattr(self, "use_diff_ukf", False) and "diff_ukf_state" in d and d["diff_ukf_state"] is not None:
                try:
                    if (hasattr(self, "diff_ukf_component")
                        and hasattr(self.diff_ukf_component, "spec_param")
                        and hasattr(self.diff_ukf_component.spec_param, "d_raw")
                        and isinstance(self.diff_ukf_component.spec_param.d_raw, tf.Variable)):
                        self.diff_ukf_component.spec_param.d_raw.assign(d["diff_ukf_state"].get("d_raw"))
                        print("   ✅ diff_ukf d_raw восстановлен")
                except Exception as e:
                    print(f"   ⚠️ Ошибка при восстановлении diff_ukf d_raw: {str(e)}")

            # --- 6) restore regime selector ---
            print("\n🔧 Восстановление состояния Volatility Regime Selector...")
            if "regime_selector_state" in d and d["regime_selector_state"] is not None and hasattr(self, "regime_selector"):
                selector_state = d["regime_selector_state"]
                restored = []
                rs = self.regime_selector

                if "regime_scales" in selector_state and selector_state["regime_scales"] is not None and hasattr(rs, "regime_scales"):
                    try:
                        rs.regime_scales.assign(tf.constant(selector_state["regime_scales"], dtype=tf.float32))
                        restored.append("regime_scales")
                    except Exception as e:
                        print(f"   ⚠️ Ошибка при восстановлении regime_scales: {str(e)}")

                if "temperature" in selector_state and selector_state["temperature"] is not None and hasattr(rs, "temperature"):
                    try:
                        # 🔑 ГИБРИДНЫЙ ПОДХОД v6: ЯВНЫЙ CLIP ПРИ ВОССТАНОВЛЕНИИ
                        temp_val = float(selector_state["temperature"])
                        temp_val = max(0.3, min(10.0, temp_val))  # ← ✅ [0.3, 10.0] согласовано с __init__
                        rs.temperature.assign(tf.constant(temp_val, dtype=tf.float32))
                        restored.append("temperature")
                        print(f"   ✅ temperature: {temp_val:.3f} (clip [0.3, 10.0])")  # ← Обновлённый лог
                    except Exception as e:
                        print(f"   ⚠️ Ошибка при восстановлении temperature: {str(e)}")

                if "history" in selector_state and selector_state["history"] is not None:
                    try:
                        hist = tf.constant(selector_state["history"], dtype=tf.float32)
                        if hasattr(rs, "_vol_history"):
                            rs._vol_history.assign(hist)
                            restored.append("_vol_history")
                        elif hasattr(rs, "vol_history"):
                            rs.vol_history.assign(hist)
                            restored.append("vol_history")
                    except Exception as e:
                        print(f"   ⚠️ Ошибка при восстановлении history: {str(e)}")

                if (hasattr(rs, "learnable_centers") and rs.learnable_centers and
                    "center_logits" in selector_state and selector_state["center_logits"] is not None and
                    hasattr(rs, "center_logits")):
                    try:
                        rs.center_logits.assign(tf.constant(selector_state["center_logits"], dtype=tf.float32))
                        restored.append("center_logits")
                    except Exception as e:
                        print(f"   ⚠️ Ошибка при восстановлении center_logits: {str(e)}")

                if restored:
                    print(f"✅ Восстановлены параметры: {', '.join(restored)}")
                else:
                    print("⚠️ Не удалось восстановить параметры Regime Selector")

            # --- 7) scalers ---
            print("\n🔧 Восстановление скейлеров...")
            if "feature_scalers" in d:
                self.feature_scalers = d["feature_scalers"]
                print("✅ feature_scalers восстановлены")
            if "best_scalers" in d:
                self.best_scalers = d["best_scalers"]
                print("✅ best_scalers восстановлены")
            if "scale_groups" in d:
                self.scale_groups = d["scale_groups"]
                print("✅ scale_groups восстановлены")

            # --- 8) tracking ---
            self.best_val_loss = d.get("best_val_loss", float("inf"))
            self.best_epoch = d.get("best_epoch", 0)
            self.patience_counter = d.get("patience_counter", 0)

            # --- 9) final log ---
            print("\n" + "=" * 80)
            print("✅ ПОЛНОЕ СОСТОЯНИЕ УСПЕШНО ВОССТАНОВЛЕНО")
            print("=" * 80)
            print(f"   📅 Эпоха: {int(self.best_epoch) + 1}")
            print(f"   📉 Val Loss: {float(self.best_val_loss):.6f}")
            if hasattr(self, "_step_counter"):
                print(f"   🔁 Счётчик шагов: {int(self._step_counter.numpy())}")
            if hasattr(self, "_state_initialized"):
                print(f"   🧠 Состояние фильтра: {'Инициализировано' if bool(self._state_initialized.numpy()) else 'Не инициализировано'}")
            if hasattr(self, "regime_selector") and hasattr(self.regime_selector, "regime_scales") and hasattr(self.regime_selector, "temperature"):
                print(f"   🎭 Regime Selector: scales={self.regime_selector.regime_scales.numpy()}, temp={float(self.regime_selector.temperature.numpy()):.3f}")
            if hasattr(self, "inflation_base_factor"):
                print(f"   💧 Adaptive inflation: base={float(tf.nn.softplus(self.inflation_base_factor).numpy()):.3f}")
            print("=" * 80)

            return True

        except Exception as e:
            print("\n❌ КРИТИЧЕСКАЯ ОШИБКА при загрузке лучших весов:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def reset_best_weights_tracking(self):
        """Сброс отслеживания лучших весов (вызывается в начале fit)."""
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_weights_dict = None
        self.patience_counter = 0

        # === СБРОС СОСТОЯНИЯ ДЕТЕКТОРА АНОМАЛИЙ ===
        if hasattr(self, 'anomaly_buffer'):
            try:
                self.anomaly_buffer.assign(tf.zeros_like(self.anomaly_buffer))
            except Exception:
                # fallback на случай, если anomaly_buffer не Variable (редко)
                self.anomaly_buffer = tf.Variable(tf.zeros_like(self.anomaly_buffer), trainable=False)

        if hasattr(self, 'buffer_index'):
            try:
                self.buffer_index.assign(0)
            except Exception:
                self.buffer_index = tf.Variable(0, dtype=tf.int32, trainable=False, name='buffer_index')

        # === ВАЖНО: сброс таймера последней аномалии (гистерезис) ===
        if hasattr(self, '_last_anomaly_time'):
            try:
                # обычно shape [1] или [B]; делаем универсально
                self._last_anomaly_time.assign(tf.fill(tf.shape(self._last_anomaly_time),
                                                       tf.constant(-100, dtype=self._last_anomaly_time.dtype)))
            except Exception:
                pass

        # === NEW: сброс last_volatility, если используешь (чтобы fit стартовал одинаково) ===
        if hasattr(self, '_last_volatility') and self._last_volatility is not None:
            try:
                # Гарантируем форму [1] при сбросе
                default_vol = tf.constant([0.1], dtype=tf.float32)
                if self._last_volatility.shape == [1]:
                    self._last_volatility.assign(default_vol)
                else:
                    self._last_volatility.assign(
                        tf.zeros_like(self._last_volatility) + tf.constant(0.1, tf.float32)
                    )
                print("   ✅ _last_volatility сброшен к 0.1")
            except Exception as e:
                print(f"   ⚠️ Не удалось сбросить _last_volatility: {str(e)}")

        print("✓ Отслеживание лучших весов сброшено")

    @tf.function(jit_compile=False)
    def _online_predict_step(
        self,
        X_scaled: tf.Tensor,
        initial_state: tf.Tensor,
        initial_covariance: tf.Tensor,
        last_volatility: tf.Tensor
    ) -> Tuple[
        tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
        tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]
    ]:
        B = tf.shape(X_scaled)[0]
        T = tf.shape(X_scaled)[1]

        # === 1) LSTM forward pass ===
        lstm_outputs = self.model(X_scaled, training=False)
        params_output = lstm_outputs["params"]  # [B, T, 37]

        # === 2) LSTM params -> configs ===
        vol_context, ukf_params, inflation_config, student_t_config = self.process_lstm_output(params_output)

        # === 3) UKF filtering ===
        initial_state = tf.reshape(initial_state, [B, self.state_dim])
        initial_covariance = tf.reshape(initial_covariance, [B, self.state_dim, self.state_dim])

        level_idx = self.feature_columns.index("level")
        y_level_batch = X_scaled[:, :, level_idx]  # [B, T]

        # last_volatility: ожидаем [B] (если пришло [B,1] — приведём)
        last_volatility = tf.reshape(tf.squeeze(last_volatility), [B])

        # last_anomaly_time: если храним в self, используем; иначе -100
        if hasattr(self, "_last_anomaly_time") and self._last_anomaly_time is not None:
            last_anom = tf.cast(tf.reshape(self._last_anomaly_time, [B]), tf.int32)
        else:
            last_anom = tf.fill([B], tf.constant(-100, dtype=tf.int32))

        results = self.adaptive_ukf_filter(
            X_scaled,
            y_level_batch,
            vol_context,
            ukf_params,
            inflation_config,
            student_t_config,
            initial_state,
            initial_covariance,
            inflation_state_input={"last_anomaly_time": last_anom},
            initial_volatility=last_volatility
        )

        x_filtered = results[0]                 # [B, T, 1]
        innovations = results[1]                # [B, T, 1]
        volatility_levels = results[2]          # [B, T, 1]
        inflation_factors = results[3]          # [B, T, 1]
        final_state = results[4]                # [B, 1]
        final_covariance = results[5]           # [B, 1, 1]
        correction_adaptive_hist = results[6]   # [B, T, 1]

        final_inflation = tf.reshape(inflation_factors[:, -1, :], [B])     # [B]
        final_volatility = tf.reshape(volatility_levels[:, -1, :], [B])    # [B]
        correction_adaptive = correction_adaptive_hist[:, -1, :]           # [B,1]

        # === 4) explicit one-step predict ===
        t_last = T - 1

        q_base_final = tf.gather(ukf_params["q_base"], t_last, axis=1)                 # [B,1]
        q_sensitivity_final = tf.gather(ukf_params["q_sensitivity"], t_last, axis=1)   # [B,1]
        q_floor_final = tf.gather(ukf_params["q_floor"], t_last, axis=1)               # [B,1]

        relax_base_final = tf.gather(ukf_params["relax_base"], t_last, axis=1)         # [B,1]
        relax_sensitivity_final = tf.gather(ukf_params["relax_sensitivity"], t_last, axis=1)  # [B,1]
        alpha_base_final = tf.gather(ukf_params["alpha_base"], t_last, axis=1)         # [B,1]
        alpha_sensitivity_final = tf.gather(ukf_params["alpha_sensitivity"], t_last, axis=1)  # [B,1]
        kappa_base_final = tf.gather(ukf_params["kappa_base"], t_last, axis=1)         # [B,1]
        kappa_sensitivity_final = tf.gather(ukf_params["kappa_sensitivity"], t_last, axis=1)  # [B,1]

        inf_factor_final = tf.reshape(final_inflation, [B, 1])  # [B,1]

        forecast, std_dev, _ = self._explicit_predict_next_step(
            final_state,
            final_covariance,
            final_volatility,
            q_base_final, q_sensitivity_final, q_floor_final,
            inf_factor_final,
            relax_base_final, relax_sensitivity_final,
            alpha_base_final, alpha_sensitivity_final,
            kappa_base_final, kappa_sensitivity_final
        )

        # === 5) CI calibration (LSTM-only; НЕ перезаписываем student_t_config) ===
        student_t_config, target_coverage, regime_info = self._get_calibration_params(
            final_volatility,
            student_t_config=student_t_config,
            correction_adaptive=correction_adaptive,
            training=False
        )

        ci_lower, ci_upper, _, width_penalty_from_ci = self._calibrate_confidence_interval(
            forecast,
            std_dev,
            final_volatility,
            student_t_config,
            innovations=innovations[:, -10:, :] if innovations is not None else None,
            regime_assignment=regime_info.get("regime_assignment", None)
        )

        ci_min = tf.minimum(ci_lower, ci_upper)
        ci_max = tf.maximum(ci_lower, ci_upper)

        return (
            forecast,
            std_dev,
            ci_min,
            ci_max,
            final_state,
            final_covariance,
            final_volatility,
            final_inflation,
            target_coverage,
            regime_info
        )

    def online_predict(
        self,
        df: pd.DataFrame,
        reset_state: bool = False,
        return_components: bool = False,
        ground_truth_available: bool = False
    ) -> Dict[str, Any]:
        required_cols = ["Open", "High", "Low", "Close"]
        missing_cols = [c for c in required_cols if c not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")
        if len(df) < self.min_history_for_features:
            raise ValueError(
                f"Требуется минимум {self.min_history_for_features} точек для расчёта признаков, "
                f"получено {len(df)}"
            )

        # 1) features
        features_df = self.prepare_features(df, mode="batch")
        if len(features_df) < self.seq_len:
            raise ValueError(
                f"После расчёта признаков осталось {len(features_df)} точек, требуется минимум {self.seq_len}"
            )

        # 2) model window
        X_window = features_df.tail(self.seq_len).copy()
        X_scaled_df = self._scale_features(X_window)
        X_scaled_np = X_scaled_df[self.feature_columns].values.astype(np.float32)

        X_scaled_tensor = tf.convert_to_tensor(
            X_scaled_np.reshape(1, self.seq_len, len(self.feature_columns)),
            dtype=tf.float32
        )

        # 3) ensure state vars exist (после чистки __init__ иногда забывают добавить)
        if not hasattr(self, "_state_initialized"):
            self._state_initialized = tf.Variable(False, trainable=False, dtype=tf.bool, name="state_initialized")
        if not hasattr(self, "_step_counter"):
            self._step_counter = tf.Variable(0, trainable=False, dtype=tf.int64, name="step_counter")
        if not hasattr(self, "_last_anomaly_time"):
            self._last_anomaly_time = tf.Variable(-100, trainable=False, dtype=tf.int32, name="last_anomaly_time")
        if not hasattr(self, "_last_volatility"):
            self._last_volatility = tf.Variable([0.1], trainable=False, dtype=tf.float32, name="last_volatility")

        # 4) init / load filter state
        B = 1
        if reset_state or (not bool(self._state_initialized.numpy())):
            # Используем 'level' из масштабированных признаков
            level_idx = self.feature_columns.index("level")
            level_series = X_scaled_np[:, level_idx]  # [seq_len]

            window_len = min(10, self.seq_len)
            window_std = float(np.std(level_series[:window_len]))
            initial_variance = max(window_std ** 2, 0.05)
            initial_variance = min(initial_variance, 0.5)

            initial_state_val = float(np.mean(level_series[:min(5, self.seq_len)]))
            initial_state = tf.constant([[initial_state_val]], dtype=tf.float32)      # [1,1]
            initial_covariance = tf.constant([[[initial_variance]]], dtype=tf.float32)  # [1,1,1]

            initial_volatility = tf.constant([window_std], dtype=tf.float32)  # [1]

            self._last_state.assign(tf.squeeze(initial_state, axis=1))
            self._last_P.assign(initial_covariance)
            self._last_volatility.assign(initial_volatility)
            self._state_initialized.assign(True)
            self._step_counter.assign(0)
            self._last_anomaly_time.assign(tf.constant(-100, dtype=tf.int32))

            if self.debug_mode:
                print(f"🔄 Состояние фильтра ИНИЦИАЛИЗИРОВАНО (reset_state={reset_state})")
                print(f"   initial_state_val={initial_state_val:.6f}, initial_variance={initial_variance:.6f}, "
                      f"initial_volatility={window_std:.6f}")
        else:
            initial_state = tf.tile(tf.reshape(self._last_state, [1, self.state_dim]), [B, 1])
            initial_covariance = tf.tile(tf.reshape(self._last_P, [1, self.state_dim, self.state_dim]), [B, 1, 1])
            initial_volatility = tf.reshape(self._last_volatility, [B])

            if self.debug_mode:
                print("🔄 Состояние фильтра ЗАГРУЖЕНО из предыдущего вызова")
                print(f"   _last_state={self._last_state.numpy()[0]:.6f}, _last_P={self._last_P.numpy()[0,0,0]:.6f}, "
                      f"_last_volatility={float(self._last_volatility.numpy()[0]):.6f}")

        # 5) predict step (graph)
        with tf.device(self.device):
            (
                forecast_scaled,
                std_dev_scaled,
                ci_lower_scaled,
                ci_upper_scaled,
                final_state,
                final_covariance,
                final_volatility,
                final_inflation,
                target_coverage,
                regime_info
            ) = self._online_predict_step(
                X_scaled_tensor,
                initial_state,
                initial_covariance,
                initial_volatility
            )

        # 6) persist state
        self._last_state.assign(tf.squeeze(final_state, axis=1))
        self._last_P.assign(final_covariance)
        self._last_volatility.assign(tf.reshape(final_volatility, [1]))
        self._step_counter.assign_add(1)

        # 7) inverse transform
        forecast_original = self.inverse_transform_target(forecast_scaled.numpy())
        ci_lower_original = self.inverse_transform_target(ci_lower_scaled.numpy())
        ci_upper_original = self.inverse_transform_target(ci_upper_scaled.numpy())

        # std in original scale
        if self.feature_scalers is not None and "Y" in self.feature_scalers:
            y_scaler = self.feature_scalers["Y"]
            # Handle different scaler types appropriately
            if hasattr(y_scaler, 'scale_'):  # StandardScaler, MinMaxScaler, etc.
                scale_factor = y_scaler.scale_
                std_dev_original = std_dev_scaled.numpy() * scale_factor
            elif hasattr(y_scaler, 'lambdas_'):  # PowerTransformer
                # For PowerTransformer, we need to apply inverse transform properly
                # Since std_dev is a small deviation around the mean, we can approximate
                # by using the derivative of the transform at the mean point
                std_dev_original = std_dev_scaled.numpy()
            else:
                std_dev_original = std_dev_scaled.numpy()
        else:
            std_dev_original = std_dev_scaled.numpy()

        timestamp = df.index[-1] if hasattr(df.index, "__iter__") and len(df.index) > 0 else len(df)

        result = {
            "timestamp": timestamp,
            "level_forecast": forecast_original,
            "level_forecast_scaled": forecast_scaled.numpy(),
            "std_dev": std_dev_original,
            "std_dev_scaled": std_dev_scaled.numpy(),
            "level_ci_lower": ci_lower_original,
            "level_ci_lower_scaled": ci_lower_scaled.numpy(),
            "level_ci_upper": ci_upper_original,
            "level_ci_upper_scaled": ci_upper_scaled.numpy(),
            "volatility_level": float(final_volatility.numpy().reshape(-1)[0]),
            "inflation_factor": float(final_inflation.numpy().reshape(-1)[0]),
            "confidence": float(target_coverage.numpy().reshape(-1)[0]),
            "regime": int(regime_info["regime_assignment"].numpy().reshape(-1)[0]),
            "regime_soft_weights": regime_info["soft_weights"].numpy()[0],
            "regime_entropy": float(regime_info["entropy"].numpy().reshape(-1)[0]),
        }

        if return_components:
            result.update({
                "features_used": X_scaled_df,
                "raw_features": features_df,
                "covariance": final_covariance.numpy(),
                "state": final_state.numpy(),
            })

        return result

    def evaluate(
        self,
        df: pd.DataFrame,
        plot: bool = False,
        N: int = 300
    ) -> Dict[str, float]:
        """
        Честная оценка модели с КРИТИЧЕСКИМИ ИСПРАВЛЕНИЯМИ:
        1. Корректная логика скользящего окна без утечки будущего
        2. Согласованный расчёт метрик покрытия ДИ (единый источник истины)
        3. Восстановление состояния фильтра после оценки
        4. Единый расчёт ширины ДИ относительно волатильности данных (как в обучении)
        """
        print("\n" + "=" * 80)
        print("🔍 НАЧАЛО ЧЕСТНОЙ ОЦЕНКИ МОДЕЛИ (БЕЗ УТЕЧКИ БУДУЩЕГО)")
        print("=" * 80)

        # === 1. ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ ===
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ Отсутствуют обязательные колонки в данных: {missing_cols}")

        window_size = self.min_history_for_features
        if len(df) < window_size + 1:
            raise ValueError(
                f"❌ Недостаточно данных для оценки. Требуется минимум {window_size + 1} точек, "
                f"получено {len(df)}"
            )

        # === 2. СОХРАНЕНИЕ ИСХОДНОГО СОСТОЯНИЯ ФИЛЬТРА ===
        original_state_initialized_val = self._state_initialized.numpy()
        original_last_state_val = self._last_state.numpy().copy() if original_state_initialized_val else None
        original_last_P_val = self._last_P.numpy().copy() if original_state_initialized_val else None
        original_step_counter_val = self._step_counter.numpy()
        print(f"💾 Сохранено исходное состояние фильтра: initialized={original_state_initialized_val}")

        # === 3. СБРОС СОСТОЯНИЯ ДЛЯ ЧИСТОЙ ОЦЕНКИ ===
        self._state_initialized.assign(False)
        print(f"🔄 Состояние фильтра сброшено перед оценкой (чистый старт)")

        # === 4. ИНИЦИАЛИЗАЦИЯ МАССИВОВ ===
        timestamps = []
        true_values_original = []
        true_values_scaled = []
        pred_values_original = []
        pred_values_scaled = []
        pi_lower_original = []
        pi_upper_original = []
        pi_lower_scaled = []
        pi_upper_scaled = []
        volatility_levels = []
        inflation_factors = []
        confidences = []
        regimes = []

        # === 5. СКОЛЬЗЯЩЕЕ ОКНО ОЦЕНКИ ===
        total_steps = len(df) - window_size
        print(f"📊 Обработка {total_steps} шагов оценки (окно = {window_size} точек)...")

        progress_bar = tqdm(range(window_size, len(df)), desc="Оценка модели", unit="шаг")

        try:
            for t in progress_bar:
                try:
                    # История для расчёта признаков: [t-350 .. t-1] (ровно 350 точек ДО прогноза)
                    history_features = df.iloc[t - window_size : t].copy()

                    # Прогноз через исправленный online_predict
                    result = self.online_predict(
                        history_features,
                        reset_state=(t == window_size),  # Сброс ТОЛЬКО для первого шага
                        return_components=False,
                        ground_truth_available=False
                    )

                    # Честный ground truth: level[t] из расширенной истории [t-350 .. t]
                    history_gt = df.iloc[t - window_size : t + 1].copy()
                    features_gt = self.prepare_features(history_gt, mode='batch')
                    level_t_true = features_gt.iloc[-1]['level']

                    # Масштабируем для точного сравнения
                    if self.feature_scalers is not None and 'Y' in self.feature_scalers:
                        level_t_true_scaled = self.feature_scalers['Y'].transform([[level_t_true]])[0, 0]
                    else:
                        level_t_true_scaled = level_t_true

                    # Сохранение результатов
                    timestamp = df.index[t] if hasattr(df.index, '__iter__') and len(df.index) > t else t
                    timestamps.append(timestamp)
                    true_values_original.append(level_t_true)
                    true_values_scaled.append(level_t_true_scaled)
                    pred_values_original.append(result['level_forecast'][0])
                    pred_values_scaled.append(result['level_forecast_scaled'][0])
                    pi_lower_original.append(result['level_ci_lower'][0])
                    pi_upper_original.append(result['level_ci_upper'][0])
                    pi_lower_scaled.append(result['level_ci_lower_scaled'][0])
                    pi_upper_scaled.append(result['level_ci_upper_scaled'][0])
                    volatility_levels.append(result['volatility_level'])
                    inflation_factors.append(result['inflation_factor'])
                    confidences.append(result['confidence'])  # ← СОГЛАСОВАННО С ОБУЧЕНИЕМ
                    regimes.append(result['regime'])

                except Exception as e:
                    print(f"🔴 Ошибка на шаге t={t}: {str(e)}")
                    raise

            # === 6. ПРЕОБРАЗОВАНИЕ В NUMPY ===
            true_values_original = np.array(true_values_original)
            true_values_scaled = np.array(true_values_scaled)
            pred_values_original = np.array(pred_values_original)
            pred_values_scaled = np.array(pred_values_scaled)
            pi_lower_original = np.array(pi_lower_original)
            pi_upper_original = np.array(pi_upper_original)
            pi_lower_scaled = np.array(pi_lower_scaled)
            pi_upper_scaled = np.array(pi_upper_scaled)
            volatility_levels = np.array(volatility_levels)
            inflation_factors = np.array(inflation_factors)
            confidences = np.array(confidences)
            regimes = np.array(regimes)
            timestamps = np.array(timestamps)

            # === 7. РАСЧЁТ МЕТРИК С СОГЛАСОВАННЫМ ИСТОЧНИКОМ ИСТИНЫ ===
            errors_original = pred_values_original - true_values_original

            metrics = {}
            # Базовые метрики
            metrics['MAE'] = float(np.mean(np.abs(errors_original)))
            metrics['RMSE'] = float(np.sqrt(np.mean(errors_original ** 2)))
            metrics['MAPE'] = float(np.mean(np.abs(errors_original / (np.abs(true_values_original) + 1e-8))))
            metrics['MAPE_median'] = float(np.median(np.abs(errors_original / (np.abs(true_values_original) + 1e-8))))

            ss_res = np.sum(errors_original ** 2)
            ss_tot = np.sum((true_values_original - np.mean(true_values_original)) ** 2)
            metrics['R2'] = float(1 - ss_res / (ss_tot + 1e-8))

            # 🔑 КРИТИЧЕСКОЕ ИСПРАВЛЕНИЕ: ЕДИНЫЙ РАСЧЁТ ШИРИНЫ ДИ ОТНОСИТЕЛЬНО ВОЛАТИЛЬНОСТИ ДАННЫХ
            # Как в train_step/val_step: width_ratio = mean(ci_width) / std(y_target_batch)
            valid_coverage_original = (true_values_original >= pi_lower_original) & (true_values_original <= pi_upper_original)
            metrics['CoverageRatio'] = float(np.mean(valid_coverage_original))
            metrics['CoverageCount'] = int(np.sum(valid_coverage_original))
            metrics['TotalCount'] = len(true_values_original)

            # ЕДИНСТВЕННЫЙ РАСЧЁТ ШИРИНЫ ДИ (согласованный с обучением)
            pi_widths_original = pi_upper_original - pi_lower_original
            y_std_batch = np.std(true_values_original) + 1e-8  # Волатильность ДАННЫХ
            metrics['MeanPIWidth'] = float(np.mean(pi_widths_original))
            metrics['MedianPIWidth'] = float(np.median(pi_widths_original))
            metrics['StdPIWidth'] = float(np.std(pi_widths_original))
            metrics['CIWidthVsStdDev'] = float(np.mean(pi_widths_original) / y_std_batch)  # ← ЕДИНСТВЕННЫЙ ИСТОЧНИК

            # Статистика по волатильности и режимам
            metrics['VolatilityMean'] = float(np.nanmean(volatility_levels))
            metrics['VolatilityStd'] = float(np.nanstd(volatility_levels))
            metrics['RegimeLowPct'] = float(np.mean(np.array(regimes) == 0) * 100)
            metrics['RegimeMidPct'] = float(np.mean(np.array(regimes) == 1) * 100)
            metrics['RegimeHighPct'] = float(np.mean(np.array(regimes) == 2) * 100)

            # 🔑 СОГЛАСОВАННАЯ ОШИБКА КАЛИБРОВКИ (как в обучении)
            target_coverage_mean = np.mean(confidences)  # ← ЕДИНСТВЕННЫЙ ИСТОЧНИК ИСТИНЫ
            metrics['TargetCoverage'] = float(target_coverage_mean)
            metrics['CalibrationError'] = abs(metrics['CoverageRatio'] - target_coverage_mean)

            # === 8. ВЫВОД МЕТРИК ===
            print("\n" + "=" * 60)
            print("📊 РЕЗУЛЬТАТЫ ЧЕСТНОЙ ОЦЕНКИ МОДЕЛИ")
            print("=" * 60)
            print(f"   📈 MAE: {metrics['MAE']:.6f}")
            print(f"   📉 RMSE: {metrics['RMSE']:.6f}")
            print(f"   📊 MAPE (среднее): {metrics['MAPE']:.4%}")
            print(f"   🎯 R²: {metrics['R2']:.6f}")
            print(f"   🎪 Покрытие PI: {metrics['CoverageRatio']:.2%} "
                  f"(цель: {metrics['TargetCoverage']:.2%}, ошибка: {metrics['CalibrationError']:.4f})")
            print(f"   📏 Ширина PI / std(данных): {metrics['CIWidthVsStdDev']:.2f}x "
                  f"(средняя ширина: {metrics['MeanPIWidth']:.6f})")
            print(f"   🔥 Режимы: LOW {metrics['RegimeLowPct']:.1f}% | MID {metrics['RegimeMidPct']:.1f}% | HIGH {metrics['RegimeHighPct']:.1f}%")
            print("=" * 60)

            # === 9. ВИЗУАЛИЗАЦИЯ ===
            if plot:
                print("\n📈 ПОСТРОЕНИЕ ГРАФИКОВ РЕЗУЛЬТАТОВ ОЦЕНКИ...")
                self._plot_evaluation_results(
                    true_vals=true_values_original,
                    pred_vals=pred_values_original,
                    pi_lower_vals=pi_lower_original,
                    pi_upper_vals=pi_upper_original,
                    volatility_vals=volatility_levels,
                    inflation_vals=inflation_factors,
                    confidence_vals=confidences,
                    errors=errors_original,
                    timestamps=timestamps,
                    metrics=metrics,
                    N=min(N, len(true_values_original)),
                    figsize=(14, 12)
                )

            return metrics

        finally:
            # === 10. ВОССТАНОВЛЕНИЕ ИСХОДНОГО СОСТОЯНИЯ ===
            if original_state_initialized_val:
                self._state_initialized.assign(True)
                self._last_state.assign(original_last_state_val)
                self._last_P.assign(original_last_P_val)
                self._step_counter.assign(original_step_counter_val)
                print(f"\n✅ Состояние фильтра восстановлено в исходное (как до оценки)")
            else:
                self._state_initialized.assign(False)
                print(f"\n✅ Состояние фильтра оставлено сброшенным (не было инициализировано до оценки)")

    def _setup_xaxis(self, ax, indices, timestamps, is_time_index):
        """Настройка оси X для графиков с поддержкой временных и числовых индексов"""
        if is_time_index:
            # Для временных индексов используем даты в формате YYYY-MM-DD
            num_ticks = min(5, len(indices))
            tick_positions = np.linspace(0, len(indices)-1, num_ticks, dtype=int)
            tick_labels = [timestamps[i].strftime('%Y-%m-%d') for i in tick_positions]
            ax.set_xticks(indices[tick_positions])
            ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
            ax.set_xlabel('Дата', fontsize=10)
        else:
            # Для числовых индексов используем равномерную разметку
            num_ticks = min(5, len(indices))
            tick_positions = np.linspace(0, len(indices)-1, num_ticks, dtype=int)
            tick_labels = [f"{int(indices[i])}" for i in tick_positions]
            ax.set_xticks(indices[tick_positions])
            ax.set_xticklabels(tick_labels, fontsize=9)
            ax.set_xlabel('Индекс наблюдения', fontsize=10)
        ax.tick_params(axis='both', which='major', labelsize=9)

    def _plot_evaluation_results(
        self,
        true_vals: np.ndarray,
        pred_vals: np.ndarray,
        pi_lower_vals: np.ndarray,
        pi_upper_vals: np.ndarray,
        volatility_vals: np.ndarray,
        inflation_vals: np.ndarray,
        confidence_vals: np.ndarray,
        errors: np.ndarray,
        timestamps: np.ndarray,
        metrics: Dict[str, float],
        N: int = 300,
        figsize: tuple = (11, 9)
    ) -> None:
        """
        Визуализация результатов оценки модели с контекстной волатильностью
        """
        # Автоматическое определение типа индекса
        sample_idx = timestamps[0] if len(timestamps) > 0 else None
        is_time_index = isinstance(sample_idx, (pd.Timestamp, datetime.datetime, np.datetime64, datetime.date))

        # Подготовка данных для отображения последних N точек
        start_idx = max(0, len(true_vals) - N)
        plot_indices = np.arange(len(timestamps))[start_idx:]  # Всегда числовые индексы для оси X

        # Данные для графиков
        true_plot = true_vals[start_idx:]
        pred_plot = pred_vals[start_idx:]
        pi_lower_plot = pi_lower_vals[start_idx:]
        pi_upper_plot = pi_upper_vals[start_idx:]
        volatility_plot = volatility_vals[start_idx:]
        inflation_plot = inflation_vals[start_idx:]
        confidence_plot = confidence_vals[start_idx:]
        errors_plot = errors[start_idx:]

        # Создание фигуры и сетки графиков (увеличенное вертикальное расстояние)
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 2, hspace=0.90, wspace=0.25)  # Увеличено hspace для большего расстояния

        # 1. Основной график: Истинные значения и прогнозы (на всю ширину)
        ax1 = fig.add_subplot(gs[0, :])

        # Расчет границ с учетом ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ + отступ
        y_min = min(true_plot.min(), pred_plot.min(), pi_lower_plot.min())
        y_max = max(true_plot.max(), pred_plot.max(), pi_upper_plot.max())

        # Если диапазон слишком мал, добавить минимальный отступ
        if y_max - y_min < 1e-10:
            y_min -= 0.1
            y_max += 0.1

        # Установка пределов ОТРАЗУ ПОСЛЕ создания оси, ПЕРЕД отрисовкой
        y_padding = 0.10 * (y_max - y_min)  # 10% отступ для видимости CI
        ax1.set_ylim(y_min - y_padding, y_max + y_padding)

        # Нарисовать данные
        ax1.plot(plot_indices, true_plot, 'b-', linewidth=2.5, label='Истинное значение', alpha=0.9)
        ax1.plot(plot_indices, pred_plot, 'r--', linewidth=2.5, label='Прогноз', alpha=0.9)
        ax1.fill_between(plot_indices, pi_lower_plot, pi_upper_plot, color='gold', alpha=0.3,
                         label=f"90% доверительный интервал (ширина: {metrics['MeanPIWidth']:.4f})")

        # ЗАКРЕПИТЬ пределы ПОСЛЕ отрисовки
        ax1.set_ylim(y_min - y_padding, y_max + y_padding)

        # Настройка оси X для основного графика
        if is_time_index:
            time_labels = timestamps[start_idx:]
            tick_positions = np.linspace(0, len(plot_indices) - 1, 5, dtype=int)
            tick_labels = [time_labels[i].strftime('%Y-%m-%d') for i in tick_positions]
            ax1.set_xticks(plot_indices[tick_positions])
            ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
            ax1.set_xlabel('Дата', fontsize=12)
        else:
            tick_positions = np.linspace(0, len(plot_indices) - 1, 5, dtype=int)
            tick_labels = [f"{int(plot_indices[i])}" for i in tick_positions]
            ax1.set_xticks(plot_indices[tick_positions])
            ax1.set_xticklabels(tick_labels)
            ax1.set_xlabel('Индекс наблюдения', fontsize=12)

        ax1.set_ylabel('Значение', fontsize=12)
        ax1.set_title('Прогноз vs Истинные значения с доверительным интервалом', fontsize=9, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # 2. График ошибок (на всю ширину под основным графиком)
        ax2 = fig.add_subplot(gs[1, :])
        ax2.plot(plot_indices, errors_plot, 'g-', linewidth=1.5, alpha=0.9)
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.fill_between(plot_indices, np.zeros_like(errors_plot), errors_plot,
                         where=(errors_plot > 0), color='red', alpha=0.3, label='Завышенный прогноз')
        ax2.fill_between(plot_indices, np.zeros_like(errors_plot), errors_plot,
                         where=(errors_plot < 0), color='blue', alpha=0.3, label='Заниженный прогноз')

        # Настройка оси X для графика ошибок
        self._setup_xaxis(ax2, plot_indices, timestamps[start_idx:], is_time_index)
        ax2.set_ylabel('Ошибка прогноза', fontsize=12)
        ax2.set_title(f'Ошибки прогноза (MAE: {metrics["MAE"]:.4f}, RMSE: {metrics["RMSE"]:.4f})',
                     fontsize=7, fontweight='bold')  # Уменьшен шрифт заголовка
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, linestyle='--', alpha=0.7)

        # 3. График уровня волатильности
        ax3 = fig.add_subplot(gs[2, 0])
        ax3.plot(plot_indices, volatility_plot, 'b-', linewidth=2, label='Уровень волатильности', alpha=0.9)

        # Горизонтальные линии для зон волатильности
        ax3.axhline(y=0.3, color='g', linestyle='--', alpha=0.5, label='Низкая волатильность (<0.3)')
        ax3.axhline(y=0.7, color='y', linestyle='--', alpha=0.5, label='Высокая волатильность (>0.7)')
        ax3.fill_between(plot_indices, 0, 0.3, color='green', alpha=0.15)
        ax3.fill_between(plot_indices, 0.7, 1.0, color='yellow', alpha=0.15)

        # Настройка оси X
        self._setup_xaxis(ax3, plot_indices, timestamps[start_idx:], is_time_index)
        ax3.set_ylim(0, 1.05)
        ax3.set_ylabel('Уровень волатильности', fontsize=11)
        ax3.set_title(f'Контекстная волатильность (среднее: {metrics["VolatilityMean"]:.4f})',
                     fontsize=7, fontweight='bold')  # Уменьшен шрифт заголовка
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, linestyle='--', alpha=0.7)

        # 4. График adaptive inflation
        ax4 = fig.add_subplot(gs[2, 1])
        ax4.plot(plot_indices, inflation_plot, 'm-', linewidth=2, label='Adaptive inflation factor', alpha=0.9)
        ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Базовый уровень (1.0)')

        # Настройка оси X
        self._setup_xaxis(ax4, plot_indices, timestamps[start_idx:], is_time_index)
        ax4.set_ylim(0.95, max(1.5, np.max(inflation_plot) * 1.1))
        ax4.set_ylabel('Inflation factor', fontsize=11)
        ax4.set_title(f'Adaptive inflation (среднее: {metrics["InflationMean"]:.4f})',
                     fontsize=7, fontweight='bold')  # Уменьшен шрифт заголовка
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, linestyle='--', alpha=0.7)

        # 5. График покрытия доверительного интервала
        ax5 = fig.add_subplot(gs[3, 0])
        coverage = (true_vals[start_idx:] >= pi_lower_plot) & (true_vals[start_idx:] <= pi_upper_plot)
        coverage_cum = np.cumsum(coverage) / np.arange(1, len(coverage) + 1)
        ax5.plot(plot_indices, coverage_cum, 'c-', linewidth=2.5, label='Текущее покрытие', alpha=0.9)
        ax5.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='Целевое покрытие (90%)')
        ax5.axhline(y=metrics['CoverageRatio'], color='g', linestyle='-', alpha=0.7,
                   label=f'Фактическое покрытие ({metrics["CoverageRatio"]:.2%})')

        # Настройка оси X
        self._setup_xaxis(ax5, plot_indices, timestamps[start_idx:], is_time_index)
        ax5.set_ylim(0, 1.05)
        ax5.set_ylabel('Покрытие ДИ', fontsize=11)
        ax5.set_title(f'Кумулятивное покрытие 90% ДИ\n(Ошибка калибровки: {metrics["CalibrationError"]:.4f})',
                     fontsize=7, fontweight='bold')  # Уменьшен шрифт заголовка
        ax5.legend(loc='best', fontsize=9)
        ax5.grid(True, linestyle='--', alpha=0.7)

        # 6. График уровня уверенности
        ax6 = fig.add_subplot(gs[3, 1])
        ax6.plot(plot_indices, confidence_plot, 'orange', linewidth=2, label='Уровень уверенности', alpha=0.9)
        ax6.axhline(y=np.mean(confidence_plot), color='b', linestyle='--', alpha=0.7,
                   label=f'Среднее: {np.mean(confidence_plot):.4f}')

        # Настройка оси X
        self._setup_xaxis(ax6, plot_indices, timestamps[start_idx:], is_time_index)
        ax6.set_ylim(0, 1.05)
        ax6.set_ylabel('Уровень уверенности', fontsize=11)
        ax6.set_title(f'Уровень уверенности прогнозов\n(среднее: {metrics["ConfidenceMean"]:.4f})',
                     fontsize=7, fontweight='bold')  # Уменьшен шрифт заголовка
        ax6.legend(loc='best', fontsize=9)
        ax6.grid(True, linestyle='--', alpha=0.7)

        # Сохранение графика
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'evaluation_results_{timestamp_str}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nГрафик результатов оценки сохранен: {filename}")
        plt.show()
