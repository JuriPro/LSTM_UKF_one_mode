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


class VolatilityRegimeSelector(tf.Module):
    """
    Мягкое распределение по режимам волатильности (LOW/MID/HIGH).
    Адаптивная калибровка масштабов доверительных интервалов.
    """

    def __init__(self,
                 num_regimes: int = 3,
                 history_window: int = 100,
                 learnable_centers: bool = True,
                 name: str = None):
        """
        Args:
            num_regimes: количество режимов (3: LOW, MID, HIGH)
            history_window: размер окна истории для расчета квантилей
            learnable_centers: использовать ли learnable центроиды режимов
            name: имя модуля
        """
        super().__init__(name=name)
        self.num_regimes = num_regimes
        self.history_window = history_window
        self.learnable_centers = learnable_centers

        # Learnable параметры для каждого режима
        # regime_scales[i] = масштаб CI для режима i
        # ИСПРАВЛЕНО: УВЕЛИЧЕННЫЕ ЦЕЛЕВЫЕ ЗНАЧЕНИЯ для режимов волатильности (увеличено на 30% для компенсации низкого покрытия)
        # LOW: 1.56 × (-ln(1-0.85)) = 1.56 × 1.895 = 2.96
        # MID: 2.34 × 1.895 = 4.44
        # HIGH: 3.25 × 1.895 = 6.16
        self.regime_scales = tf.Variable(
            tf.constant([2.96, 4.44, 6.16], dtype=tf.float32),  # LOW, MID, HIGH - увеличены на ~30%
            trainable=True,
            name='regime_scales',
            constraint=lambda x: tf.clip_by_value(x, 1.0, 8.0)  # Расширенный диапазон для адаптации
        )
        # Learnable центроиды для мягкого распределения
        if self.learnable_centers:
            # Инициализация на основе ожидаемых квантилей волатильности
            # LOW режим: 0.1 (низкая волатильность)
            # MID режим: 0.3 (средняя волатильность)
            # HIGH режим: 0.6 (высокая волатильность)
            initial_centers = np.log([0.1, 0.3, 0.6])
            self.center_logits = tf.Variable(
                tf.constant(initial_centers, dtype=tf.float32),
                trainable=True,
                name='center_logits'
            )
        else:
            self.centers = tf.constant([0.1, 0.3, 0.6], dtype=tf.float32)
        # Температура для softmax (управляет "остротой" распределения)
        # ИСПРАВЛЕНО: Начальное значение 0.4 для более резкого начального распределения
        self.temperature = tf.Variable(
            0.4,
            trainable=True,
            name='regime_temperature',
            constraint=lambda x: tf.clip_by_value(x, 0.1, 1.5)  # сужаем диапазон
        )

        # История волатильности для расчета статистики
        self._vol_history = tf.Variable(
            tf.zeros([1, self.history_window], dtype=tf.float32),
            trainable=False,
            name='vol_history'
        )

    def update_history(self, vol_current: tf.Tensor) -> None:
        """
        Обновить историю волатильности.

        Args:
            vol_current: [B] текущая волатильность
        """
        # vol_current: [B] → усредняем по батчу
        vol_mean = tf.reduce_mean(vol_current)  # скаляр

        # Сдвиг истории: убираем первый элемент, добавляем новый в конец
        new_history = tf.concat([
            self._vol_history[:, 1:],  # все кроме первого
            tf.reshape(vol_mean, [1, 1])  # новое значение
        ], axis=1)

        self._vol_history.assign(new_history)

    @tf.function
    def get_centers(self) -> tf.Tensor:
        """Графобезопасная версия без условий на Python-атрибуты"""
        # ВСЕГДА используем center_logits, даже если learnable_centers=False
        # (при необучаемых центрах они просто не обновляются градиентами)
        base_centers = tf.nn.softplus(self.center_logits)  # [num_regimes]
        
        # Адаптация к истории (безопасная для графа)
        flat_history = tf.reshape(self._vol_history, [-1])
        valid_mask = flat_history > 0.0
        valid_count = tf.reduce_sum(tf.cast(valid_mask, tf.int32))
        
        # Безопасные квантили с заглушками
        q25 = tf.where(valid_count >= 3,
                      tf.reduce_min(flat_history + (1.0 - tf.cast(valid_mask, tf.float32)) * 1e6),
                      0.1)
        q50 = tf.where(valid_count >= 5,
                      tf.reduce_min(tf.abs(flat_history - tf.reduce_mean(flat_history)) +
                                    (1.0 - tf.cast(valid_mask, tf.float32)) * 1e6),
                      0.3)
        q75 = tf.where(valid_count >= 7,
                      tf.reduce_max(flat_history * tf.cast(valid_mask, tf.float32)),
                      0.6)
        
        adaptive_centers = tf.stack([q25 * 0.8, q50, q75 * 1.2])
        adaptation_weight = tf.minimum(0.9, tf.cast(valid_count, tf.float32) / 50.0)
        centers = (1.0 - adaptation_weight) * base_centers + adaptation_weight * adaptive_centers
        
        return tf.clip_by_value(centers, 0.01, 0.99)

    @tf.function
    def assign_soft_regimes(self, vol_current: tf.Tensor) -> Dict[str, tf.Tensor]:
        """
        Мягкое распределение текущей волатильности по режимам.

        Args:
            vol_current: [B] текущая нормализованная волатильность

        Returns:
            Dict с ключами:
                'soft_weights': [B, num_regimes] - мягкие веса для каждого режима
                'regime_assignment': [B] - индекс доминирующего режима
                'entropy': [B] - энтропия распределения (для мониторинга)
        """
        batch_size = tf.shape(vol_current)[0]
        centers = self.get_centers()  # [num_regimes]

        # Расстояния от текущей волатильности до центров каждого режима
        # vol_current: [B] → [B, 1]
        vol_expanded = tf.expand_dims(vol_current, axis=1)  # [B, 1]

        # Евклидовы расстояния: [B, num_regimes]
        distances = tf.abs(vol_expanded - tf.expand_dims(centers, axis=0))

        # Softmax с температурой (управляет мягкостью распределения)
        # Чем выше температура, тем более равномерное распределение
        soft_weights = tf.nn.softmax(-distances / (self.temperature + 1e-6), axis=1)
        # soft_weights: [B, num_regimes]

        # Доминирующий режим для каждого элемента батча
        regime_assignment = tf.argmax(soft_weights, axis=1)  # [B]

        # Энтропия распределения (для мониторинга)
        entropy = -tf.reduce_sum(
            soft_weights * tf.math.log(soft_weights + 1e-8),
            axis=1
        )  # [B]

        return {
            'soft_weights': soft_weights,      # [B, num_regimes]
            'regime_assignment': regime_assignment,  # [B]
            'entropy': entropy                 # [B]
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

        regime_scale = tf.matmul(soft_weights, tf.expand_dims(self.regime_scales, axis=1))
        # результат: [B, 1]

        return regime_scale  # [B, 1]

    def get_spectrum_info(self) -> Dict[str, tf.Tensor]:
        """Информация о параметрах режимов для мониторинга"""
        return {
            'regime_scales': self.regime_scales,
            'centers': self.get_centers(),
            'temperature': self.temperature
        }


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
    def predict(self, x, Q, relax_factor, alpha_t, kappa_t):
        """
        JIT-совместимый этап предсказания UKF для скалярного состояния (state_dim=1).
        Полностью векторизованная версия для обработки батчей.
        """
        batch_size = tf.shape(x)[0]
        # ✅ Гарантируем правильные размерности входных данных
        x = tf.ensure_shape(x, [None, 1])
        Q = tf.ensure_shape(Q, [None, 1, 1])
        relax_factor = tf.reshape(relax_factor, [batch_size])
        alpha_t = tf.reshape(alpha_t, [batch_size])
        kappa_t = tf.reshape(kappa_t, [batch_size])

        # ✅ Прямое извлечение значений без лишних reshape
        x_scalar = tf.squeeze(x, axis=-1)  # [B]
        Q_scalar = tf.squeeze(Q, axis=[-2, -1])  # [B]

        # ✅ Параметры UKF для каждого элемента батча (векторизовано)
        n = 1
        lam = alpha_t**2 * (n + kappa_t) - n
        lam = tf.maximum(lam, 1e-3)

        # ✅ Вычисление масштаба для сигма-точек
        base_scale = tf.sqrt(n + lam)  # [B]
        scale = base_scale * relax_factor  # [B]
        P_sqrt_val = tf.sqrt(tf.maximum(Q_scalar, 1e-8))  # [B]
        scaled_offset = P_sqrt_val * scale  # [B]

        # ✅ Генерация сигма-точек для каждого элемента батча
        sigma_points = tf.stack([
            x_scalar,
            x_scalar + scaled_offset,
            x_scalar - scaled_offset
        ], axis=1)  # [B, 3]

        # ✅ ВЕСА ДЛЯ КАЖДОГО ЭЛЕМЕНТА БАТЧА (критически важно!)
        Wm_0 = lam / (n + lam)  # [B]
        Wm_i = 1.0 / (2.0 * (n + lam))  # [B]

        # ✅ Формирование весов в правильной размерности [B, 3]
        Wm = tf.stack([Wm_0, Wm_i, Wm_i], axis=1)  # [B, 3]

        # ✅ Веса для ковариации (с beta=2.0 по умолчанию)
        Wc_0 = Wm_0 + (1.0 - alpha_t**2 + 2.0)  # [B]
        Wc = tf.stack([Wc_0, Wm_i, Wm_i], axis=1)  # [B, 3]

        # ✅ Предсказание состояния (векторизовано)
        x_pred = tf.reduce_sum(Wm * sigma_points, axis=1, keepdims=True)  # [B, 1]

        # ✅ Предсказание ковариации (векторизовано)
        diff = sigma_points - tf.squeeze(x_pred, axis=-1)[:, tf.newaxis]  # [B, 3]
        weighted_var = Wc * tf.square(diff)  # [B, 3]

        # ✅ ИСПРАВЛЕНО: УДАЛЕНА СТРОКА С ИСПОЛЬЗОВАНИЕМ inflation_factor
        # Q_scalar_inflated = Q_scalar * inflation_factor  # ← УДАЛЕНО, инфляция уже применена
        P_pred_scalar = tf.reduce_sum(weighted_var, axis=1) + Q_scalar  # [B]

        # ✅ Добавляем синхронизацию с ограничениями из SpectralCovarianceParam
        min_P = 0.01  # Синхронизировано с _min_eigenvalue
        max_P = 8.0   # Синхронизировано с _max_eigenvalue

        # ✅ Финальная форма ковариации
        P_pred = tf.reshape(P_pred_scalar, [batch_size, 1, 1])  # [B, 1, 1]

        # ✅ СИНХРОНИЗИРОВАННОЕ ОГРАНИЧЕНИЕ
        P_pred = tf.clip_by_value(P_pred, min_P, max_P)  # ← ПРАВИЛЬНОЕ ОГРАНИЧЕНИЕ

        # ✅ Явное указание размерностей для JIT
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
        self.debug_mode = False

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
        target_coverage = 0.90
        confidence_factor = -np.log(1.0 - target_coverage)  # ≈ 2.30259
        # ИСПРАВЛЕНО: Увеличиваем начальные значения regime_scales для компенсации низкого покрытия
        self.regime_selector.regime_scales.assign(tf.constant(
            [0.95 * confidence_factor, 1.35 * confidence_factor, 1.85 * confidence_factor],  # Уменьшены, так как остальные параметры увеличили ширину
            dtype=tf.float32
        ))
        # Центроиды режимов адаптированы под реальную статистику волатильности
        self.regime_selector.center_logits.assign(tf.constant(
        np.log([0.12, 0.35, 0.75]),  # LOW, MID, HIGH - расширяем диапазон
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

            # Обучаемые параметры для базовых шумов С ОГРАНИЧЕНИЯМИ
            self.base_q_logit = tf.Variable(
                tf.math.log(0.15),  # СНИЖЕНО для меньшего Q (было 0.869)
                trainable=True,
                dtype=tf.float32,
                constraint=lambda x: tf.clip_by_value(x, tf.math.log(0.1), tf.math.log(1.0)),  # ← ДОБАВЛЕНО CONSTRAINT
                name='base_q_logit'
            )
            self.base_r_logit = tf.Variable(
                tf.math.log(1.8),  # УВЕЛИЧЕНО для большего R (было 0.499)
                trainable=True,
                dtype=tf.float32,
                constraint=lambda x: tf.clip_by_value(x, tf.math.log(1.0), tf.math.log(2.5)),  # ← ДОБАВЛЕНО CONSTRAINT
                name='base_r_logit'
            )

            # Вычисление начальной ковариации — ЕДИНАЯ ФОРМА [1, state_dim, state_dim]
            base_q = tf.nn.softplus(self.base_q_logit) + 1e-6
            # Явное преобразование в трехмерную форму [1, 1, 1]
            initial_P = tf.reshape(
                tf.eye(self.state_dim, dtype=tf.float32) * base_q,
                [1, self.state_dim, self.state_dim]
            )
            self._last_P = tf.Variable(
                initial_P,
                trainable=False,
                name='last_P',
                dtype=tf.float32
            )
            print(f"✅ _last_P: shape={self._last_P.shape} (единая форма [1, 1, 1])")

            # Флаг инициализации состояния
            self._state_initialized = tf.Variable(
                False,
                trainable=False,
                name='state_initialized',
                dtype=tf.bool
            )
            print("✅ _state_initialized: флаг инициализации")

            # Счетчик шагов
            self._step_counter = tf.Variable(
                0,
                trainable=False,
                name='step_counter',
                dtype=tf.int64
            )
            print("✅ _step_counter: для отслеживания количества шагов")

            # Время последней аномалии (для adaptive inflation)
            self._last_anomaly_time = tf.Variable(
                -100,
                trainable=False,
                name='last_anomaly_time',
                dtype=tf.int64
            )
            print("✅ _last_anomaly_time: для отслеживания времени последней аномалии")

            # Параметры для адаптивной волатильности
            self.volatility_sensitivity = tf.Variable(
                1.0,
                trainable=True,
                dtype=tf.float32,
                name='volatility_sensitivity'
            )

            # Параметры для Student-t распределения
            self.student_t_base_dof = tf.Variable(
                tf.math.log(2.5),  # еще меньше степеней свободы = толще хвосты
                trainable=True,
                dtype=tf.float32,
                name='student_t_base_dof'
            )
            self.student_t_vol_sensitivity = tf.Variable(
                0.3,  # меньшая чувствительность к волатильности
                trainable=True,
                dtype=tf.float32,
                constraint=lambda x: tf.clip_by_value(x, 0.0, 0.7),
                name='student_t_vol_sensitivity'
            )

            # Параметры для adaptive inflation
            self.inflation_base_factor = tf.Variable(
                tf.math.log(0.05),  # СНИЖЕНО с 1.572 до 0.5 (базовое значение ~1.6 вместо ~4.8)
                trainable=True,
                dtype=tf.float32,
                constraint=lambda x: tf.clip_by_value(x, tf.math.log(0.01), tf.math.log(0.3)),  # ← ДОБАВЛЕНО CONSTRAINT
                name='inflation_base_factor'
            )
            self.inflation_vol_sensitivity = tf.Variable(
                0.2,  # снижение чувствительности к волатильности
                trainable=True,
                dtype=tf.float32,
                constraint=lambda x: tf.clip_by_value(x, 0.0, 0.5),
                name='inflation_vol_sensitivity'
            )
            self.inflation_decay_rate = tf.Variable(
                0.95,  # ускорение затухания инфляции
                trainable=True,
                dtype=tf.float32,
                constraint=lambda x: tf.clip_by_value(x, 0.9, 0.99),
                name='inflation_decay_rate'
            )

            # Параметры для калибровки доверительных интервалов
            self.confidence_base = tf.Variable(
                0.90,  # повышение целевого покрытия
                trainable=True,
                dtype=tf.float32,
                constraint=lambda x: tf.clip_by_value(x, 0.85, 0.95),
                name='confidence_base'
            )
            self.confidence_vol_sensitivity = tf.Variable(
                0.1,  # снижение чувствительности к волатильности
                trainable=True,
                dtype=tf.float32,
                constraint=lambda x: tf.clip_by_value(x, 0.0, 0.3),
                name='confidence_vol_sensitivity'
            )

            # === ДОБАВЛЕНО: ПАРАМЕТР ДЛЯ МАКСИМАЛЬНОЙ ШИРИНЫ ДОВЕРИТЕЛЬНОГО ИНТЕРВАЛА ===
            self.max_width_factor_logit = tf.Variable(
                tf.math.log(1.5),  # exp(ln(2.5)) + 1.0 = 3.5 - увеличено для компенсации низкого покрытия
                trainable=True,
                dtype=tf.float32,
                constraint=lambda x: tf.clip_by_value(x, tf.math.log(1.0), tf.math.log(4.0)),  # Расширенный диапазон
                name='max_width_factor_logit'
            )

        # Инициализация оптимизатора
        print("\n✅ Инициализация оптимизатора с Loss Scale...")
        base_optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, amsgrad=False)
        current_policy = tf.keras.mixed_precision.global_policy()
        if current_policy.name == 'mixed_float16':
            self._optimizer = tf.keras.mixed_precision.LossScaleOptimizer(base_optimizer)
            print("✅ Loss Scale Optimizer инициализирован для mixed precision")
        else:
            self._optimizer = base_optimizer
            print("✅ Стандартный оптимизатор инициализирован (без mixed precision)")

        # Ранняя остановка
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_weights_dict = None
        self.best_scalers = None
        self.patience_counter = 0

        # Группы признаков для масштабирования
        self.scale_groups = {
            'robust': ['level', 'velocity', 'acceleration', 'energy', 'st_comp_diff',
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
            0.02,  # увеличено до 2% от основной loss
            trainable=False,
            dtype=tf.float32,
            name='lambda_entropy'
        )
        print(f"✅ Энтропийный регуляризатор инициализирован с lambda={self.lambda_entropy.numpy():.4f}")

        # Инициализация EMA для адаптивного порога аномалий
        self.threshold_ema = tf.Variable(3.0, trainable=False, dtype=tf.float32)

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
        block_size: int = 200  # ← Добавлен параметр для согласованности с HonestDataPreparator
    ) -> Tuple[Dict, Dict, Dict]:
        """
        УМНЫЙ ИНТЕРФЕЙС: автоматически проверяет кэш и загружает/готовит данные.
        
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
            
            preparator = HonestDataPreparator(
                model=self,
                seq_len=self.seq_len,
                min_history_for_features=self.min_history_for_features,
                buffer_size=buffer_size,
                block_size=block_size,  # ← Согласованность параметров
                seed=42
            )
            # Передаём путь БЕЗ расширения — метод сам обработает оба варианта
            train_data, val_data, test_data = preparator.load_prepared_datasets(cache_path)
            
            # Сохраняем для внутреннего использования в fit()
            self._honest_preparation = {
                'train_data': train_data,
                'val_data': val_data,
                'test_data': test_data,
                'preparator': preparator,
                'cache_path': cache_path
            }
            
            print(f"✅ Данные успешно загружены из кэша: {cache_file_with_ext}")
            return train_data, val_data, test_data
        
        # === ШАГ 2: ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ ===
        if full_df is None:
            raise ValueError(
                "full_df обязателен при первом запуске (когда кэш не существует). "
                "При загрузке из кэша передайте только cache_path."
            )
        
        # === ШАГ 3: ПОЛНАЯ ПОДГОТОВКА ===
        print(f"⚠️  Кэш не найден или требуется пересчёт: {cache_file_with_ext}")
        print("   Запускаем полную честную подготовку данных (без утечки будущего)...")
        
        preparator = HonestDataPreparator(
            model=self,
            seq_len=self.seq_len,
            min_history_for_features=self.min_history_for_features,
            buffer_size=buffer_size,
            block_size=block_size,  # ← Согласованность параметров
            seed=42
        )
        
        train_data, val_data, test_data = preparator.prepare_datasets(
            df=full_df,
            save_path=cache_path,  # ← БЕЗ расширения — метод сам добавит .pkl
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            n_jobs=n_jobs,
            force_recompute=force_recompute
        )
        
        # Сохраняем для внутреннего использования
        self._honest_preparation = {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'preparator': preparator,
            'cache_path': cache_path
        }
        
        print(f"✅ Данные подготовлены и сохранены в кэш: {cache_file_with_ext}")
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
        h2 = tf.keras.layers.Dropout(0.3)(h2, training=training)

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
            'q_base': tf.nn.softplus(base_ukf_params[..., 0:1]) + 1e-6,  # базовый Q
            'r_base': tf.nn.softplus(base_ukf_params[..., 1:2]) + 1e-6,  # базовый R
            'relax_base': 0.8 + 0.7 * tf.nn.sigmoid(base_ukf_params[..., 2:3]),
            'alpha_base': 0.5 + 0.5 * tf.nn.sigmoid(base_ukf_params[..., 3:4]),
            'kappa_base': 1.0 + 1.5 * tf.nn.sigmoid(base_ukf_params[..., 4:5]),
            'q_sensitivity': tf.nn.softplus(base_ukf_params[..., 5:6]) + 0.1,
            'r_sensitivity': tf.nn.softplus(base_ukf_params[..., 6:7]) + 0.1,
            'relax_sensitivity': tf.nn.sigmoid(base_ukf_params[..., 7:8]),
            'alpha_sensitivity': tf.nn.sigmoid(base_ukf_params[..., 8:9]),
            'kappa_sensitivity': tf.nn.sigmoid(base_ukf_params[..., 9:10]),
            'q_floor': tf.nn.softplus(base_ukf_params[..., 10:11]) + 1e-8,
            'r_floor': tf.nn.softplus(base_ukf_params[..., 11:12]) + 1e-8
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
        """Student-t обновление UKF с коррекцией под толстые хвосты и асимметрию"""
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

        # === СТРОГИЕ ОГРАНИЧЕНИЯ STUDENT-T ПАРАМЕТРОВ ===
        # Степени свободы: 4.0-8.0 вместо 2.0-15.0
        dof_scalar = tf.clip_by_value(dof_scalar, 3.0, 15.0)  # [B] Уменьшенный диапазон (было 5.0-7.0)

        # ✅ Нормализуем инновацию в единицах сигмы
        sigma_total = tf.sqrt(P_pred_scalar + R_scalar + 1e-8)  # [B]
        normalized_innov = innov_scalar / sigma_total  # [B]

        # ✅ Фактор увеличения ковариации для толстых хвостов (БОЛЕЕ СТРОГИЙ)
        # dof_scalar ∈ [4, 8] → heavy_tail_factor ∈ [1.0, 1.6]
        heavy_tail_factor = 1.0 + (dof_scalar - 4.0) / 6.0  # [B]
        heavy_tail_factor = tf.clip_by_value(heavy_tail_factor, 1.0, 1.6)  # [B]

        # ✅ УЖЕСТВЛЕННОЕ обнаружение больших инноваций
        tail_adjustment = tf.nn.softplus(1.0 * (tf.abs(normalized_innov) - 2.0))  # [B]
        tail_adjustment = tf.clip_by_value(tail_adjustment, 0.0, 1.5)  # [B]

        # АДАПТИВНЫЕ ограничения на асимметрию с учетом волатильности
        vol_factor = 0.8 + 1.2 * volatility_level  # Увеличиваем асимметрию при высокой волатильности
        asymmetry_pos_bound = 0.85 * vol_factor  # [B] - расширяем диапазон до 0.7-1.4
        asymmetry_neg_bound = 1.15 * vol_factor  # [B]
        asymmetry_pos_scalar = tf.clip_by_value(asymmetry_pos_scalar, 0.7, 1.4)
        asymmetry_neg_scalar = tf.clip_by_value(asymmetry_neg_scalar, 0.7, 1.4)

        # ✅ Асимметричное взвешивание хвостов (БОЛЕЕ КОНСЕРВАТИВНОЕ)
        asymmetry_weight = tf.where(
            innov_scalar >= 0,
            asymmetry_pos_scalar,  # [B]
            asymmetry_neg_scalar   # [B]
        )

        # ✅ Финальная коррекция ковариации с УЖЕСТВЛЕННЫМИ ПАРАМЕТРАМИ
        correction_factor = 1.0 + (heavy_tail_factor * tail_adjustment * asymmetry_weight)  # [B]
        correction_factor = tf.clip_by_value(correction_factor, 1.0, 2.0)  # [B]
        P_upd_scalar = P_upd_scalar * correction_factor  # [B]
        P_upd_scalar = tf.maximum(P_upd_scalar, 1e-8)  # [B]

        # ===== ДОПОЛНИТЕЛЬНЫЕ СТРОГИЕ ОГРАНИЧЕНИЯ =====
        # АСИММЕТРИЯ: ограничиваем в очень узком диапазоне
        asymmetry_pos = tf.clip_by_value(asymmetry_pos, 0.8, 1.2)  # [B, 1]
        asymmetry_neg = tf.clip_by_value(asymmetry_neg, 0.8, 1.2)  # [B, 1]

        # Фактор хвостов (для внешних расчетов)
        tail_weight_pos = tf.ones([batch_size, 1]) * 1.0  # Фиксированный базовый вес
        tail_weight_neg = tf.ones([batch_size, 1]) * 1.0  # Фиксированный базовый вес

        # ===== КОНЕЦ СТРОГИХ ОГРАНИЧЕНИЙ =====

        # ✅ Форматирование выходных данных с правильными размерностями
        x_upd = tf.reshape(x_upd_scalar, [batch_size, 1])  # [B, 1]
        P_upd = tf.reshape(P_upd_scalar, [batch_size, 1, 1])  # [B, 1, 1]
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
        inflation_state_input: Optional[Dict[str, tf.Tensor]] = None
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
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

        # АДАПТИВНАЯ инициализация текущих состояний с учетом волатильности
        current_state = initial_state  # [B, state_dim]
        # Инициализация ковариации зависит от уровня волатильности в данных
        initial_vol = tf.math.reduce_std(y_level_batch[:, :10], axis=1) + 1e-6
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
        if self.debug_mode:
            # DEBUG
            tf.print("\n🔍 ИНИЦИАЛИЗАЦИЯ ОКНА ИННОВАЦИЙ:")
            tf.print("   • Форма окна:", tf.shape(innov_window_init))
            tf.print("   • Размер батча (B):", B)
            tf.print("   • Размер окна:", innov_window_size)

        def cond(t, state, cov, vol, innov_win, inf_factor, rem_steps, last_anom_time,
                 s_hist, i_hist, v_hist, f_hist, high_infl_steps):
            return t < T

        def body(t, state, cov, vol, innov_win, inf_factor, rem_steps, last_anom_time,
                 s_hist, i_hist, v_hist, f_hist, high_infl_steps):
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
                innov_prev,  # [B]
                leverage_strength_t,  # [B, 1]
                q_base_t, q_sensitivity_t, q_floor_t,
                r_base_t, r_sensitivity_t, r_floor_t,
                tf.reshape(new_volatility, [B_batch, 1])  # [B] → [B, 1]
            )  # → Q_t: [B, 1, 1], R_t: [B, 1, 1]

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
            # ЭТАП 7: СИММЕТРИЧНОЕ ПРИМЕНЕНИЕ ИНФЛЯЦИИ К Q И R
            # ════════════════════════════════════════════════════════════════

            # === СИММЕТРИЧНОЕ ПРИМЕНЕНИЕ ИНФЛЯЦИИ К R ===
            inflation_limit_val = tf.squeeze(inflation_limit_t, axis=-1)  # [B]
            inflation_factor_for_R = tf.reshape(inf_factor, [B_batch, 1, 1])  # [B] → [B, 1, 1]
            inflation_limit_val_reshape = tf.reshape(inflation_limit_val, [B_batch, 1, 1])  # [B] → [B, 1, 1]
            R_inflated = R_t * inflation_factor_for_R
            R_inflated = tf.clip_by_value(R_inflated, 1e-8, inflation_limit_val_reshape)  # [B, 1, 1]

            # === СИММЕТРИЧНОЕ ПРИМЕНЕНИЕ ИНФЛЯЦИИ К Q ===
            # Менее агрессивное ослабление для Q (0.03 вместо 0.05)
            time_penalty_Q = tf.exp(-0.03 * tf.cast(t, tf.float32))
            inflation_factor_for_Q = inflation_factor_for_R * (0.6 + 0.4 * time_penalty_Q)

            # Применяем инфляцию к Q с более мягкими ограничениями
            Q_inflated = Q_t * inflation_factor_for_Q  # [B, 1, 1]
            Q_inflated = tf.clip_by_value(Q_inflated, 1e-8, inflation_limit_val_reshape * 0.8)  # Ограничение 80% от максимума для R

            # ════════════════════════════════════════════════════════════════
            # ЭТАП 8: UKF ПРЕДСКАЗАНИЕ (PREDICT)
            # ════════════════════════════════════════════════════════════════

            x_pred, P_pred = self.diff_ukf_component.predict(
                state,  # [B, 1]
                Q_inflated,  # [B, 1, 1] ← ИСПРАВЛЕНО: используем инфлированный Q
                relax_factor=relax_factor,  # [B]
                alpha_t=alpha_t,            # [B]
                kappa_t=kappa_t             # [B]
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

            max_allowed_innov = 5.0 + 3.0 * volatility_level  # Вместо фиксированного 10.0

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
            base_confidence = tf.squeeze(self.confidence_base)  # [скаляр]
            confidence_adjusted = base_confidence * (0.85 + 0.15 * volatility_level)
            confidence_adjusted = tf.clip_by_value(confidence_adjusted, 0.75, 0.99)

            # Адаптивный порог с учетом confidence_base
            adaptive_threshold = (3.5 - 2.0 * confidence_adjusted) * (1.0 + 0.5 * volatility_level)
            adaptive_threshold = tf.clip_by_value(adaptive_threshold, 2.3, 3.8)

            # Нормализация инновации с использованием confidence_base
            confidence_factor = 1.0 + (1.0 - confidence_adjusted) * 2.0
            normalized_threshold = adaptive_threshold / confidence_factor

            is_anomaly_t = normalized_innov > normalized_threshold
            is_anomaly = tf.cast(is_anomaly_t, tf.float32) * detection_strength  # [B]

            if self.debug_mode:
                if t < 10:  # Первые 10 шагов
                    tf.print(f"=== Step {t} ===")
                    tf.print("adaptive_threshold min/max:",
                             tf.reduce_min(adaptive_threshold), "/",
                             tf.reduce_max(adaptive_threshold))
                    tf.print("normalized_innov min/max abs:",
                             tf.reduce_min(tf.abs(normalized_innov)), "/",
                             tf.reduce_max(tf.abs(normalized_innov)))
                    tf.print("sigmoid(vol):", tf.reduce_mean(volatility_level))
                    tf.print("is_anomaly %:",
                             tf.reduce_mean(tf.cast(is_anomaly, tf.float32)) * 100.0)

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
            # Если реальный уровень выше целевого, увеличиваем порог
            confidence_factor = 1.0 + tf.maximum(0.0, rolling_anomaly_mean - target_anomaly_rate) * 3.0

            # Корректируем порог детекции
            adaptive_threshold = adaptive_threshold * confidence_factor

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
                fallback_multiplier_new, is_missed_jump_flag_new = self._fallback_inflation_correction(
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

            # Объединяем детекцию: аномалия ИЛИ пропущенный скачок
            should_activate_combined = tf.cast(is_anomaly, tf.bool)

            if self.debug_mode:
                # DEBUG: Внутри adaptive_ukf_filter, после вычисления is_anomaly
                is_missed_jump_flag = tf.zeros([B_batch, 1], dtype=tf.float32)

                if t < 5 and tf.reduce_sum(tf.cast(is_anomaly, tf.int32)) > 0:
                    tf.print("\n🔍 ДИАГНОСТИКА ШАГА", t)
                    tf.print("• Нормализованная инновация:", normalized_innov[:5])
                    tf.print("• Текущий порог:", adaptive_threshold[:5])
                    tf.print("• Волатильность (сырая):", new_volatility[:5])
                    tf.print("• Волатильность (sigmoid):", volatility_level[:5])
                    tf.print("• Стандартное отклонение инноваций:", innov_std[:5])
                    tf.print("• is_anomaly:", is_anomaly[:5])
                    tf.print("• is_missed_jump_flag:", tf.squeeze(is_missed_jump_flag)[:5])
                    tf.print("• should_activate_combined:", should_activate_combined[:5])

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
            # ЭТАП 13: ПРИМЕНЕНИЕ ИНФЛЯЦИИ К КОВАРИАЦИИ ИЗМЕРЕНИЙ И ПРОЦЕССА
            # ════════════════════════════════════════════════════════════════

            # === СИММЕТРИЧНОЕ ПРИМЕНЕНИЕ ИНФЛЯЦИИ К R ===
            R_inflated = R_t * inflation_factor_for_R  # [B, 1, 1]
            inflation_limit_val_reshape = tf.reshape(inflation_limit_val, [B_batch, 1, 1])  # [B] → [B, 1, 1]
            R_inflated = tf.clip_by_value(R_inflated, 1e-8, inflation_limit_val_reshape)  # [B, 1, 1]

            # === СИММЕТРИЧНОЕ ПРИМЕНЕНИЕ ИНФЛЯЦИИ К Q ===
            inflation_factor_for_Q = tf.reshape(inflation_factor_limited, [B_batch, 1, 1])  # [B] → [B, 1, 1]
            # Менее агрессивное ослабление для Q (0.03 вместо 0.05)
            time_penalty_Q = tf.exp(-0.03 * tf.cast(t, tf.float32))
            inflation_factor_for_Q = inflation_factor_for_Q * (0.6 + 0.4 * time_penalty_Q)

            # Применяем инфляцию к Q с более мягкими ограничениями
            Q_inflated = Q_t * inflation_factor_for_Q  # [B, 1, 1]
            Q_inflated = tf.clip_by_value(Q_inflated, 1e-8, inflation_limit_val_reshape * 0.8)  # Ограничение 80% от максимума для R

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
                high_infl_steps  # high_infl_steps_hist
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
            high_infl_steps_hist
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
        final_inflation_factor, final_remaining_steps, final_last_anom_time,  # исправлено имя
        s_hist, i_hist, v_hist, f_hist, high_infl_steps_hist) = final_vars

        # Восстановление формы выходных данных
        states_out = tf.transpose(s_hist.stack(), [1, 0])  # [T, B] → [B, T]
        innovations_out = tf.transpose(i_hist.stack(), [1, 0])  # [T, B] → [B, T]
        volatility_out = tf.transpose(v_hist.stack(), [1, 0])  # [T, B] → [B, T]
        inflation_factors_out = tf.transpose(f_hist.stack(), [1, 0])  # [T, B] → [B, T]

        result = (
            tf.expand_dims(states_out, axis=-1),  # [B, T] → [B, T, 1]
            tf.expand_dims(innovations_out, axis=-1),  # [B, T] → [B, T, 1]
            tf.expand_dims(volatility_out, axis=-1),  # [B, T] → [B, T, 1]
            tf.expand_dims(inflation_factors_out, axis=-1),  # [B, T] → [B, T, 1]
            final_state,  # [B, 1]
            final_covariance,  # [B, 1, 1]
        )

        return result

    @tf.function
    def compute_loss(self, predictions, targets, volatility_levels, inflation_factors,
                     ukf_params, calibration_loss, entropy_loss=0.0):
        """
        ✅ ИСПРАВЛЕННАЯ Функция потерь с правильными целями и штрафами

        Компоненты:
        1. MSE: базовая ошибка прогноза
        2. Calibration: покрытие CI, ширина, асимметрия
        3. Smoothness: плавность изменения параметров
        4. Stability: контроль инфляции
        5. Spectral: спектральная регуляризация UKF
        6. Additional: регуляризация всех обучаемых параметров
        7. Entropy: качество разделения режимов волатильности
        """

        # === 1. MSE LOSS ===
        mse_loss = tf.reduce_mean(tf.square(predictions - targets))

        # === 2. ПЛАВНОСТЬ ПАРАМЕТРОВ ===
        smoothness_loss = 0.0
        if hasattr(self, 'last_ukf_params') and self.last_ukf_params is not None:
            smoothness_loss = 0.001 * tf.reduce_mean(tf.square(ukf_params - self.last_ukf_params))

        # === 3. СТАБИЛЬНОСТЬ ИНФЛЯЦИИ ===
        # Штраф за отклонение от базового значения 1.0
        inflation_clipped = tf.clip_by_value(inflation_factors - 1.0, -5.0, 5.0)
        stability_penalty = 0.05 * tf.reduce_mean(tf.square(inflation_clipped))

        # === 4. СПЕКТРАЛЬНАЯ РЕГУЛЯРИЗАЦИЯ ===
        spectral_reg = 0.0
        if hasattr(self, 'diff_ukf_component') and self.diff_ukf_component is not None:
            spectrum = self.diff_ukf_component.get_spectrum_info()
            min_eig = spectrum['min_eigenvalue']
            # Штраф если собственное значение слишком близко к 0
            spectral_reg = 0.01 * tf.reduce_mean(tf.nn.relu(1e-4 - min_eig))

        # === 5. РЕГУЛЯРИЗАЦИЯ БАЗОВЫХ ПАРАМЕТРОВ ===
        additional_reg = (
            1e-4 * (tf.square(self.base_q_logit) + tf.square(self.base_r_logit)) +
            1e-4 * tf.square(self.volatility_sensitivity - 0.5) +
            1e-4 * tf.square(self.student_t_base_dof - tf.math.log(2.5)) +
            1e-4 * tf.square(self.student_t_vol_sensitivity) +
            1e-4 * tf.square(self.inflation_base_factor) +
            1e-4 * tf.square(self.inflation_vol_sensitivity) +
            1e-4 * tf.square(self.inflation_decay_rate - 0.95) +
            1e-4 * tf.square(self.confidence_base - 0.90) +
            1e-4 * tf.square(self.confidence_vol_sensitivity)
        )

        # === НОВОЕ: ШТРАФ НА МАКСИМАЛЬНУЮ ШИРИНУ ===
        # Удерживаем max_width_factor в целевом диапазоне 2.0-3.0
        if hasattr(self, 'max_width_factor_logit'):
            target_max_width = 2.5
            max_width_factor_value = tf.nn.softplus(self.max_width_factor_logit) + 1.0
            width_factor_penalty = 1.0 * tf.square(max_width_factor_value - target_max_width)
            additional_reg += width_factor_penalty
        else:
            width_factor_penalty = 0.0

        # === 6. ✅ ИСПРАВЛЕННАЯ РЕГУЛЯРИЗАЦИЯ VOLATILITY REGIME SELECTOR ===
        selector_reg = 0.0
        if hasattr(self, 'regime_selector') and self.regime_selector is not None:
            # ✅ ИСПРАВЛЕНО: Добавляем штраф за regime_scales
            regime_scales_penalty = 0.02 * tf.reduce_mean(
                tf.square(self.regime_selector.regime_scales - 2.5)  # 2.5 - целевое значение
            )
            selector_reg = regime_scales_penalty

            # УДАЛЕНЫ штрафы на целевые значения regime_scales и temperature
            # Они уже имеют constraints в инициализации и контролируют себя через механизм softmax

            # Оставляем только регуляризацию center_logits (если обучаемые)
            if (self.regime_selector.learnable_centers and
                hasattr(self.regime_selector, 'center_logits')):
                # Штраф на излишнее отклонение логитов от нуля
                selector_reg = selector_reg + 0.001 * tf.reduce_sum(
                    tf.square(self.regime_selector.center_logits)
                )

        # === 7. ✅ ИСПРАВЛЕННАЯ ЭНТРОПИЙНАЯ РЕГУЛЯРИЗАЦИЯ ===
        current_vol = tf.squeeze(volatility_levels[:, -1, :])  # [B] последний временной шаг
        regime_info = self.regime_selector.assign_soft_regimes(current_vol)
        current_entropy = tf.reduce_mean(regime_info['entropy'])

        # Максимальная энтропия для softmax(3 режимов)
        max_entropy = tf.math.log(3.0)  # ln(3) = 1.0986

        # Нормализуем энтропию в диапазон [0, 1]
        normalized_entropy = current_entropy / (max_entropy + 1e-8)

        # Целевая нормализованная энтропия = 0.75 (хороший баланс между спецификой и гибкостью)
        target_entropy_normalized = 0.75
        entropy_deviation = tf.abs(normalized_entropy - target_entropy_normalized)
        entropy_penalty = 2.0 * tf.square(entropy_deviation)  # Повышенный штраф

        # Дополнительный штраф за РАВНОМЕРНОЕ распределение (максимальная неопределенность)
        is_uniform_distribution = tf.cast(
            tf.abs(current_entropy - max_entropy) < 0.05,
            tf.float32
        )
        entropy_penalty += is_uniform_distribution * 1.0

        # Штраф за СЛИШКОМ ОСТРОЕ распределение (потеря гибкости адаптации)
        is_too_sharp = tf.cast(normalized_entropy < 0.3, tf.float32)
        entropy_penalty += is_too_sharp * 0.5

        # === ИТОГОВАЯ ПОТЕРЯ ===
        total_loss = (
            mse_loss +                          # Основная ошибка прогноза
            0.3 * calibration_loss +            # Контроль доверительных интервалов
            0.1 * smoothness_loss +             # Плавность параметров
            0.05 * stability_penalty +          # Стабильность инфляции
            0.01 * spectral_reg +               # Спектральная регуляризация
            additional_reg +                    # Регуляризация всех параметров
            selector_reg +                      # ✅ Исправленная регуляризация режимов
            entropy_penalty +                   # ✅ Исправленная энтропийная регуляризация
            self.lambda_entropy * entropy_loss * 2.0  # Энтропия скрытых состояний LSTM
        )

        # ✅ ИСПРАВЛЕНО: Явно суммируем все компоненты
        total_loss = total_loss + 0.5 * entropy_penalty

        # === ЗАЩИТА ОТ NaN/Inf ===
        total_loss = tf.where(
            tf.math.is_nan(total_loss),
            tf.constant(1e6, dtype=tf.float32),
            total_loss
        )

        # === СОХРАНЕНИЕ СОСТОЯНИЯ ДЛЯ СЛЕДУЮЩЕЙ ИТЕРАЦИИ ===
        self.last_ukf_params = ukf_params

        return total_loss

    @tf.function
    def train_step(self, X_batch, y_for_filtering_batch, y_target_batch,
                  initial_state, initial_covariance):
        """
        Шаг обучения для адаптивной UKF с контекстной волатильностью
    
        Args:
            X_batch: [B, T=72, n_features=20] - Технические признаки за 72 дня
            y_for_filtering_batch: [B, T=72] - Уровни для фильтрации дней 0-71
            y_target_batch: [B] - Целевой уровень на день 72 (t+1 после окна)
            initial_state: [B, state_dim=1] - Начальное состояние UKF
            initial_covariance: [B, state_dim=1, state_dim=1] - Начальная ковариация UKF
    
        Returns:
            loss: скаляр - Total loss (MSE + calibration + entropy)
            metrics: dict - Словарь с метриками обучения
            final_state: [B, 1] - Фильтрованное состояние на последний день окна (день 71)
            final_covariance: [B, 1, 1] - Ковариация на последний день окна
            forecast: [B] - Прогноз уровня на день 72 (t+1)
            std_dev: [B] - Стандартное отклонение прогноза
            volatility_levels: [B, T, 1] - Уровни волатильности для всех шагов
            regime_info: dict - Информация о распределении по режимам волатильности
            final_volatility: [B] - Финальный уровень волатильности
            entropy_stats: dict - Статистика энтропии скрытых состояний LSTM
            normalized_innovations: [B, 10, 1] - Нормализованные инновации последних 10 шагов
    
        Физический смысл:
            - Фильтруем дни 0-71, используя y_for_filtering_batch
            - Прогнозируем день 72 (t+1)
            - Сравниваем прогноз с y_target_batch (реальное значение дня 72)
        """
        B = tf.shape(X_batch)[0]
        with tf.device(self.device):
            with tf.GradientTape() as tape:
                # 1. FORWARD PASS LSTM
                lstm_outputs = self.model(X_batch, training=True)
                params_output = lstm_outputs['params']  # [B, T, 37]
                h_lstm2 = lstm_outputs['h_lstm2']       # [B, T, 128]
    
                # 2. ЭНТРОПИЙНАЯ РЕГУЛЯРИЗАЦИЯ
                entropy_loss = self.entropy_regularizer.compute_entropy_loss(h_lstm2)
    
                # 3. ОБРАБОТКА ВЫХОДОВ LSTM
                vol_context, ukf_params, inflation_config, student_t_config = self.process_lstm_output(params_output)
                
                # 4. АДАПТИВНАЯ ФИЛЬТРАЦИЯ UKF
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
                x_filtered = results[0]        # [B, T, 1]
                innovations = results[1]       # [B, T, 1]
                volatility_levels = results[2] # [B, T, 1]
                inflation_factors = results[3] # [B, T, 1]
                final_state = results[4]       # [B, 1]
                final_covariance = results[5]  # [B, 1, 1]
    
                # Сбор нормализованных инноваций для анализа
                normalized_innovations = tf.abs(innovations[:, -10:, :])  # последние 10 шагов
    
                # 5. ЯВНЫЙ PREDICT НА СЛЕДУЮЩИЙ ШАГ
                final_volatility = tf.squeeze(volatility_levels[:, -1, :])  # [B, T, 1] → [B]
                t_last = tf.shape(ukf_params['q_base'])[1] - 1
    
                # Явное извлечение параметров последнего шага
                q_base_final = tf.gather(ukf_params['q_base'], t_last, axis=1)  # [B, 1]
                q_sensitivity_final = tf.gather(ukf_params['q_sensitivity'], t_last, axis=1)  # [B, 1]
                q_floor_final = tf.gather(ukf_params['q_floor'], t_last, axis=1)  # [B, 1]
                r_base_final = tf.gather(ukf_params['r_base'], t_last, axis=1)  # [B, 1]
                r_sensitivity_final = tf.gather(ukf_params['r_sensitivity'], t_last, axis=1)  # [B, 1]
                r_floor_final = tf.gather(ukf_params['r_floor'], t_last, axis=1)  # [B, 1]
                relax_base_final = tf.gather(ukf_params['relax_base'], t_last, axis=1)  # [B, 1]
                relax_sensitivity_final = tf.gather(ukf_params['relax_sensitivity'], t_last, axis=1)  # [B, 1]
                alpha_base_final = tf.gather(ukf_params['alpha_base'], t_last, axis=1)  # [B, 1]
                alpha_sensitivity_final = tf.gather(ukf_params['alpha_sensitivity'], t_last, axis=1)  # [B, 1]
                kappa_base_final = tf.gather(ukf_params['kappa_base'], t_last, axis=1)  # [B, 1]
                kappa_sensitivity_final = tf.gather(ukf_params['kappa_sensitivity'], t_last, axis=1)  # [B, 1]
                inf_factor_final = tf.gather(inflation_factors[:, -1, :], t_last, axis=1)
    
                # Вычисление распределения по режимам волатильности
                regime_info = self.regime_selector.assign_soft_regimes(final_volatility)
    
                forecast, std_dev = self._explicit_predict_next_step(
                    final_state,
                    final_covariance,
                    final_volatility,
                    q_base_final, q_sensitivity_final, q_floor_final,
                    inf_factor_final,
                    relax_base_final, relax_sensitivity_final,
                    alpha_base_final, alpha_sensitivity_final,
                    kappa_base_final, kappa_sensitivity_final
                )
    
                # 6. РАСЧЕТ АДАПТИВНЫХ ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ
                # Параметры калибровки ДИ (85-95% покрытие в зависимости от волатильности)
                base_confidence = 0.88
                confidence_range = 0.15
                confidence_ceil = tf.fill(tf.shape(final_volatility), base_confidence + confidence_range/2)  # 0.955
                confidence_floor = tf.fill(tf.shape(final_volatility), base_confidence - confidence_range/2)  # 0.805
                target_coverage_adaptive = confidence_ceil - (confidence_ceil - confidence_floor) * final_volatility
    
                # Настройка параметров Student-t для асимметричной калибровки
                batch_shape = tf.shape(final_volatility)
                student_t_config_final = {
                    'confidence_ceil': tf.fill(batch_shape, 0.95),
                    'confidence_floor': tf.fill(batch_shape, 0.70),
                    'dof_base': tf.fill(batch_shape, 6.0),
                    'dof_sensitivity': tf.fill(batch_shape, 0.5),
                    'asymmetry_pos': tf.fill(batch_shape, 0.7)[:, tf.newaxis],
                    'asymmetry_neg': tf.fill(batch_shape, 1.3)[:, tf.newaxis],
                    'calibration_sensitivity': tf.fill(batch_shape, 1.0),
                    'tail_weight_pos': tf.fill(batch_shape, 0.8)[:, tf.newaxis],
                    'tail_weight_neg': tf.fill(batch_shape, 1.2)[:, tf.newaxis],
                    'confidence_base': tf.fill(batch_shape, 0.85),
                }
    
                # Адаптивные асимметричные границы доверительного интервала
                ci_lower, ci_upper, target_coverage = self._calibrate_confidence_interval(
                    forecast, std_dev, final_volatility, student_t_config_final,
                    innovations=innovations[:, -10:, :] if innovations is not None else None,
                    regime_assignment=regime_info['regime_assignment']
                )
    
                # Проверка валидности границ
                ci_min = tf.minimum(ci_lower, ci_upper)
                ci_max = tf.maximum(ci_lower, ci_upper)
    
                # Вычисление фактического покрытия
                y_target_flat = tf.reshape(y_target_batch, [-1])
                ci_min_flat = tf.reshape(ci_min, [-1])
                ci_max_flat = tf.reshape(ci_max, [-1])
                covered = tf.cast((y_target_flat >= ci_min_flat) & (y_target_flat <= ci_max_flat), tf.float32)
                actual_coverage = tf.reduce_mean(covered)
                target_coverage_mean = tf.reduce_mean(target_coverage)  # ЕДИНСТВЕННЫЙ источник истины
    
                # === ЕДИНСТВЕННЫЙ СОГЛАСОВАННЫЙ РАСЧЕТ ШИРИНЫ ДИ ===
                # Используем ВОЛАТИЛЬНОСТЬ ДАННЫХ (не неопределенность модели!) как объективную шкалу
                ci_width = ci_max - ci_min
                y_std_batch = tf.math.reduce_std(y_target_batch) + 1e-8  # Волатильность ДАННЫХ
                width_ratio = tf.reduce_mean(ci_width) / y_std_batch      # ЕДИНСТВЕННЫЙ расчет
    
                # === ШТРАФ ЗА НЕДОСТАТОЧНОЕ/ИЗБЫТОЧНОЕ ПОКРЫТИЕ ===
                # Используем ТО ЖЕ покрытие, что и для калибровки ДИ (согласованность!)
                coverage_diff = tf.abs(actual_coverage - target_coverage_mean)
                undercoverage_penalty = tf.cond(
                    actual_coverage < target_coverage_mean,
                    lambda: 5.0 * tf.square(target_coverage_mean - actual_coverage),  # Усиленный штраф за недостаточное покрытие
                    lambda: 2.0 * tf.square(actual_coverage - target_coverage_mean)   # Стандартный штраф за избыточное покрытие
                )
    
                # === ШТРАФ ЗА ШИРИНУ ИНТЕРВАЛОВ ===
                # Целевая ширина для 85-90% ДИ при нормальном распределении
                target_width_ratio = 3.5
                width_penalty = 2.0 * tf.square(tf.maximum(width_ratio - target_width_ratio, 0.0))
                width_penalty += 3.0 * tf.square(tf.maximum(target_width_ratio * 0.7 - width_ratio, 0.0))
    
                # === ШТРАФ ЗА АСИММЕТРИЧНОСТЬ ===
                asymmetry = (ci_max - forecast) / (forecast - ci_min + 1e-8)
                asymmetry_penalty = 0.5 * tf.reduce_mean(tf.square(tf.math.log(asymmetry + 1e-8)))
    
                calibration_loss = undercoverage_penalty + width_penalty + asymmetry_penalty
    
                # 7. РАСЧЕТ ИТОГОВОЙ ПОТЕРИ
                loss = self.compute_loss(
                    forecast,
                    y_target_batch,
                    volatility_levels,
                    inflation_factors,
                    ukf_params,
                    calibration_loss,
                    entropy_loss
                )
    
                # Нормализация потери по размеру батча и времени
                B_float = tf.cast(tf.shape(y_target_batch)[0], tf.float32)
                T_float = tf.cast(tf.shape(y_for_filtering_batch)[1], tf.float32)
                loss = loss / (B_float * T_float)
    
                # 8. СБОР ОБУЧАЕМЫХ ПЕРЕМЕННЫХ
                trainable_vars = []
                if self.model is not None:
                    trainable_vars.extend(self.model.trainable_variables)
                if self.use_diff_ukf and hasattr(self, 'diff_ukf_component'):
                    trainable_vars.extend(self.diff_ukf_component.trainable_variables)
    
                additional_vars = [
                    self.base_q_logit, self.base_r_logit,
                    self.volatility_sensitivity,
                    self.student_t_base_dof, self.student_t_vol_sensitivity,
                    self.inflation_base_factor, self.inflation_vol_sensitivity,
                    self.inflation_decay_rate,
                    self.confidence_base, self.confidence_vol_sensitivity
                ]
                for var in additional_vars:
                    if var is not None and isinstance(var, tf.Variable):
                        trainable_vars.append(var)
    
                if hasattr(self, 'regime_selector') and self.regime_selector is not None:
                    trainable_vars.extend([
                        self.regime_selector.regime_scales,
                        self.regime_selector.temperature,
                    ])
                    if self.regime_selector.learnable_centers and hasattr(self.regime_selector, 'center_logits'):
                        trainable_vars.append(self.regime_selector.center_logits)
    
                # 9. ВЫЧИСЛЕНИЕ И ПРИМЕНЕНИЕ ГРАДИЕНТОВ
                gradients = tape.gradient(loss, trainable_vars)
                clipped_grads, global_norm = tf.clip_by_global_norm(gradients, 1.0)
                self._optimizer.apply_gradients(zip(clipped_grads, trainable_vars))
    
                # 10. РАСЧЕТ МЕТРИК
                mse_loss = tf.reduce_mean(tf.square(forecast - y_target_batch))
                avg_volatility = tf.reduce_mean(final_volatility)
    
                # Статистика режимов волатильности
                regime_soft_weights = tf.reduce_mean(regime_info['soft_weights'], axis=0)  # [3]
                regime_entropy = tf.reduce_mean(regime_info['entropy'])  # скаляр
    
                # Q/R мониторинг
                q_current = tf.reduce_mean(q_base_final)
                r_current = tf.reduce_mean(r_base_final)
                qr_ratio = q_current / (r_current + 1e-8)
    
                # Статистика адаптивного inflation
                avg_inflation = tf.reduce_mean(inflation_factors[:, -1, :])
                dynamic_threshold, inflation_anomaly_ratio = compute_adaptive_threshold(
                    inflation_factors,
                    final_volatility,
                    self.threshold_ema,
                    target_anomaly_ratio=0.35
                )
    
                # Спектральная стабильность UKF
                spectrum_info = self.diff_ukf_component.get_spectrum_info()
                min_eigenvalue = spectrum_info['min_eigenvalue']
    
                # Сборка метрик (СОГЛАСОВАННАЯ МЕТРИКА ДЛЯ ОТЧЕТА)
                metrics = {
                    'total_loss': loss,
                    'mse_loss': mse_loss,
                    'entropy_loss': entropy_loss,
                    'avg_volatility': avg_volatility,
                    'avg_inflation': avg_inflation,
                    'global_norm': global_norm,
                    'qr_ratio': qr_ratio,
                    'q_value': q_current,
                    'r_value': r_current,
                    'regime_low_weight': regime_soft_weights[0],
                    'regime_mid_weight': regime_soft_weights[1],
                    'regime_high_weight': regime_soft_weights[2],
                    'regime_entropy': regime_entropy,
                    'inflation_anomaly_ratio': inflation_anomaly_ratio,
                    'ukf_min_eigenvalue': min_eigenvalue,
                    'coverage_ratio': actual_coverage,
                    'target_coverage': target_coverage_mean,  # Согласовано с калибровкой
                    'ci_width_vs_stddev': width_ratio,        # Согласовано с штрафом (через волатильность данных)
                    'calibration_error': tf.abs(actual_coverage - target_coverage_mean),
                }
    
                # 11. ДОПОЛНИТЕЛЬНАЯ ИНФОРМАЦИЯ ОБ ЭНТРОПИИ
                entropy_stats = self.entropy_regularizer.get_entropy_stats(h_lstm2)
    
                # 12. ОБНОВЛЕНИЕ ИСТОРИИ ВОЛАТИЛЬНОСТИ
                self.regime_selector.update_history(final_volatility)
    
                # 13. АДАПТАЦИЯ ПАРАМЕТРОВ РЕЖИМОВ ПРИ ВЫСОКОЙ ЭНТРОПИИ
                entropy_val = tf.reduce_mean(regime_info['entropy'])
                if entropy_val > 1.05:
                    new_temp = tf.maximum(0.4, self.regime_selector.temperature * 0.95)
                    self.regime_selector.temperature.assign(new_temp)
                    regime_info = self.regime_selector.assign_soft_regimes(final_volatility)
                    entropy_val = tf.reduce_mean(regime_info['entropy'])
    
                # ВЕРНЁМ РЕЗУЛЬТАТЫ
                return loss, metrics, final_state, final_covariance, forecast, std_dev, \
                       volatility_levels, regime_info, final_volatility, entropy_stats, normalized_innovations

    def _explicit_predict_next_step(self, final_state, final_covariance, current_volatility,
                                   q_base_final, q_sensitivity_final, q_floor_final,
                                   inf_factor,  # ← ДОБАВЛЕН ПАРАМЕТР ИНФЛЯЦИИ
                                   relax_base_final, relax_sensitivity_final,
                                   alpha_base_final, alpha_sensitivity_final,
                                   kappa_base_final, kappa_sensitivity_final):
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

        # Вычисление Q с инфляцией
        q_val = q_base_final * (1.0 + q_sensitivity_final * current_vol_scalar)  # [B]
        q_val = tf.maximum(q_val, q_floor_final)  # [B]
        inflation_factor_for_Q = tf.reshape(inf_factor, [batch_size, 1, 1])  # [B, 1, 1]
        Q_t = tf.reshape(q_val, [batch_size, 1, 1]) * inflation_factor_for_Q  # ← ПРИМЕНЕНИЕ ИНФЛЯЦИИ
        Q_t = tf.clip_by_value(Q_t, 1e-8, 5.0)  # жесткие ограничения

        # Параметры адаптации UKF
        relax_factor = relax_base_final * (1.0 + relax_sensitivity_final * current_vol_scalar)
        alpha_t = alpha_base_final * (1.0 + alpha_sensitivity_final * current_vol_scalar)
        kappa_t = kappa_base_final * (1.0 + kappa_sensitivity_final * current_vol_scalar)

        # PREDICT шаг
        x_pred, P_pred = self.diff_ukf_component.predict(
            final_state,
            Q_t,  # ← УЖЕ ИНФЛИРОВАННЫЙ Q
            relax_factor=relax_factor,
            alpha_t=alpha_t,
            kappa_t=kappa_t
        )

        # Прогнозируемая дисперсия
        forecast_var = P_pred[:, 0, 0] + Q_t[:, 0, 0]
        std_dev = tf.sqrt(tf.maximum(forecast_var, 1e-8))

        forecast_value = tf.squeeze(x_pred, axis=-1)
        std_dev_value = tf.squeeze(std_dev)

        return forecast_value, std_dev_value

    def _calibrate_confidence_interval(self, forecast, stddev, volatility_level, student_t_config, innovations=None, regime_assignment=None, true_values=None):
        """
        Калибрирует асимметричные доверительные интервалы с адаптацией к текущей волатильности.
        ИСПРАВЛЕННАЯ версия с учетом широкого диапазона значений и высокой волатильности.

        Args:
            forecast: [B] предсказанное значение
            stddev: [B] стандартное отклонение предсказания
            volatility_level: [B] текущий уровень волатильности (нормализованный)
            student_t_config: dict с параметрами Student-t распределения
            innovations: [B, window_size, 1] опционально, инновации для анализа асимметрии
            true_values: [B] опционально, истинные значения для дополнительной адаптации

        Returns:
            ci_lower: [B] нижняя граница ДИ
            ci_upper: [B] верхняя граница ДИ
            target_coverage: [B] целевое покрытие для каждого элемента батча
        """
        batch_size = tf.shape(forecast)[0]

        # ===== 1. НОРМАЛИЗАЦИЯ ВХОДНЫХ ДАННЫХ =====
        forecast = tf.squeeze(forecast)  # [B]
        stddev = tf.squeeze(stddev)  # [B]
        volatility_level = tf.squeeze(volatility_level)  # [B]

        # Гарантируем положительность stddev
        stddev = tf.maximum(stddev, 1e-8)

        # ===== 2. АДАПТИРОВАННЫЙ ПОД ШИРОКИЙ ДИАПАЗОН УРОВЕНЬ ДОВЕРИЯ =====
        base_confidence_ceil = 0.95  # Повышаем целевое покрытие для компенсации низкой точности
        base_confidence_floor = 0.80  # Увеличиваем минимум для более широких интервалов
        # Адаптивное изменение в зависимости от волатильности
        confidence_range = 0.15  # Диапазон покрытия 0.80-0.95
        # Используем более агрессивную адаптацию в HIGH режиме
        vol_adjustment = 0.05 * volatility_level  # Умеренная корректировка в зависимости от волатильности
        target_coverage = base_confidence_floor + confidence_range * vol_adjustment
        target_coverage = tf.clip_by_value(target_coverage, base_confidence_floor, base_confidence_ceil)  # [B]

        # ===== 3. БЕЗОПАСНОЕ ИЗВЛЕЧЕНИЕ ПАРАМЕТРОВ Student-t =====
        def safe_get_param(param_dict, key, default_value=0.5):
            """Безопасное извлечение параметра из словаря"""
            if key not in param_dict or param_dict[key] is None:
                return tf.ones([batch_size], dtype=tf.float32) * default_value
            param = param_dict[key]
            return tf.squeeze(param)

        # Используем параметры с толстыми хвостами для сигналов с широким диапазоном
        dof_base = safe_get_param(student_t_config, 'dof_base', 2.0)  # Еще меньше степеней свободы для толстых хвостов
        dof_sensitivity = safe_get_param(student_t_config, 'dof_sensitivity', 0.8)
        tail_weight_pos = safe_get_param(student_t_config, 'tail_weight_pos', 1.0)  # Более сбалансированные веса
        tail_weight_neg = safe_get_param(student_t_config, 'tail_weight_neg', 1.0)  # Более сбалансированные веса
        regime_scale = safe_get_param(student_t_config, 'regime_scale', 1.0)

        # ===== 4. АДАПТИВНОЕ ВЫЧИСЛЕНИЕ СТЕПЕНЕЙ СВОБОДЫ =====
        # Уменьшаем степени свободы для толстых хвостов
        dof_adjusted = dof_base + 2.0 * (1.0 - volatility_level) * dof_sensitivity
        dof_adjusted = tf.clip_by_value(dof_adjusted, 1.0, 8.0)  # [B] - еще более низкое минимальное значение

        # ===== 5. ВЫЧИСЛЕНИЕ Z-КВАНТИЛЕЙ ДЛЯ t-РАСПРЕДЕЛЕНИЯ =====
        # Используем аппроксимацию: t_α ≈ sqrt((df-1)/(df*(1-α)^2 - 1))

        # Нижний квантиль (для нижней границы)
        prob_lower = (1.0 - target_coverage) / 2.0  # [B]
        prob_lower = tf.maximum(prob_lower, 0.001)  # Избегаем деления на ноль

        denominator_lower = dof_adjusted * (prob_lower ** 2) - 1.0
        denominator_lower = tf.maximum(denominator_lower, 0.01)  # Численная стабильность

        z_lower_raw = -tf.sqrt((dof_adjusted - 1.0) / denominator_lower)
        z_lower = tf.clip_by_value(z_lower_raw, -8.0, -0.1)  # Расширяем диапазон для экстремальных значений

        # Верхний квантиль (для верхней границы)
        prob_upper = (1.0 - target_coverage) / 2.0  # [B]
        prob_upper = tf.maximum(prob_upper, 0.001)

        denominator_upper = dof_adjusted * (prob_upper ** 2) - 1.0
        denominator_upper = tf.maximum(denominator_upper, 0.01)

        z_upper_raw = tf.sqrt((dof_adjusted - 1.0) / denominator_upper)
        z_upper = tf.clip_by_value(z_upper_raw, 0.1, 8.0)   # Расширяем диапазон для экстремальных значений

        # ===== 6. АНАЛИЗ АСИММЕТРИИ ИННОВАЦИЙ (если доступны) =====
        center_shift = tf.zeros([batch_size], dtype=tf.float32)  # [B]

        if innovations is not None:
            innovations_processed = tf.squeeze(innovations, axis=-1)  # [B, window_size]
            actual_batch_size = tf.shape(innovations_processed)[0]

            # Проверка размера батча
            if actual_batch_size == batch_size:
                pos_mask = tf.cast(innovations_processed > 0, tf.float32)
                neg_mask = tf.cast(innovations_processed <= 0, tf.float32)

                abs_innov = tf.abs(innovations_processed)

                pos_sum = tf.reduce_sum(pos_mask * abs_innov, axis=1)
                pos_count = tf.reduce_sum(pos_mask, axis=1)
                pos_magnitude = pos_sum / tf.maximum(pos_count, 1.0)

                neg_sum = tf.reduce_sum(neg_mask * abs_innov, axis=1)
                neg_count = tf.reduce_sum(neg_mask, axis=1)
                neg_magnitude = neg_sum / tf.maximum(neg_count, 1.0)

                # Center shift = смещение центра прогноза на основе асимметрии
                # ОГРАНИЧИВАЕМ до ±0.5*stddev
                center_shift = tf.clip_by_value(
                    stddev * (pos_magnitude - neg_magnitude) / (pos_magnitude + neg_magnitude + 1e-8),
                    -0.5 * stddev,
                    0.5 * stddev
                )  # [B]

        # ===== 7. ВЫЧИСЛЕНИЕ МАРЖ С УЧЕТОМ ШИРОКОГО ДИАПАЗОНА И ВЫСОКОЙ ВОЛАТИЛЬНОСТИ =====
        # Увеличиваем базовые множители для компенсации низкого покрытия
        regime_scale_factor = tf.maximum(1.5, regime_scale)  # Увеличиваем минимальный масштаб
        max_width_factor = 15.0 + 10.0 * regime_scale_factor  # Максимум ~8-10x вместо 30+

        # Добавляем прямую коррекцию на основе stddev
        stddev_factor = 2.0 + 1.0 * (stddev / tf.reduce_mean(stddev + 1e-8))  # Учет относительной волатильности
        max_width_factor = max_width_factor * stddev_factor

        # Увеличиваем минимальные значения марж для компенсации низкого покрытия
        margin_lower = stddev * tf.clip_by_value(
            tf.abs(z_lower) * tail_weight_neg * regime_scale,
            2.0,  # Увеличиваем минимум с 1.0 до 2.0 для компенсации низкого покрытия
            max_width_factor
        )
        margin_upper = stddev * tf.clip_by_value(
            tf.abs(z_upper) * tail_weight_pos * regime_scale,
            2.0,  # Увеличиваем минимум с 1.0 до 2.0 для компенсации низкого покрытия
            max_width_factor
        )

        # ===== 8. УЛУЧШЕНИЕ АСИММЕТРИИ ДЛЯ ЭКСТРЕМАЛЬНЫХ ЗНАЧЕНИЙ =====
        # Увеличиваем чувствительность к экстремальным значениям
        extreme_vol_threshold = 0.4  # Снижаем порог для более ранней активации
        extreme_vol_mask = tf.cast(volatility_level > extreme_vol_threshold, tf.float32)
        # Усиливаем асимметрию для широкого диапазона данных
        lower_expansion_factor = 1.0 + 8.0 * extreme_vol_mask * volatility_level  # Увеличиваем с 5.0 до 8.0
        upper_expansion_factor = 1.0 + 5.0 * extreme_vol_mask * volatility_level  # Увеличиваем с 2.5 до 5.0

        # Добавляем прямую коррекцию на основе анализа инноваций
        if innovations is not None:
            # Анализ асимметрии инноваций
            innovations_processed = tf.squeeze(innovations, axis=-1)
            if tf.shape(innovations_processed)[0] == batch_size:
                # Вычисляем асимметрию инноваций
                positive_innov = tf.boolean_mask(innovations_processed, innovations_processed > 0)
                negative_innov = tf.boolean_mask(innovations_processed, innovations_processed <= 0)
                if tf.size(positive_innov) > 0 and tf.size(negative_innov) > 0:
                    pos_std = tf.math.reduce_std(positive_innov)
                    neg_std = tf.math.reduce_std(negative_innov)
                    asymmetry_ratio = pos_std / (neg_std + 1e-8)
                    # Усиливаем асимметрию для сигналов с широким диапазоном
                    asymmetry_factor = tf.where(
                        asymmetry_ratio > 1.5,
                        1.0 + 1.2 * (asymmetry_ratio - 1.5),  # Увеличиваем с 0.8 до 1.2
                        tf.where(
                            asymmetry_ratio < 0.67,
                            1.0 + 1.2 * (1.5 - 1.0/asymmetry_ratio),  # Увеличиваем с 0.8 до 1.2
                            1.0
                        )
                    )
                    # Применяем асимметричное усиление
                    margin_lower = margin_lower * asymmetry_factor
                    margin_upper = margin_upper * (1.0 / asymmetry_factor)

        # Применяем расширение для экстремальных значений
        margin_lower = margin_lower * lower_expansion_factor
        margin_upper = margin_upper * upper_expansion_factor

        # ===== 9. УЛУЧШЕННАЯ ПРЯМАЯ КОРРЕКТИРОВКА ПО ПОКРЫТИЮ =====
        # Учитываем широкий диапазон данных и высокую волатильность
        if hasattr(self, 'coverage_history') and hasattr(self.coverage_history, '__len__') and len(self.coverage_history) >= 50:
            recent_coverage = tf.stack(self.coverage_history[-50:])
            current_coverage = tf.reduce_mean(recent_coverage)
            target_coverage_mean = tf.reduce_mean(target_coverage)

            coverage_error = target_coverage_mean - current_coverage
            # Увеличиваем агрессивность корректировки для сигналов с высокой волатильностью
            adjustment_factor = tf.where(
                coverage_error > 0.10,  # Уменьшаем порог для более частой коррекции
                1.0 + 15.0 * coverage_error,  # Увеличиваем с 8.0 до 15.0
                tf.where(
                    coverage_error > 0.05,  # Уменьшаем порог для более частой коррекции
                    1.0 + 10.0 * coverage_error,  # Увеличиваем с 5.0 до 10.0
                    1.0 + 0.5 * coverage_error  # Увеличиваем минимальную коррекцию
                )
            )

            # Дополнительная коррекция для сигналов с высокой волатильностью
            volatility_correction = 1.0 + 3.0 * volatility_level  # Увеличиваем с 1.5 до 3.0
            adjustment_factor = adjustment_factor * volatility_correction

            # Применяем корректировку с учетом асимметрии
            margin_lower = margin_lower * adjustment_factor
            margin_upper = margin_upper * adjustment_factor

            # Дополнительная асимметричная коррекция
            if coverage_error > 0.05:  # Уменьшаем порог для более частой коррекции
                asymmetry_factor = 1.0 + 2.5 * coverage_error * volatility_level  # Увеличиваем с 1.5 до 2.5
                margin_lower = margin_lower * asymmetry_factor  # Больше расширяем нижнюю границу

        # ===== 10. ВЫЧИСЛЕНИЕ ГРАНИЦ =====
        ci_lower = forecast - margin_lower + center_shift  # [B]
        ci_upper = forecast + margin_upper + center_shift  # [B]

        # ===== 11. ПРОВЕРКА И КОРРЕКЦИЯ ВАЛИДНОСТИ =====
        ci_min = tf.minimum(ci_lower, ci_upper)
        ci_max = tf.maximum(ci_lower, ci_upper)

        # ===== УЛУЧШЕННАЯ КОРРЕКЦИЯ ДЛЯ ЭКСТРЕМАЛЬНЫХ ЗНАЧЕНИЙ =====
        # Учитываем широкий диапазон данных
        current_volatility_level = tf.squeeze(volatility_level)
        expansion_factor = 1.0 + 2.0 * current_volatility_level  # Увеличиваем влияние волатильности

        # Для сигналов с широким диапазоном увеличиваем базовую ширину
        base_expansion = 5.0 + 3.0 * (stddev / tf.reduce_mean(stddev + 1e-8))  # Адаптация к относительной волатильности
        expansion_factor = expansion_factor * base_expansion

        needs_expansion = tf.logical_or(
            forecast < ci_min,
            forecast > ci_max
        )

        # Увеличиваем минимальную ширину интервалов
        min_width = 5.0 * stddev  # Увеличиваем базовую ширину
        current_width = ci_max - ci_min
        needs_widening = current_width < min_width

        # Объединяем условия для расширения
        needs_expansion = tf.logical_or(needs_expansion, needs_widening)

        ci_min = tf.where(needs_expansion, forecast - stddev * expansion_factor, ci_min)
        ci_max = tf.where(needs_expansion, forecast + stddev * expansion_factor, ci_max)

        # Дополнительная проверка для экстремальных значений
        if true_values is not None:
            extreme_value_mask = tf.logical_or(
                forecast < tf.reduce_min(true_values) * 1.1,
                forecast > tf.reduce_max(true_values) * 0.9
            )
            if tf.reduce_any(extreme_value_mask):
                # Увеличиваем ширину для экстремальных значений
                ci_min = tf.where(extreme_value_mask, forecast - stddev * 8.0, ci_min)
                ci_max = tf.where(extreme_value_mask, forecast + stddev * 8.0, ci_max)

        return ci_min, ci_max, target_coverage

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

        return inflation_correction, is_missed_jump_flag

    @tf.function
    def val_step(self, X_batch, y_for_filtering_batch, y_target_batch,
                 initial_state, initial_covariance):
        """Шаг валидации для адаптивной UKF с контекстной волатильностью"""
        B = tf.shape(X_batch)[0]
        with tf.device(self.device):
            # 1. LSTM forward pass
            lstm_outputs = self.model(X_batch, training=False)
            params_output = lstm_outputs['params']  # [B, T, 37]
            h_lstm2 = lstm_outputs['h_lstm2']       # [B, T, 128]
    
            # 2. ЭНТРОПИЙНАЯ РЕГУЛЯРИЗАЦИЯ
            entropy_loss = self.entropy_regularizer.compute_entropy_loss(h_lstm2)
    
            # 3. Обработка выходов LSTM
            vol_context, ukf_params, inflation_config, student_t_config = self.process_lstm_output(params_output)
            
            # 4. Адаптивная UKF фильтрация
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
            x_filtered = results[0]        # [B, T, 1]
            innovations = results[1]       # [B, T, 1]
            volatility_levels = results[2] # [B, T, 1]
            inflation_factors = results[3] # [B, T, 1]
            raw_final_state = results[4]   # [B, 1]
            final_covariance = results[5]  # [B, 1, 1]
    
            # Извлечение финальной волатильности (единственный расчет)
            final_volatility = tf.squeeze(volatility_levels[:, -1, :])  # [B]
    
            # === КРИТИЧЕСКИ ВАЖНО: ВЫЧИСЛИТЬ regime_info СРАЗУ ПОСЛЕ ПОЛУЧЕНИЯ final_volatility ===
            regime_info = self.regime_selector.assign_soft_regimes(final_volatility)
    
            # === ЯВНАЯ ПРОВЕРКА СООТВЕТСТВИЯ FINAL STATE ===
            expected_final_state = x_filtered[:, -1, :]  # [B, 1]
            final_state = tf.reshape(expected_final_state, [B, 1])  # [B, 1]
    
            if self.debug_mode:
                state_diff = tf.reduce_mean(tf.abs(raw_final_state - final_state))
                tf.print("Проверка соответствия final state (val):", state_diff)
    
            # 5. ЯВНЫЙ PREDICT-ШАГ НА СЛЕДУЮЩИЙ ШАГ
            t_last = tf.shape(ukf_params['q_base'])[1] - 1
    
            # Явное извлечение параметров последнего шага
            q_base_final = tf.gather(ukf_params['q_base'], t_last, axis=1)  # [B, 1]
            q_sensitivity_final = tf.gather(ukf_params['q_sensitivity'], t_last, axis=1)  # [B, 1]
            q_floor_final = tf.gather(ukf_params['q_floor'], t_last, axis=1)  # [B, 1]
            r_base_final = tf.gather(ukf_params['r_base'], t_last, axis=1)  # [B, 1]
            r_sensitivity_final = tf.gather(ukf_params['r_sensitivity'], t_last, axis=1)  # [B, 1]
            r_floor_final = tf.gather(ukf_params['r_floor'], t_last, axis=1)  # [B, 1]
            relax_base_final = tf.gather(ukf_params['relax_base'], t_last, axis=1)  # [B, 1]
            relax_sensitivity_final = tf.gather(ukf_params['relax_sensitivity'], t_last, axis=1)  # [B, 1]
            alpha_base_final = tf.gather(ukf_params['alpha_base'], t_last, axis=1)  # [B, 1]
            alpha_sensitivity_final = tf.gather(ukf_params['alpha_sensitivity'], t_last, axis=1)  # [B, 1]
            kappa_base_final = tf.gather(ukf_params['kappa_base'], t_last, axis=1)  # [B, 1]
            kappa_sensitivity_final = tf.gather(ukf_params['kappa_sensitivity'], t_last, axis=1)  # [B, 1]
            inf_factor_final = tf.gather(inflation_factors[:, -1, :], t_last, axis=1)
    
            forecast, std_dev = self._explicit_predict_next_step(
                final_state,
                final_covariance,
                final_volatility,
                q_base_final, q_sensitivity_final, q_floor_final,
                inf_factor_final,
                relax_base_final, relax_sensitivity_final,
                alpha_base_final, alpha_sensitivity_final,
                kappa_base_final, kappa_sensitivity_final
            )
    
            # 6. РАСЧЕТ КАЛИБРОВКИ ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ
            # Адаптивное целевое покрытие (0.75-0.85 в зависимости от волатильности)
            base_confidence = 0.80
            confidence_range = 0.10
            confidence_ceil = tf.fill(tf.shape(final_volatility), base_confidence + confidence_range/2)  # 0.85
            confidence_floor = tf.fill(tf.shape(final_volatility), base_confidence - confidence_range/2)  # 0.75
            target_coverage = confidence_ceil - (confidence_ceil - confidence_floor) * final_volatility
    
            # Настройка параметров Student-t для асимметричной калибровки
            batch_shape_tensor = tf.shape(final_volatility)
            student_t_config_final = {
                'confidence_ceil': tf.fill(batch_shape_tensor, 0.95),
                'confidence_floor': tf.fill(batch_shape_tensor, 0.70),
                'dof_base': tf.fill(batch_shape_tensor, 6.0),
                'dof_sensitivity': tf.fill(batch_shape_tensor, 0.5),
                'asymmetry_pos': tf.fill(batch_shape_tensor, 0.7)[:, tf.newaxis],
                'asymmetry_neg': tf.fill(batch_shape_tensor, 1.3)[:, tf.newaxis],
                'calibration_sensitivity': tf.fill(batch_shape_tensor, 1.0),
                'tail_weight_pos': tf.fill(batch_shape_tensor, 1.2)[:, tf.newaxis],
                'tail_weight_neg': tf.fill(batch_shape_tensor, 1.5)[:, tf.newaxis],
                'confidence_base': tf.fill(batch_shape_tensor, 0.85),
            }
    
            # Адаптивные асимметричные границы доверительного интервала
            ci_lower, ci_upper, target_coverage = self._calibrate_confidence_interval(
                forecast, std_dev, final_volatility, student_t_config_final,
                innovations=innovations[:, -10:, :] if innovations is not None else None,
                regime_assignment=regime_info['regime_assignment']
            )
    
            # Проверка валидности границ
            ci_min = tf.minimum(ci_lower, ci_upper)
            ci_max = tf.maximum(ci_lower, ci_upper)
    
            # Вычисление фактического покрытия
            covered = tf.cast(
                (y_target_batch >= ci_min) & (y_target_batch <= ci_max),
                tf.float32
            )
            actual_coverage = tf.reduce_mean(covered)
            target_coverage_mean = tf.reduce_mean(target_coverage)
    
            # === ЕДИНЫЙ СОГЛАСОВАННЫЙ РАСЧЕТ ШИРИНЫ ДИ ОТНОСИТЕЛЬНО ВОЛАТИЛЬНОСТИ ДАННЫХ ===
            ci_width = ci_max - ci_min
            y_std_batch = tf.math.reduce_std(y_target_batch) + 1e-8  # Волатильность ДАННЫХ (не неопределенность модели!)
            width_ratio = tf.reduce_mean(ci_width) / y_std_batch      # ЕДИНСТВЕННЫЙ расчет для штрафа и метрики
    
            # Штраф за недостаточное/избыточное покрытие
            undercoverage_penalty = tf.cond(
                actual_coverage < target_coverage_mean,
                lambda: 5.0 * tf.square(target_coverage_mean - actual_coverage),
                lambda: 2.0 * tf.square(actual_coverage - target_coverage_mean)
            )
    
            # Штраф за ширину интервалов (адаптирован под волатильность данных)
            target_width_ratio = 3.5  # Для 85% ДИ при нормальном распределении
            width_penalty = 2.0 * tf.square(tf.maximum(width_ratio - target_width_ratio, 0.0))
            width_penalty += 3.0 * tf.square(tf.maximum(target_width_ratio * 0.7 - width_ratio, 0.0))
    
            calibration_loss = undercoverage_penalty + width_penalty
    
            # 7. РАСЧЕТ ИТОГОВОЙ ПОТЕРИ
            loss = self.compute_loss(
                forecast,
                y_target_batch,
                volatility_levels,
                inflation_factors,
                ukf_params,
                calibration_loss,
                entropy_loss
            )
    
            # Нормализация потери
            B = tf.cast(tf.shape(y_target_batch)[0], tf.float32)
            T = tf.cast(tf.shape(y_for_filtering_batch)[1], tf.float32)
            loss = loss / (B * T)
    
            # 8. РАСЧЕТ МЕТРИК
            mse_loss = tf.reduce_mean(tf.square(forecast - y_target_batch))
            avg_volatility = tf.reduce_mean(final_volatility)
            avg_inflation = tf.reduce_mean(inflation_factors[:, -1, :])
            forecast_std = tf.reduce_mean(std_dev)
    
            # Статистика режимов волатильности
            regime_soft_weights = tf.reduce_mean(regime_info['soft_weights'], axis=0)
            regime_entropy = tf.reduce_mean(regime_info['entropy'])
    
            # Q/R мониторинг
            q_val = tf.reduce_mean(q_base_final)
            r_val = tf.reduce_mean(r_base_final)
            qr_ratio_val = q_val / (r_val + 1e-8)
    
            # Адаптивный inflation
            dynamic_threshold, inflation_anomaly_ratio = compute_adaptive_threshold(
                inflation_factors,
                final_volatility,
                self.threshold_ema,
                target_anomaly_ratio=0.35
            )
    
            # Спектральная стабильность UKF
            spectrum_info_val = self.diff_ukf_component.get_spectrum_info()
            min_eigenvalue_val = spectrum_info_val['min_eigenvalue']
    
            # === СОГЛАСОВАННАЯ МЕТРИКА ДЛЯ ОТЧЕТА ===
            # Используем ТОТ ЖЕ РАСЧЕТ, что и для штрафа (через волатильность данных)
            ci_width_vs_stddev = width_ratio  # ← ЕДИНСТВЕННЫЙ источник истины
    
            # Сборка метрик
            metrics = {
                'total_loss': loss,
                'mse_loss': mse_loss,
                'entropy_loss': entropy_loss,
                'coverage_ratio': actual_coverage,
                'avg_volatility': avg_volatility,
                'avg_inflation': avg_inflation,
                'forecast_std': forecast_std,
                'target_coverage': target_coverage_mean,
                'ci_width_vs_stddev': ci_width_vs_stddev,  # ← СОГЛАСОВАНО С ШТРАФОМ И evaluate_coverage
                'qr_ratio': qr_ratio_val,
                'q_value': q_val,
                'r_value': r_val,
                'regime_low_weight': regime_soft_weights[0],
                'regime_mid_weight': regime_soft_weights[1],
                'regime_high_weight': regime_soft_weights[2],
                'regime_entropy': regime_entropy,
                'inflation_anomaly_ratio': inflation_anomaly_ratio,
                'ukf_min_eigenvalue': min_eigenvalue_val,
                'calibration_error': tf.abs(actual_coverage - target_coverage_mean),
            }
    
            return loss, metrics, final_state, final_covariance, forecast, std_dev, ci_min, ci_max, target_coverage

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

    def _scale_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Масштабирование признаков для онлайн-прогнозирования.
        Применяет сохранённые скейлеры к одиночному окну признаков.
        
        Args:
            features_df: DataFrame с признаками формы [seq_len, n_features]
        
        Returns:
            scaled_df: Масштабированный DataFrame той же формы
        """
        if self.feature_scalers is None or not hasattr(self, 'feature_scalers'):
            raise ValueError(
                "Скейлеры не загружены! "
                "Выполните подготовку данных через prepare_honest_datasets() "
                "или загрузите модель с сохранёнными скейлерами через load()."
            )
        
        # Создаём результирующий DataFrame
        scaled_df = pd.DataFrame(
            index=features_df.index,
            columns=self.feature_columns,
            dtype=np.float32
        )
        
        # 1. Масштабирование по группам
        for group_name, features in self.scale_groups.items():
            valid_features = [f for f in features if f in features_df.columns]
            if not valid_features:
                continue
            
            if group_name == 'none':
                # Признаки без масштабирования — копируем с клиппированием для стабильности
                for col in valid_features:
                    values = features_df[col].values.astype(np.float32)
                    # Специфичные ограничения для семантических признаков
                    if col == 'asymmetry_ratio':
                        values = np.clip(values, 0.1, 3.0)
                    elif col == 'percentile_pos_fisher':
                        values = np.clip(values, -5.0, 5.0)
                    scaled_df[col] = values
            else:
                # Масштабируемые признаки
                scaler = self.feature_scalers.get(group_name)
                if scaler is None:
                    # Резерв: копируем без масштабирования если скейлер отсутствует
                    for col in valid_features:
                        scaled_df[col] = features_df[col].values.astype(np.float32)
                    continue
                
                # Применяем скейлер (требует 2D массив [n_samples, n_features])
                try:
                    values_2d = features_df[valid_features].values.astype(np.float32)
                    scaled_values_2d = scaler.transform(values_2d)
                    for i, col in enumerate(valid_features):
                        scaled_df[col] = scaled_values_2d[:, i]
                except Exception as e:
                    print(f"⚠️ Ошибка масштабирования группы '{group_name}' для {valid_features}: {e}")
                    # Резервный вариант при ошибке
                    for col in valid_features:
                        scaled_df[col] = features_df[col].values.astype(np.float32)
        
        # 2. Обработка пропущенных признаков (защита от неполного покрытия группами)
        missing_cols = [col for col in self.feature_columns 
                       if col not in scaled_df.columns or scaled_df[col].isna().any()]
        if missing_cols:
            print(f"⚠️ Нераспределённые признаки: {', '.join(missing_cols)}")
            for col in missing_cols:
                if col in features_df.columns:
                    # Стратегия по умолчанию: используем RobustScaler
                    if 'robust' in self.feature_scalers and self.feature_scalers['robust'] is not None:
                        try:
                            values_2d = features_df[[col]].values.astype(np.float32)
                            scaled_values_2d = self.feature_scalers['robust'].transform(values_2d)
                            scaled_df[col] = scaled_values_2d[:, 0]
                        except:
                            scaled_df[col] = features_df[col].values.astype(np.float32)
                    else:
                        scaled_df[col] = features_df[col].values.astype(np.float32)
        
        # 3. Финальная обработка численной стабильности
        scaled_df = scaled_df.replace([np.inf, -np.inf], np.nan)
        scaled_df = scaled_df.fillna(0.0)
        scaled_df = scaled_df.astype(np.float32)
        
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
        for X_batch, y_for_filtering_batch, y_target_batch in train_ds.take(1):
            input_shape = (int(X_batch.shape[1]), int(X_batch.shape[2]))
            print(f"✅ Определена форма входа: {input_shape}")
            with tf.device('/GPU:0' if self._gpu_available else '/CPU:0'):
                self.model = self._build_model(input_shape, training=True)
            print(f"✅ LSTM модель инициализирована с формой входа: {input_shape}")
            break

        # 3. Инициализация оптимизатора
        if self._optimizer is None:
            base_lr = 5e-4
            self._optimizer = tf.keras.optimizers.Adam(
                learning_rate=base_lr,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-7,
                amsgrad=False
            )
            print(f"✅ Оптимизатор инициализирован с learning_rate={base_lr:.1e}")

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
                    baselr=5e-4,   # адаптивный базовый LR
                    minlr=1e-5,
                    warmupepochs = 3,
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

                for batch_idx, (X_batch, y_for_filtering_batch, y_target_batch) in enumerate(train_ds):
                    batch_size = tf.shape(X_batch)[0]
                    current_state_size = tf.shape(self._last_state)[0]

                    # === ИНИЦИАЛИЗАЦИЯ СОСТОЯНИЯ UKF С АДАПТИВНОЙ ДИСПЕРСИЕЙ ===
                    if batch_idx == 0 and epoch == 0:
                        # Адаптивная инициализация с учетом волатильности данных
                        window_std = tf.math.reduce_std(y_for_filtering_batch[:, :20], axis=1)
                        initial_variance = tf.maximum(window_std ** 2, 0.05)
                        max_variance = 0.5 * tf.math.reduce_variance(y_for_filtering_batch) + 0.1
                        initial_variance = tf.minimum(initial_variance, max_variance)
                        
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
                        X_batch, y_for_filtering_batch, y_target_batch,
                        initial_state, initial_covariance
                    )
                    loss, metrics, final_state, final_covariance, forecast, std_devs, volatility_levels, \
                    regime_info, vol_final, entropy_stats, batch_normalized_innov = results

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
                    self._state_initialized.assign(True)
                    self._step_counter.assign_add(1)

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
                        soft_weights = regime_info['soft_weights']  # [B, num_regimes]
                        soft_weights_mean = tf.reduce_mean(soft_weights, axis=0).numpy()

                        # Красивый вывод
                        regime_names = ['LOW', 'MID', 'HIGH']
                        regime_dist = ' | '.join([
                            f"{regime_names[i]}: {soft_weights_mean[i]:.1%}"
                            for i in range(3)
                        ])
                        print(f"\r   Batch {batch_idx+1}/{len(train_ds)} | "
                              f"Progress: {progress:.1f}% | loss={loss:.6f} | "
                              f"LSTM_Ent:{entropy_val:.1f} | "  # сократить название
                              f"Regimes(LOW/MID/HIGH): {soft_weights_mean[0]:.0%}/{soft_weights_mean[1]:.0%}/{soft_weights_mean[2]:.0%}", end='', flush=True)
                print()  # Новая строка после прогресс-бара

                # Средние метрики по эпохе
                train_loss_avg = tf.reduce_mean(epoch_losses)
                train_mse_avg = tf.reduce_mean(epoch_mse_losses)
                train_volatility_avg = tf.reduce_mean(tf.stack(epoch_volatility_levels))

                # ===== ВАЛИДАЦИЯ =====
                print("\n📉 ВАЛИДАЦИЯ...")
                val_losses = []
                val_mse_losses = []
                self.all_val_covered = []
                val_volatility_levels = []
                val_metrics = []  # Список для хранения всех метрик по шагам валидации
                for batch_idx, (X_val_batch, y_val_for_filtering_batch, y_val_target_batch) in enumerate(val_ds):
                    B_val = tf.shape(X_val_batch)[0]

                    # === ФУНКЦИИ ДЛЯ ВЫБОРА СОСТОЯНИЯ ДЛЯ ВАЛИДАЦИИ ===
                    def use_saved_state_for_val():
                        # ✅ ЕДИНЫЙ ПАТТЕРН: расширение скаляра [1] → [B_val, 1] через tile
                        initial_state = tf.tile(
                            tf.reshape(self._last_state, [1, self.state_dim]),
                            [B_val, 1]
                        )
                        initial_covariance = tf.tile(
                            tf.reshape(self._last_P, [1, self.state_dim, self.state_dim]),
                            [B_val, 1, 1]
                        )
                        return initial_state, initial_covariance
                    
                    def initialize_val_from_data():
                        base_value = y_val_for_filtering_batch[:, 0]
                        initial_state = tf.reshape(base_value, [B_val, self.state_dim])
                        initial_variance = tf.math.reduce_variance(y_val_for_filtering_batch, axis=1) + 1e-6
                        initial_covariance = tf.reshape(initial_variance, [B_val, self.state_dim, self.state_dim])
                        initial_covariance = tf.maximum(initial_covariance, 1e-8)
                        return initial_state, initial_covariance
                    
                    # Используем tf.cond для графового режима
                    initial_state_val, initial_covariance_val = tf.cond(
                        tf.logical_and(self._state_initialized, self._step_counter > 0),
                        use_saved_state_for_val,
                        initialize_val_from_data
                    )

                    # Шаг валидации
                    results_val = self.val_step(
                        X_val_batch, y_val_for_filtering_batch, y_val_target_batch,
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

                    self.all_forecasts.extend(forecast_val.numpy())
                    self.all_ci_lowers.extend(ci_lower_val.numpy())
                    self.all_ci_uppers.extend(ci_upper_val.numpy())
                    self.all_actuals.extend(y_val_target_batch.numpy())
                    self.all_target_coverages.extend(target_coverage_val.numpy())

                    # Агрегация метрик
                    val_losses.append(val_loss)
                    val_mse_losses.append(metrics_val['mse_loss'])

                    # ===== СОХРАНЕНИЕ СОСТОЯНИЯ ПОСЛЕ ВАЛИДАЦИИ =====
                    # Важно: не инкрементируем _step_counter в валидации (только в обучении)
                    self._last_state.assign(
                        tf.reduce_mean(tf.squeeze(final_state_val, axis=1), axis=0, keepdims=True)
                    )
                    self._last_P.assign(
                        tf.reduce_mean(final_covariance_val, axis=0, keepdims=True)
                    )
                    self._state_initialized.assign(True)
                    
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

                # Средние метрики валидации
                val_loss_avg = tf.reduce_mean(val_losses)
                val_mse_avg = tf.reduce_mean(val_mse_losses)
                val_volatility_avg = tf.reduce_mean(tf.stack(val_volatility_levels))

                # ===== ОБРАБОТКА РЕЗУЛЬТАТОВ =====
                epoch_end_time = datetime.datetime.now()
                epoch_duration = (epoch_end_time - epoch_start_time).total_seconds()

                if all_normalized_innov:
                    all_normalized_innov = np.concatenate(all_normalized_innov, axis=0)
                    all_normalized_innov = np.abs(all_normalized_innov).flatten()  # делаем одномерным

                # Генерация и вывод детального отчета
                epoch_report = self.generate_epoch_report(epoch, train_metrics, val_metrics, all_normalized_innov)
                print(epoch_report)

                # ===== СОХРАНЕНИЕ ЛУЧШИХ ВЕСОВ И РАННЯЯ ОСТАНОВКА =====
                current_val_loss = val_loss_avg.numpy()
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
                    elif current_val_loss > self.best_val_loss * 1.5 and epoch > min_epochs:
                        print(f"\n🛑 РАННЯЯ ОСТАНОВКА: резкое ухудшение качества (loss вырос более чем в 1.5 раза)")
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
        """Генерация компактного и информативного отчета по результатам эпохи"""
        report = f"\n{'='*80}\n📊 ОТЧЕТ ПО ЭПОХЕ {epoch+1}\n{'='*80}\n"
        
        # === 1. БАЗОВЫЕ МЕТРИКИ ===
        train_loss = tf.reduce_mean([m['total_loss'] for m in train_metrics]).numpy()
        train_mse = tf.reduce_mean([m['mse_loss'] for m in train_metrics]).numpy()
        val_loss = tf.reduce_mean([m['total_loss'] for m in val_metrics]).numpy()
        val_mse = tf.reduce_mean([m['mse_loss'] for m in val_metrics]).numpy()
        
        report += "📈 БАЗОВЫЕ МЕТРИКИ:\n"
        report += f"   TRAIN → Loss: {train_loss:.6f} | MSE: {train_mse:.6f}\n"
        report += f"   VAL   → Loss: {val_loss:.6f} | MSE: {val_mse:.6f}\n"
        
        # === 2. КАЛИБРОВКА ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ (единый формат для TRAIN/VAL) ===
        def evaluate_ci_quality(coverage, width_ratio, target_coverage):
            """Оценка качества ДИ: покрытие + ширина"""
            # Оценка покрытия
            cov_status = '✅' if 0.80 <= coverage <= 0.95 else '⚠️'
            cov_msg = 'Хорошо' if 0.80 <= coverage <= 0.95 else ('Низкое' if coverage < 0.80 else 'Избыточное')
            
            # Оценка ширины (на основе отношения к волатильности данных)
            if width_ratio < 2.0:
                width_status = '⚠️'
                width_msg = 'Слишком узкие'
            elif width_ratio > 5.0:
                width_status = '⚠️'
                width_msg = 'Слишком широкие'
            else:
                width_status = '✅'
                width_msg = 'Нормальные'
            
            # Общий статус
            overall_status = '✅ Оптимально' if (cov_status == '✅' and width_status == '✅') else '⚠️ Требует настройки'
            
            return cov_status, cov_msg, width_status, width_msg, overall_status
        
        # TRAIN
        train_cov = tf.reduce_mean([m['coverage_ratio'] for m in train_metrics]).numpy() if 'coverage_ratio' in train_metrics[0] else 0.0
        train_target_cov = tf.reduce_mean([m['target_coverage'] for m in train_metrics]).numpy()
        train_cal_err = tf.reduce_mean([m['calibration_error'] for m in train_metrics]).numpy()
        train_width_ratio = tf.reduce_mean([m['ci_width_vs_stddev'] for m in train_metrics]).numpy()
        
        cov_st, cov_msg, width_st, width_msg, overall_st = evaluate_ci_quality(train_cov, train_width_ratio, train_target_cov)
        
        report += "\n📊 КАЛИБРОВКА ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ (TRAIN):\n"
        report += f"   • Покрытие: {train_cov:.2%} (цель: {train_target_cov:.2%}) → {cov_st} {cov_msg}\n"
        report += f"   • Ошибка калибровки: {train_cal_err:.4f}\n"
        report += f"   • Ширина ДИ / волатильность данных: {train_width_ratio:.2f}x → {width_st} {width_msg}\n"
        report += f"   • Итог: {overall_st}\n"
        
        # VAL
        val_cov = tf.reduce_mean([m['coverage_ratio'] for m in val_metrics]).numpy() if 'coverage_ratio' in val_metrics[0] else 0.0
        val_target_cov = tf.reduce_mean([m['target_coverage'] for m in val_metrics]).numpy()
        val_cal_err = tf.reduce_mean([m['calibration_error'] for m in val_metrics]).numpy()
        val_width_ratio = tf.reduce_mean([m['ci_width_vs_stddev'] for m in val_metrics]).numpy()
        
        cov_sv, cov_msg_v, width_sv, width_msg_v, overall_sv = evaluate_ci_quality(val_cov, val_width_ratio, val_target_cov)
        
        report += "\n📊 КАЛИБРОВКА ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ (VAL):\n"
        report += f"   • Покрытие: {val_cov:.2%} (цель: {val_target_cov:.2%}) → {cov_sv} {cov_msg_v}\n"
        report += f"   • Ошибка калибровки: {val_cal_err:.4f}\n"
        report += f"   • Ширина ДИ / волатильность данных: {val_width_ratio:.2f}x → {width_sv} {width_msg_v}\n"
        report += f"   • Итог: {overall_sv}\n"
        
        # === 3. КОНТЕКСТ ВОЛАТИЛЬНОСТИ ===
        regime_weights = [
            tf.reduce_mean([m['regime_low_weight'] for m in train_metrics]).numpy(),
            tf.reduce_mean([m['regime_mid_weight'] for m in train_metrics]).numpy(),
            tf.reduce_mean([m['regime_high_weight'] for m in train_metrics]).numpy()
        ]
        avg_entropy = tf.reduce_mean([m['regime_entropy'] for m in train_metrics]).numpy()
        avg_vol = tf.reduce_mean([m['avg_volatility'] for m in train_metrics]).numpy()
        
        report += "\n🧠 КОНТЕКСТ ВОЛАТИЛЬНОСТИ (TRAIN):\n"
        report += f"   • Распределение режимов: LOW {regime_weights[0]:.1%} | MID {regime_weights[1]:.1%} | HIGH {regime_weights[2]:.1%}\n"
        report += f"   • Энтропия распределения: {avg_entropy:.3f} (оптимум ≈1.10)\n"
        report += f"   • Средняя волатильность: {avg_vol:.4f}\n"
        
        # === 4. Q/R ДИНАМИКА ===
        avg_qr = tf.reduce_mean([m['qr_ratio'] for m in train_metrics]).numpy()
        avg_q = tf.reduce_mean([m['q_value'] for m in train_metrics]).numpy()
        avg_r = tf.reduce_mean([m['r_value'] for m in train_metrics]).numpy()
        qr_interpretation = "доверяет измерениям" if avg_qr < 0.5 else "доверяет прогнозу"
        
        report += "\n⚙️  Q/R ДИНАМИКА:\n"
        report += f"   • Q/R ratio: {avg_qr:.3f} (Q={avg_q:.6f}, R={avg_r:.6f})\n"
        report += f"   • Интерпретация: {qr_interpretation}\n"
        
        # === 5. ADAPTIVE INFLATION ===
        avg_inflation = tf.reduce_mean([m['avg_inflation'] for m in train_metrics]).numpy()
        inflation_anomalies = tf.reduce_mean([m['inflation_anomaly_ratio'] for m in train_metrics]).numpy()
        
        report += "\n💦 ADAPTIVE INFLATION:\n"
        report += f"   • Средний фактор: {avg_inflation:.3f}\n"
        report += f"   • Доля аномалий: {inflation_anomalies:.1%}\n"
        
        # === 6. СПЕКТРАЛЬНАЯ СТАБИЛЬНОСТЬ ===
        min_eig = tf.reduce_mean([m['ukf_min_eigenvalue'] for m in train_metrics]).numpy()
        eig_status = '✅ Стабильно' if min_eig > 0.01 else '⚠️ Риск нестабильности'
        
        report += "\n🔬 UKF СПЕКТРАЛЬНАЯ СТАБИЛЬНОСТЬ:\n"
        report += f"   • Мин. собственное значение: {min_eig:.6f} → {eig_status}\n"
        
        # === 7. ИННОВАЦИИ (компактно) ===
        if all_normalized_innov is not None and len(all_normalized_innov) > 0:
            valid_innov = all_normalized_innov[np.isfinite(all_normalized_innov)]
            if len(valid_innov) > 0:
                mean_innov = np.mean(valid_innov)
                p95_innov = np.percentile(valid_innov, 95)
                innov_status = '✅ Нормальное' if mean_innov < 2.0 and p95_innov < 4.0 else '⚠️ Требует анализа'
                
                report += "\n📉 НОРМАЛИЗОВАННЫЕ ИННОВАЦИИ:\n"
                report += f"   • Среднее: {mean_innov:.3f} | 95-й перцентиль: {p95_innov:.3f} → {innov_status}\n"
        
        report += f"\n{'='*80}"
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
        """Сохранение модели с гарантированной канонической формой состояния фильтра [1] / [1, 1]"""
        import os
        import pickle
        import io
        import joblib
        
        # === 1. ВАЛИДАЦИЯ И ПОДГОТОВКА ПУТИ ===
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
        
        # === 2. СОХРАНЕНИЕ LSTM МОДЕЛИ ===
        if self.model is not None:
            lstm_keras_path = f"{full_path}_lstm.keras"
            self.model.save(lstm_keras_path)
            print(f"✅ LSTM модель сохранена: {lstm_keras_path}")
        
        # === 3. СОХРАНЕНИЕ МЕТАДАННЫХ ===
        metadata = {
            'version': '1.0.0',
            'state_dim': self.state_dim,
            'seq_len': self.seq_len,
            'feature_columns': self.feature_columns,
            'vol_window_short': self.vol_window_short,
            'vol_window_long': self.vol_window_long,
            'rolling_window_percentile': self.rolling_window_percentile,
            'emd_window': self.emd_window,
            'min_history_for_features': self.min_history_for_features,
            'num_modes': self.num_modes,
            'use_diff_ukf': self.use_diff_ukf,
            'saved_at': str(datetime.datetime.now()),
            'architecture': 'contextual_volatility'
        }
        metadata_path = f"{full_path}_metadata.pkl"
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ Метаданные сохранены: {metadata_path}")
        
        # === 4. СЕРИАЛИЗАЦИЯ СОСТОЯНИЯ С ГАРАНТИРОВАННОЙ КАНОНИЧЕСКОЙ ФОРМОЙ ===
        _last_state_val = self._last_state.numpy()
        if len(_last_state_val.shape) > 0 and _last_state_val.shape[0] > 1:
            _last_state_val = np.mean(_last_state_val).item()  # → скаляр
        else:
            _last_state_val = _last_state_val.item() if _last_state_val.shape == () else _last_state_val[0]
        
        _last_P_val = self._last_P.numpy()
        if len(_last_P_val.shape) == 3 and _last_P_val.shape[0] > 1:
            _last_P_val = np.mean(_last_P_val, axis=0).tolist()  # → [1, 1]
        elif len(_last_P_val.shape) == 2:
            _last_P_val = _last_P_val.tolist()
        else:
            _last_P_val = _last_P_val.item() if _last_P_val.shape == () else _last_P_val.tolist()
        
        model_state = {
            # UKF состояние (УСРЕДНЁННОЕ)
            '_last_state': _last_state_val,  # СКАЛЯР
            '_last_P': _last_P_val,          # [1, 1]
            '_state_initialized': self._state_initialized.numpy(),
            '_step_counter': self._step_counter.numpy(),
            '_last_anomaly_time': self._last_anomaly_time.numpy(),
            # Обучаемые параметры...
            'base_q_logit': self.base_q_logit.numpy(),
            'base_r_logit': self.base_r_logit.numpy(),
            'volatility_sensitivity': self.volatility_sensitivity.numpy(),
            'student_t_base_dof': self.student_t_base_dof.numpy(),
            'student_t_vol_sensitivity': self.student_t_vol_sensitivity.numpy(),
            'inflation_base_factor': self.inflation_base_factor.numpy(),
            'inflation_vol_sensitivity': self.inflation_vol_sensitivity.numpy(),
            'inflation_decay_rate': self.inflation_decay_rate.numpy(),
            'confidence_base': self.confidence_base.numpy(),
            'confidence_vol_sensitivity': self.confidence_vol_sensitivity.numpy(),
            # Трекинг лучших весов
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
            'use_diff_ukf': self.use_diff_ukf,
            'num_modes': self.num_modes
        }
        
        # === 5. СЕРИАЛИЗАЦИЯ СКЕЙЛЕРОВ ===
        def _safe_serialize_scaler(scaler):
            if scaler is None:
                return None
            try:
                buffer = io.BytesIO()
                joblib.dump(scaler, buffer)
                return {'type': 'joblib', 'data': buffer.getvalue(), 'class_name': scaler.__class__.__name__}
            except Exception as e:
                params = {}
                for attr in ['scale_', 'mean_', 'var_', 'n_samples_seen_', 'center_', 'scale', 'quantile_range']:
                    if hasattr(scaler, attr):
                        value = getattr(scaler, attr)
                        params[attr] = value.tolist() if isinstance(value, np.ndarray) else value
                return {'type': 'manual', 'class_name': scaler.__class__.__name__, 'params': params}
        
        if hasattr(self, 'feature_scalers') and self.feature_scalers is not None:
            model_state['feature_scalers'] = {
                'robust': _safe_serialize_scaler(self.feature_scalers.get('robust')),
                'standard': _safe_serialize_scaler(self.feature_scalers.get('standard')),
                'minmax': _safe_serialize_scaler(self.feature_scalers.get('minmax')),
                'Y': _safe_serialize_scaler(self.feature_scalers.get('Y'))
            }
            print("✅ Скейлеры признаков сохранены")
        
        if hasattr(self, 'scale_groups') and self.scale_groups is not None:
            model_state['scale_groups'] = self.scale_groups
        
        if hasattr(self, 'best_scalers') and self.best_scalers is not None:
            model_state['best_scalers'] = self.best_scalers
        
        # === 6. СОХРАНЕНИЕ СОСТОЯНИЯ ===
        state_path = f"{full_path}_state.pkl"
        with open(state_path, 'wb') as f:
            pickle.dump(model_state, f)
        print(f"✅ Состояние модели сохранено: {state_path}")
        
        # === 7. СОХРАНЕНИЕ ДОПОЛНИТЕЛЬНЫХ КОМПОНЕНТОВ ===
        if self.use_diff_ukf and hasattr(self, 'diff_ukf_component'):
            ukf_path = f"{full_path}_diff_ukf_state.pkl"
            with open(ukf_path, 'wb') as f:
                pickle.dump({'d_raw': self.diff_ukf_component.spec_param.d_raw.numpy()}, f)
            print(f"✅ Состояние дифференцируемого UKF сохранено: {ukf_path}")
        
        if hasattr(self, 'regime_selector') and self.regime_selector is not None:
            selector_path = f"{full_path}_regime_selector.pkl"
            selector_state = {
                'regime_scales': self.regime_selector.regime_scales.numpy(),
                'temperature': self.regime_selector.temperature.numpy(),
                'history': self.regime_selector._vol_history.numpy(),
                'learnable_centers': self.regime_selector.learnable_centers
            }

            if hasattr(self.regime_selector, 'center_logits'):
                selector_state['center_logits'] = self.regime_selector.center_logits.numpy()
            else:
                print("  ⚠️ center_logits не сохранён")

            with open(selector_path, 'wb') as f:
                pickle.dump(selector_state, f)
            print(f"✅ Состояние Volatility Regime Selector сохранено: {selector_path}")
        
        print("\n" + "=" * 60)
        print(f"✅ МОДЕЛЬ ПОЛНОСТЬЮ СОХРАНЕНА: {full_path}")
        print(f"   Версия: {metadata['version']}")
        print(f"   Лучшая эпоха: {self.best_epoch}, Val Loss: {self.best_val_loss:.6f}")
        print("=" * 60)
    
    
    def load(self, path: str):
        """Загрузка модели с обязательной проверкой всех критически важных компонентов"""
        import os
        import pickle
        import joblib
        import io
        
        print("\n" + "=" * 60)
        print("📥 ЗАГРУЗКА LSTM-UKF МОДЕЛИ")
        print("=" * 60)
        
        # === 1. ЗАГРУЗКА МЕТАДАННЫХ (ОБЯЗАТЕЛЬНО) ===
        metadata_path = f"{path}_metadata.pkl"
        if not os.path.exists(metadata_path):
            raise RuntimeError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: файл метаданных не найден: {metadata_path}")
        
        try:
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            print(f"✅ Метаданные загружены: версия {metadata.get('version', 'N/A')}")
        except Exception as e:
            raise RuntimeError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: не удалось загрузить метаданные из {metadata_path}: {str(e)}")
        
        # === 2. ВОССТАНОВЛЕНИЕ БАЗОВЫХ ПАРАМЕТРОВ ===
        try:
            for param_name, default_value in [
                ('seq_len', 72), ('state_dim', 1), ('num_modes', 1),
                ('vol_window_short', 36), ('vol_window_long', 150),
                ('min_history_for_features', 350), ('use_diff_ukf', True)
            ]:
                setattr(self, param_name, metadata.get(param_name, default_value))
            
            self.feature_columns = metadata.get('feature_columns', self._default_feature_columns())
            print("✅ Базовые параметры восстановлены")
        except Exception as e:
            raise RuntimeError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: не удалось восстановить базовые параметры: {str(e)}")
        
        # === 3. ЗАГРУЗКА LSTM МОДЕЛИ (ОБЯЗАТЕЛЬНО) ===
        keras_path = f"{path}_lstm.keras"
        if not os.path.exists(keras_path):
            raise RuntimeError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: файл LSTM модели не найден: {keras_path}")
        
        try:
            if self.model is None:
                self.model = tf.keras.models.load_model(
                    keras_path,
                    custom_objects={'MultiHeadAttention': tf.keras.layers.MultiHeadAttention}
                )
            print(f"✅ LSTM модель загружена: {keras_path}")
        except Exception as e:
            raise RuntimeError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: не удалось загрузить LSTM модель из {keras_path}: {str(e)}")
        
        # === 4. ЗАГРУЗКА СОСТОЯНИЯ (ОБЯЗАТЕЛЬНО) ===
        state_path = f"{path}_state.pkl"
        if not os.path.exists(state_path):
            raise RuntimeError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: файл состояния не найден: {state_path}")
        
        try:
            with open(state_path, 'rb') as f:
                model_state = pickle.load(f)
            print("✅ Состояние модели загружено")
        except Exception as e:
            raise RuntimeError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: не удалось загрузить состояние из {state_path}: {str(e)}")
        
        # === 5. ВОССТАНОВЛЕНИЕ СОСТОЯНИЯ ФИЛЬТРА (ОБЯЗАТЕЛЬНО) ===
        print("🔧 Восстановление состояния фильтра UKF...")
        
        # _last_state (обязательно)
        if '_last_state' not in model_state or model_state['_last_state'] is None:
            raise RuntimeError("❌ КРИТИЧЕСКАЯ ОШИБКА: отсутствует обязательный параметр '_last_state' в состоянии модели")
        
        try:
            saved_val = model_state['_last_state']
            if np.isscalar(saved_val) or (isinstance(saved_val, np.ndarray) and saved_val.shape == ()):
                new_val = tf.constant([float(saved_val)], dtype=tf.float32)
            else:
                new_val = tf.constant([np.mean(saved_val).item()], dtype=tf.float32)
            self._last_state.assign(new_val)
            print(f"   ✅ _last_state восстановлен как скаляр [1] = {new_val.numpy()[0]:.6f}")
        except Exception as e:
            raise RuntimeError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: не удалось восстановить _last_state: {str(e)}")
        
        # _last_P (обязательно)
        if '_last_P' not in model_state or model_state['_last_P'] is None:
            raise RuntimeError("❌ КРИТИЧЕСКАЯ ОШИБКА: отсутствует обязательный параметр '_last_P' в состоянии модели")
        
        try:
            saved_val = model_state['_last_P']
            if isinstance(saved_val, list):
                saved_val = np.array(saved_val)
            
            if len(saved_val.shape) == 3 and saved_val.shape[0] > 1:
                new_val = np.mean(saved_val, axis=0, keepdims=True)
            elif len(saved_val.shape) == 2:
                new_val = saved_val.reshape(1, 1, 1)
            elif saved_val.shape == () or len(saved_val.shape) == 1:
                new_val = np.array([[saved_val]]).reshape(1, 1, 1)
            else:
                new_val = saved_val.reshape(1, 1, 1)
            
            # Приводим к текущей форме переменной
            current_shape = self._last_P.shape.as_list()
            if current_shape == [1, 1]:
                new_val = new_val.reshape(1, 1)
            
            self._last_P.assign(new_val)
            print(f"   ✅ _last_P восстановлен как {new_val.shape}")
        except Exception as e:
            raise RuntimeError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: не удалось восстановить _last_P: {str(e)}")
        
        # Остальные состояния фильтра (обязательно)
        for var_name in ['_state_initialized', '_step_counter', '_last_anomaly_time']:
            if var_name not in model_state:
                raise RuntimeError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: отсутствует обязательный параметр '{var_name}' в состоянии модели")
            try:
                getattr(self, var_name).assign(model_state[var_name])
                print(f"   ✅ {var_name} восстановлен")
            except Exception as e:
                raise RuntimeError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: не удалось восстановить {var_name}: {str(e)}")
        
        print("✅ Состояние фильтра полностью восстановлено")
        
        # === 6. ВОССТАНОВЛЕНИЕ ОБУЧАЕМЫХ ПАРАМЕТРОВ (ОБЯЗАТЕЛЬНО) ===
        required_params = [
            'base_q_logit', 'base_r_logit', 'volatility_sensitivity',
            'student_t_base_dof', 'student_t_vol_sensitivity',
            'inflation_base_factor', 'inflation_vol_sensitivity',
            'inflation_decay_rate', 'confidence_base', 'confidence_vol_sensitivity'
        ]
        
        for param_name in required_params:
            if param_name not in model_state:
                raise RuntimeError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: отсутствует обязательный параметр '{param_name}' в состоянии модели")
            try:
                getattr(self, param_name).assign(model_state[param_name])
            except Exception as e:
                raise RuntimeError(f"❌ КРИТИЧЕСКАЯ ОШИБКА: не удалось восстановить параметр '{param_name}': {str(e)}")
        print("✅ Обучаемые параметры восстановлены")
        
        # === 7. ЗАГРУЗКА VOLATILITY REGIME SELECTOR ===
        selector_path = f"{path}_regime_selector.pkl"
        if not os.path.exists(selector_path):
            raise RuntimeError(f"❌ Файл regime_selector не найден: {selector_path}")
        
        try:
            with open(selector_path, 'rb') as f:
                selector_state = pickle.load(f)
            print(f"✅ Состояние Volatility Regime Selector загружено")
        except Exception as e:
            raise RuntimeError(f"❌ Ошибка загрузки regime_selector: {str(e)}")
        
        # КРИТИЧЕСКИ ВАЖНО: Проверяем все обязательные атрибуты
        required_attrs = ['regime_scales', 'temperature', 'history']
        
        for attr_name in required_attrs:
            if attr_name not in selector_state:
                raise RuntimeError(f"❌ Отсутствует обязательный атрибут '{attr_name}' в состоянии regime_selector")
        
        try:
            # Восстанавливаем regime_scales
            if hasattr(self.regime_selector, 'regime_scales'):
                self.regime_selector.regime_scales.assign(
                    tf.constant(selector_state['regime_scales'], dtype=tf.float32)
                )
            
            # Восстанавливаем temperature
            if hasattr(self.regime_selector, 'temperature'):
                self.regime_selector.temperature.assign(
                    tf.constant(selector_state['temperature'], dtype=tf.float32)
                )
            
            # Восстанавливаем историю волатильности
            if hasattr(self.regime_selector, 'history'):
                self.regime_selector._vol_history.assign(
                    tf.constant(selector_state['history'], dtype=tf.float32)
                )
            
            # Восстанавливаем center_logits (если существует и сохранен)
            if hasattr(self.regime_selector, 'center_logits') and 'center_logits' in selector_state:
                self.regime_selector.center_logits.assign(
                    tf.constant(selector_state['center_logits'], dtype=tf.float32)
                )
            elif hasattr(self.regime_selector, 'centers') and 'centers' in selector_state:
                self.regime_selector.centers = tf.constant(
                    selector_state['centers'], dtype=tf.float32
                )
                
            self.regime_selector.learnable_centers = selector_state.get('learnable_centers', True)
        
        except Exception as e:
            raise RuntimeError(f"❌ Ошибка при восстановлении regime_selector: {str(e)}")
        
        print(f"✅ Volatility Regime Selector восстановлен")
        
        # === 8. ВОССТАНОВЛЕНИЕ СКЕЙЛЕРОВ (ОПЦИОНАЛЬНО) ===
        def _safe_deserialize_scaler(data):
            if data is None:
                return None
            try:
                if data.get('type') == 'joblib':
                    return joblib.load(io.BytesIO(data['data']))
                elif data.get('type') == 'manual':
                    cls_name = data.get('class_name', 'StandardScaler')
                    cls_map = {
                        'RobustScaler': RobustScaler, 'StandardScaler': StandardScaler,
                        'MinMaxScaler': MinMaxScaler, 'PowerTransformer': PowerTransformer
                    }
                    scaler = cls_map.get(cls_name, StandardScaler)()
                    for k, v in data.get('params', {}).items():
                        if hasattr(scaler, k):
                            setattr(scaler, k, np.array(v) if isinstance(v, list) else v)
                    if hasattr(scaler, 'n_features_in_'):
                        scaler.n_features_in_ = 1
                    return scaler
            except Exception as e:
                print(f"   ⚠️ Предупреждение: не удалось десериализовать скейлер: {str(e)}")
                return None
        
        if 'feature_scalers' in model_state:
            try:
                self.feature_scalers = {
                    k: _safe_deserialize_scaler(v) for k, v in model_state['feature_scalers'].items()
                }
                print("✅ Скейлеры признаков восстановлены")
            except Exception as e:
                print(f"   ⚠️ Предупреждение: не удалось полностью восстановить скейлеры: {str(e)}")
        
        if 'scale_groups' in model_state:
            self.scale_groups = model_state['scale_groups']
            print("✅ scale_groups восстановлены")
        
        if 'best_scalers' in model_state:
            self.best_scalers = model_state['best_scalers']
            print("✅ best_scalers восстановлены")
        
        # === 9. ВОССТАНОВЛЕНИЕ ДИФФЕРЕНЦИРУЕМОГО UKF (ОПЦИОНАЛЬНО) ===
        ukf_path = f"{path}_diff_ukf_state.pkl"
        if os.path.exists(ukf_path) and self.use_diff_ukf and hasattr(self, 'diff_ukf_component'):
            try:
                with open(ukf_path, 'rb') as f:
                    state = pickle.load(f)
                    if 'd_raw' in state:
                        self.diff_ukf_component.spec_param.d_raw.assign(state['d_raw'])
                        print("✅ Состояние дифференцируемого UKF восстановлено")
            except Exception as e:
                print(f"   ⚠️ Предупреждение: не удалось восстановить состояние дифференцируемого UKF: {str(e)}")
        
        # === 10. ФИНАЛИЗАЦИЯ ===
        if self._optimizer is None:
            self._optimizer = tf.keras.optimizers.Adam(learning_rate=5e-4, amsgrad=False)
            print("✅ Оптимизатор инициализирован")
        
        print("\n" + "=" * 60)
        print("✅ МОДЕЛЬ УСПЕШНО ЗАГРУЖЕНА")
        print(f"   Счетчик шагов: {self._step_counter.numpy()}")
        print(f"   Состояние фильтра: {'Инициализировано' if self._state_initialized.numpy() else 'Не инициализировано'}")
        print(f"   _last_state shape: {self._last_state.shape}")
        print(f"   _last_P shape: {self._last_P.shape}")
        print(f"   Regime Selector: scales={self.regime_selector.regime_scales.numpy()}, temp={self.regime_selector.temperature.numpy():.3f}")
        if hasattr(self.regime_selector, 'center_logits'):
            print(f"   center_logits: {self.regime_selector.center_logits.numpy()}")
        print("=" * 60)
        return self

    def get_current_weights(self) -> Dict[str, Any]:
        """
        Сохранение ПОЛНОГО состояния модели с УСРЕДНЕНИЕМ по батчу для независимости от размера батча.
        Все состояния фильтра сохраняются как скаляры/матрицы без размерности батча.
        """
        # === УСРЕДНЕНИЕ СОСТОЯНИЯ ФИЛЬТРА ПО БАТЧУ ===
        _last_state_val = self._last_state.numpy()
        if len(_last_state_val.shape) > 0 and _last_state_val.shape[0] > 1:
            _last_state_val = np.mean(_last_state_val).item()  # → скаляр
        else:
            _last_state_val = _last_state_val.item() if _last_state_val.shape == () else _last_state_val[0]
    
        _last_P_val = self._last_P.numpy()
        if len(_last_P_val.shape) == 3 and _last_P_val.shape[0] > 1:
            _last_P_val = np.mean(_last_P_val, axis=0).tolist()  # → [1, 1]
        elif len(_last_P_val.shape) == 2:
            _last_P_val = _last_P_val.tolist()
        else:
            _last_P_val = _last_P_val.item() if _last_P_val.shape == () else _last_P_val.tolist()
    
        state_dict = {
            # === LSTM ВЕСА ===
            'lstm_weights': self.model.get_weights() if self.model is not None else None,
            # === UKF СОСТОЯНИЕ (УСРЕДНЁННОЕ) ===
            '_last_state': _last_state_val,  # СКАЛЯР (без размерности батча)
            '_last_P': _last_P_val,          # [1, 1] (без размерности батча)
            '_state_initialized': self._state_initialized.numpy() if hasattr(self, '_state_initialized') else False,
            '_step_counter': self._step_counter.numpy() if hasattr(self, '_step_counter') else 0,
            '_last_anomaly_time': self._last_anomaly_time.numpy() if hasattr(self, '_last_anomaly_time') else -100,
            # === СОСТОЯНИЕ БУФЕРА АНОМАЛИЙ ===
            'anomaly_buffer': self.anomaly_buffer.value().numpy() if hasattr(self, 'anomaly_buffer') else None,
            'buffer_index': self.buffer_index.numpy() if hasattr(self, 'buffer_index') else 0,
            'anomaly_buffer_size': getattr(self, 'anomaly_buffer_size', 100),
            # === ОБУЧАЕМЫЕ ПАРАМЕТРЫ ===
            'base_q_logit': self.base_q_logit.numpy() if hasattr(self, 'base_q_logit') else np.log(0.15),
            'base_r_logit': self.base_r_logit.numpy() if hasattr(self, 'base_r_logit') else np.log(1.8),
            'volatility_sensitivity': self.volatility_sensitivity.numpy() if hasattr(self, 'volatility_sensitivity') else 1.0,
            'student_t_base_dof': self.student_t_base_dof.numpy() if hasattr(self, 'student_t_base_dof') else np.log(2.5),
            'student_t_vol_sensitivity': self.student_t_vol_sensitivity.numpy() if hasattr(self, 'student_t_vol_sensitivity') else 0.3,
            'inflation_base_factor': self.inflation_base_factor.numpy() if hasattr(self, 'inflation_base_factor') else np.log(0.05),
            'inflation_vol_sensitivity': self.inflation_vol_sensitivity.numpy() if hasattr(self, 'inflation_vol_sensitivity') else 0.2,
            'inflation_decay_rate': self.inflation_decay_rate.numpy() if hasattr(self, 'inflation_decay_rate') else 0.95,
            'confidence_base': self.confidence_base.numpy() if hasattr(self, 'confidence_base') else 0.90,
            'confidence_vol_sensitivity': self.confidence_vol_sensitivity.numpy() if hasattr(self, 'confidence_vol_sensitivity') else 0.1,
            'max_width_factor_logit': self.max_width_factor_logit.numpy() if hasattr(self, 'max_width_factor_logit') else np.log(1.5),
            # === СОСТОЯНИЕ ДИФФЕРЕНЦИРУЕМОГО UKF ===
            'diff_ukf_state': {
                'd_raw': self.diff_ukf_component.spec_param.d_raw.numpy()
                if self.use_diff_ukf and hasattr(self, 'diff_ukf_component') and hasattr(self.diff_ukf_component.spec_param, 'd_raw')
                else np.log(0.1)
            },
            # === СОСТОЯНИЕ VOLATILITY REGIME SELECTOR ===
            'regime_selector_state': {
                'regime_scales': self.regime_selector.regime_scales.numpy()
                if hasattr(self, 'regime_selector') and hasattr(self.regime_selector, 'regime_scales')
                else np.array([2.96, 4.44, 6.16], dtype=np.float32),
                'temperature': self.regime_selector.temperature.numpy()
                if hasattr(self, 'regime_selector') and hasattr(self.regime_selector, 'temperature')
                else 0.4,
                'history': self.regime_selector._vol_history.numpy()
                if hasattr(self, 'regime_selector') and hasattr(self.regime_selector, '_vol_history')
                else np.zeros([1, 100], dtype=np.float32),
                'center_logits': self.regime_selector.center_logits.numpy()
                if hasattr(self, 'regime_selector') and hasattr(self.regime_selector, 'center_logits') and self.regime_selector.learnable_centers
                else np.log([0.12, 0.35, 0.75])
            },
            # === СКЕЙЛЕРЫ ===
            'feature_scalers': self.feature_scalers.copy() if hasattr(self, 'feature_scalers') and self.feature_scalers is not None else None,
            'best_scalers': self.best_scalers.copy() if hasattr(self, 'best_scalers') and self.best_scalers is not None else None,
            'scale_groups': self.scale_groups.copy() if hasattr(self, 'scale_groups') and self.scale_groups is not None else None,
            # === МЕТРИКИ ТРЕКИНГА ===
            'best_val_loss': self.best_val_loss,
            'best_epoch': self.best_epoch,
            'patience_counter': self.patience_counter,
            # === ФЛАГИ ===
            'use_diff_ukf': self.use_diff_ukf,
            'num_modes': self.num_modes,
            'state_dim': self.state_dim,
            'seq_len': self.seq_len,
            # === ЭНТРОПИЙНАЯ РЕГУЛЯРИЗАЦИЯ ===
            'lambda_entropy': self.lambda_entropy.numpy() if hasattr(self, 'lambda_entropy') else 0.02,
            'threshold_ema': self.threshold_ema.numpy() if hasattr(self, 'threshold_ema') else 3.0,
        }
        print(f"✅ Полное состояние модели сохранено (усреднённое по батчу)")
        print(f"   • _last_state: скаляр = {_last_state_val:.6f}")
        print(f"   • _last_P: форма = {[1, 1]}")
        return state_dict

    def load_best_weights(self) -> bool:
        """
        ЗАГРУЗКА ЛУЧШИХ ВЕСОВ — полное восстановление состояния модели на момент лучшей эпохи.
        
        ВАЖНО: Этот метод восстанавливает ПОЛНОЕ состояние (включая обучаемые параметры),
        что необходимо для воспроизводимости результатов. Это НЕ нарушает принцип "не перезаписывать
        обучаемые параметры" — мы восстанавливаем состояние на момент лучшей эпохи, а не загружаем
        внешние веса в середине обучения.
        
        Возвращает:
            bool — успешность загрузки
        """
        if self.best_weights_dict is None:
            print("⚠️  Нет лучших весов для загрузки (best_weights_dict is None)")
            return False
        
        try:
            print("\n" + "=" * 80)
            print("📥 ЗАГРУЗКА ЛУЧШИХ ВЕСОВ — ВОССТАНОВЛЕНИЕ ПОЛНОГО СОСТОЯНИЯ")
            print("=" * 80)
            
            # === 0. ВАЛИДАЦИЯ СТРУКТУРЫ ===
            required_keys = ['lstm_weights', '_last_state', '_last_P', 'base_q_logit', 'base_r_logit', 'regime_selector_state']
            missing_keys = [k for k in required_keys if k not in self.best_weights_dict or self.best_weights_dict[k] is None]
            if missing_keys:
                print(f"❌ КРИТИЧЕСКАЯ ОШИБКА: отсутствуют обязательные компоненты: {missing_keys}")
                return False
            
            print("✅ Структура весов валидна, начинаем восстановление...")
            
            # === 1. ЗАГРУЗКА LSTM ВЕСОВ ===
            if self.best_weights_dict['lstm_weights'] is not None and self.model is not None:
                try:
                    self.model.set_weights(self.best_weights_dict['lstm_weights'])
                    print("✅ LSTM веса загружены")
                except Exception as e:
                    print(f"   ⚠️ Ошибка при загрузке LSTM весов: {str(e)}")
                    return False
            
            # === 2. ВОССТАНОВЛЕНИЕ СОСТОЯНИЯ ФИЛЬТРА С УСРЕДНЕНИЕМ ===
            print("\n🔧 Восстановление состояния фильтра UKF (скалярное)...")
            # Восстановление _last_state как скаляра [1]
            saved_state = self.best_weights_dict['_last_state']
            if np.isscalar(saved_state) or (isinstance(saved_state, np.ndarray) and saved_state.shape == ()):
                new_state = tf.constant([float(saved_state)], dtype=tf.float32)
            else:
                new_state = tf.constant([np.mean(saved_state).item()], dtype=tf.float32)
            self._last_state.assign(new_state)
            print(f"   ✅ _last_state: восстановлен как скаляр [1] = {new_state.numpy()[0]:.6f}")
            
            # Восстановление _last_P как [1, 1, 1]
            saved_P = self.best_weights_dict['_last_P']
            if isinstance(saved_P, list):
                saved_P = np.array(saved_P)
            if len(saved_P.shape) == 3 and saved_P.shape[0] > 1:
                new_P = tf.constant(np.mean(saved_P, axis=0).reshape(1, 1, 1), dtype=tf.float32)
            elif saved_P.shape == (1, 1):
                new_P = tf.constant(saved_P.reshape(1, 1, 1), dtype=tf.float32)
            else:
                new_P = tf.constant(saved_P.reshape(1, 1, 1), dtype=tf.float32)
            self._last_P.assign(new_P)
            print(f"   ✅ _last_P: восстановлен как [1, 1, 1]")
            
            # === 3. ВОССТАНОВЛЕНИЕ БУФЕРА АНОМАЛИЙ ===
            print("\n🔧 Восстановление состояния детектора аномалий...")
            if 'anomaly_buffer' in self.best_weights_dict and self.best_weights_dict['anomaly_buffer'] is not None:
                if hasattr(self, 'anomaly_buffer'):
                    try:
                        self.anomaly_buffer.assign(self.best_weights_dict['anomaly_buffer'])
                        print("   ✅ anomaly_buffer восстановлен")
                    except Exception as e:
                        print(f"   ⚠️ Ошибка при восстановлении anomaly_buffer: {str(e)}")
            
            if 'buffer_index' in self.best_weights_dict and self.best_weights_dict['buffer_index'] is not None:
                if hasattr(self, 'buffer_index'):
                    try:
                        self.buffer_index.assign(self.best_weights_dict['buffer_index'])
                        print("   ✅ buffer_index восстановлен")
                    except Exception as e:
                        print(f"   ⚠️ Ошибка при восстановлении buffer_index: {str(e)}")
            
            # === 4. ВОССТАНОВЛЕНИЕ ОБУЧАЕМЫХ ПАРАМЕТРОВ ===
            print("\n🔧 Восстановление обучаемых параметров...")
            trainable_params = [
                ('base_q_logit', 'base_q_logit'),
                ('base_r_logit', 'base_r_logit'),
                ('volatility_sensitivity', 'volatility_sensitivity'),
                ('student_t_base_dof', 'student_t_base_dof'),
                ('student_t_vol_sensitivity', 'student_t_vol_sensitivity'),
                ('inflation_base_factor', 'inflation_base_factor'),
                ('inflation_vol_sensitivity', 'inflation_vol_sensitivity'),
                ('inflation_decay_rate', 'inflation_decay_rate'),
                ('confidence_base', 'confidence_base'),
                ('confidence_vol_sensitivity', 'confidence_vol_sensitivity'),
                ('max_width_factor_logit', 'max_width_factor_logit'),
                ('lambda_entropy', 'lambda_entropy'),
                ('threshold_ema', 'threshold_ema'),
            ]
            
            for attr_name, dict_key in trainable_params:
                if dict_key in self.best_weights_dict and self.best_weights_dict[dict_key] is not None:
                    if hasattr(self, attr_name):
                        try:
                            var = getattr(self, attr_name)
                            if isinstance(var, tf.Variable):
                                var.assign(self.best_weights_dict[dict_key])
                                print(f"   ✅ {attr_name} восстановлен")
                            else:
                                setattr(self, attr_name, self.best_weights_dict[dict_key])
                                print(f"   ✅ {attr_name} установлен")
                        except Exception as e:
                            print(f"   ⚠️ Ошибка при восстановлении {attr_name}: {str(e)}")
            
            # === 5. ВОССТАНОВЛЕНИЕ ДИФФЕРЕНЦИРУЕМОГО UKF ===
            print("\n🔧 Восстановление состояния дифференцируемого UKF...")
            if self.use_diff_ukf and 'diff_ukf_state' in self.best_weights_dict:
                if hasattr(self, 'diff_ukf_component') and hasattr(self.diff_ukf_component.spec_param, 'd_raw'):
                    try:
                        self.diff_ukf_component.spec_param.d_raw.assign(self.best_weights_dict['diff_ukf_state']['d_raw'])
                        print("✅ Состояние дифференцируемого UKF восстановлено")
                    except Exception as e:
                        print(f"   ⚠️ Ошибка при восстановлении d_raw: {str(e)}")
            
            # === 6. ВОССТАНОВЛЕНИЕ VOLATILITY REGIME SELECTOR (КРИТИЧЕСКИ ВАЖНО) ===
            print("\n🔧 Восстановление состояния Volatility Regime Selector...")
            if 'regime_selector_state' in self.best_weights_dict and hasattr(self, 'regime_selector'):
                selector_state = self.best_weights_dict['regime_selector_state']
                restored = []
                
                # regime_scales
                if 'regime_scales' in selector_state and selector_state['regime_scales'] is not None:
                    try:
                        self.regime_selector.regime_scales.assign(selector_state['regime_scales'])
                        restored.append('regime_scales')
                    except Exception as e:
                        print(f"   ⚠️ Ошибка при восстановлении regime_scales: {str(e)}")
                
                # temperature
                if 'temperature' in selector_state and selector_state['temperature'] is not None:
                    try:
                        self.regime_selector.temperature.assign(selector_state['temperature'])
                        restored.append('temperature')
                    except Exception as e:
                        print(f"   ⚠️ Ошибка при восстановлении temperature: {str(e)}")
                
                # history
                if 'history' in selector_state and selector_state['history'] is not None:
                    try:
                        self.regime_selector._vol_history.assign(selector_state['history'])
                        restored.append('history')
                    except Exception as e:
                        print(f"   ⚠️ Ошибка при восстановлении _vol_history: {str(e)}")
                
                # center_logits (если обучаемые)
                if (hasattr(self.regime_selector, 'learnable_centers') and 
                    self.regime_selector.learnable_centers and 
                    'center_logits' in selector_state and 
                    selector_state['center_logits'] is not None):
                    try:
                        self.regime_selector.center_logits.assign(selector_state['center_logits'])
                        restored.append('center_logits')
                    except Exception as e:
                        print(f"   ⚠️ Ошибка при восстановлении center_logits: {str(e)}")
                
                if restored:
                    print(f"✅ Восстановлены параметры: {', '.join(restored)}")
                else:
                    print("⚠️ Не удалось восстановить параметры Regime Selector")
            
            # === 7. ВОССТАНОВЛЕНИЕ СКЕЙЛЕРОВ ===
            print("\n🔧 Восстановление скейлеров...")
            if 'feature_scalers' in self.best_weights_dict:
                self.feature_scalers = self.best_weights_dict['feature_scalers']
                print("✅ Скейлеры признаков восстановлены")
            
            if 'best_scalers' in self.best_weights_dict:
                self.best_scalers = self.best_weights_dict['best_scalers']
                print("✅ Лучшие скейлеры восстановлены")
            
            if 'scale_groups' in self.best_weights_dict:
                self.scale_groups = self.best_weights_dict['scale_groups']
                print("✅ scale_groups восстановлены")
            
            # === 8. ОБНОВЛЕНИЕ МЕТРИК ТРЕКИНГА ===
            self.best_val_loss = self.best_weights_dict.get('best_val_loss', float('inf'))
            self.best_epoch = self.best_weights_dict.get('best_epoch', 0)
            self.patience_counter = self.best_weights_dict.get('patience_counter', 0)
            
            # === 9. ФИНАЛЬНАЯ ВАЛИДАЦИЯ ===
            print("\n" + "=" * 80)
            print("✅ ПОЛНОЕ СОСТОЯНИЕ УСПЕШНО ВОССТАНОВЛЕНО")
            print("=" * 80)
            print(f"   📅 Эпоха: {self.best_epoch + 1}")
            print(f"   📉 Val Loss: {self.best_val_loss:.6f}")
            print(f"   🔁 Счётчик шагов: {self._step_counter.numpy()}")
            print(f"   🧠 Состояние фильтра: {'Инициализировано' if self._state_initialized.numpy() else 'Не инициализировано'}")
            print(f"   🎭 Regime Selector: scales={self.regime_selector.regime_scales.numpy()}, temp={self.regime_selector.temperature.numpy():.3f}")
            print(f"   💧 Adaptive inflation: base={tf.nn.softplus(self.inflation_base_factor).numpy():.3f}")
            print(f"   📏 Max width factor: {tf.nn.softplus(self.max_width_factor_logit).numpy() + 1.0:.2f}x")
            print("=" * 80)
            
            return True
            
        except Exception as e:
            print(f"\n❌ КРИТИЧЕСКАЯ ОШИБКА при загрузке лучших весов:")
            print(f"   {str(e)}")
            import traceback
            traceback.print_exc()
            return False

    def reset_best_weights_tracking(self):
        """Сброс отслеживания лучших весов"""
        self.best_val_loss = float('inf')
        self.best_epoch = 0
        self.best_weights_dict = None
        self.patience_counter = 0

        # === ДОБАВЛЕНО: СБРОС СОСТОЯНИЯ БУФЕРА АНОМАЛИЙ ===
        if hasattr(self, 'anomaly_buffer'):
            self.anomaly_buffer.assign(tf.zeros(self.anomaly_buffer.shape))
        if hasattr(self, 'buffer_index'):
            self.buffer_index.assign(0)

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
        """
        Внутренний графовый метод для онлайн-прогнозирования.
        Все операции совместимы с @tf.function и работают в графовом режиме.
        Входы:
            X_scaled: [B, seq_len, n_features] - масштабированные признаки
            initial_state: [B, 1] - начальное состояние фильтра
            initial_covariance: [B, 1, 1] - начальная ковариация
            last_volatility: [B] - волатильность с предыдущего шага
        Выходы (все в масштабированном пространстве):
            forecast: [B] - прогноз на следующий шаг
            std_dev: [B] - стандартное отклонение прогноза
            ci_lower: [B] - нижняя граница ДИ
            ci_upper: [B] - верхняя граница ДИ
            final_state: [B, 1] - фильтрованное состояние последнего шага
            final_covariance: [B, 1, 1] - ковариация последнего шага
            final_volatility: [B] - финальная волатильность
            inflation_factor: [B] - фактор инфляции
            target_coverage: [B] - целевое покрытие ДИ
            regime_info: Dict - информация о распределении по режимам
        """
        B = tf.shape(X_scaled)[0]
        T = tf.shape(X_scaled)[1]
        
        # === 1. LSTM FORWARD PASS ===
        lstm_outputs = self.model(X_scaled, training=False)
        params_output = lstm_outputs['params']  # [B, T, 37]
        
        # === 2. ОБРАБОТКА ВЫХОДОВ LSTM ===
        vol_context, ukf_params, inflation_config, student_t_config = self.process_lstm_output(params_output)
        
        # === 3. АДАПТИВНАЯ ФИЛЬТРАЦИЯ UKF ===
        # Гарантируем правильные размерности состояния
        initial_state = tf.reshape(initial_state, [B, self.state_dim])
        initial_covariance = tf.reshape(initial_covariance, [B, self.state_dim, self.state_dim])
        
        # Извлекаем целевую переменную из признаков (колонка 'level')
        level_idx = self.feature_columns.index('level')
        y_level_batch = X_scaled[:, :, level_idx]
        
        results = self.adaptive_ukf_filter(
            X_scaled,
            y_level_batch,
            vol_context,
            ukf_params,
            inflation_config,
            student_t_config,
            initial_state,
            initial_covariance
        )
        
        # Распаковка результатов фильтрации
        x_filtered = results[0]        # [B, T, 1]
        innovations = results[1]       # [B, T, 1]
        volatility_levels = results[2] # [B, T, 1]
        inflation_factors = results[3] # [B, T, 1]
        final_state = results[4]       # [B, 1]
        final_covariance = results[5]  # [B, 1, 1]
        
        # === 4. ИЗВЛЕЧЕНИЕ ФИНАЛЬНЫХ ПАРАМЕТРОВ (КРИТИЧЕСКИ ВАЖНО: безопасное извлечение) ===
        # ❌ НЕПРАВИЛЬНО: tf.squeeze() → может вернуть скаляр [] при B=1
        # ✅ ПРАВИЛЬНО: гарантируем векторную форму [B] через reshape
        final_volatility = tf.reshape(volatility_levels[:, -1, :], [-1])  # [B] - ГАРАНТИРОВАННО вектор
        final_inflation = tf.reshape(inflation_factors[:, -1, :], [-1])   # [B] - ГАРАНТИРОВАННО вектор
        
        # === 5. ЯВНЫЙ PREDICT НА СЛЕДУЮЩИЙ ШАГ ===
        t_last = T - 1
        
        # Извлечение параметров последнего шага
        q_base_final = tf.gather(ukf_params['q_base'], t_last, axis=1)           # [B, 1]
        q_sensitivity_final = tf.gather(ukf_params['q_sensitivity'], t_last, axis=1)
        q_floor_final = tf.gather(ukf_params['q_floor'], t_last, axis=1)
        r_base_final = tf.gather(ukf_params['r_base'], t_last, axis=1)
        r_sensitivity_final = tf.gather(ukf_params['r_sensitivity'], t_last, axis=1)
        r_floor_final = tf.gather(ukf_params['r_floor'], t_last, axis=1)
        relax_base_final = tf.gather(ukf_params['relax_base'], t_last, axis=1)
        relax_sensitivity_final = tf.gather(ukf_params['relax_sensitivity'], t_last, axis=1)
        alpha_base_final = tf.gather(ukf_params['alpha_base'], t_last, axis=1)
        alpha_sensitivity_final = tf.gather(ukf_params['alpha_sensitivity'], t_last, axis=1)
        kappa_base_final = tf.gather(ukf_params['kappa_base'], t_last, axis=1)
        kappa_sensitivity_final = tf.gather(ukf_params['kappa_sensitivity'], t_last, axis=1)
        inf_factor_final = tf.reshape(final_inflation, [B, 1])  # [B] → [B, 1]
        
        forecast, std_dev = self._explicit_predict_next_step(
            final_state,
            final_covariance,
            final_volatility,
            q_base_final, q_sensitivity_final, q_floor_final,
            inf_factor_final,
            relax_base_final, relax_sensitivity_final,
            alpha_base_final, alpha_sensitivity_final,
            kappa_base_final, kappa_sensitivity_final
        )  # forecast: [B], std_dev: [B]
        
        # === 6. РАСПРЕДЕЛЕНИЕ ПО РЕЖИМАМ ВОЛАТИЛЬНОСТИ ===
        regime_info = self.regime_selector.assign_soft_regimes(final_volatility)  # final_volatility: [B]
        
        # === 7. КАЛИБРОВКА ДОВЕРИТЕЛЬНЫХ ИНТЕРВАЛОВ С УЧЁТОМ РЕЖИМОВ ===
        # Настройка параметров Student-t для калибровки
        batch_shape = tf.shape(final_volatility)
        student_t_config_final = {
            'dof_base': tf.fill(batch_shape, 6.0),
            'dof_sensitivity': tf.fill(batch_shape, 0.5),
            'asymmetry_pos': tf.fill(batch_shape, 0.7)[:, tf.newaxis],
            'asymmetry_neg': tf.fill(batch_shape, 1.3)[:, tf.newaxis],
            'regime_scale': tf.ones([B, 1], dtype=tf.float32),
            'regime_soft_weights': tf.ones([B, 3], dtype=tf.float32)
        }
        
        # Получаем мягкие веса режимов и вычисляем адаптивный масштаб
        soft_weights = regime_info['soft_weights']  # [B, 3]
        regime_scale = self.regime_selector.get_regime_scales(soft_weights)  # [B, 1]
        student_t_config_final['regime_scale'] = regime_scale
        student_t_config_final['regime_soft_weights'] = soft_weights
        
        # Калибровка ДИ с учётом режимов
        ci_lower, ci_upper, target_coverage = self._calibrate_confidence_interval(
            forecast,
            std_dev,
            final_volatility,
            student_t_config_final,
            innovations=innovations[:, -10:, :] if innovations is not None else None,
            regime_assignment=regime_info['regime_assignment']
        )  # ci_lower/ci_upper: [B], target_coverage: [B]
        
        # === 8. ГАРАНТИЯ ВАЛИДНОСТИ ГРАНИЦ ===
        ci_min = tf.minimum(ci_lower, ci_upper)
        ci_max = tf.maximum(ci_lower, ci_upper)
        
        return (
            forecast,           # [B]
            std_dev,            # [B]
            ci_min,             # [B]
            ci_max,             # [B]
            final_state,        # [B, 1]
            final_covariance,   # [B, 1, 1]
            final_volatility,   # [B] - ГАРАНТИРОВАННО вектор
            final_inflation,    # [B] - ГАРАНТИРОВАННО вектор
            target_coverage,    # [B] - ГАРАНТИРОВАННО вектор
            regime_info         # Dict[str, tf.Tensor]
        )

    def online_predict(
        self,
        df: pd.DataFrame,
        reset_state: bool = False,
        return_components: bool = False,
        ground_truth_available: bool = False
    ) -> Dict[str, Any]:
        """
        Онлайн-прогноз на основе ПОЛНОЙ переданной истории (минимум min_history_for_features точек).
        
        КРИТИЧЕСКИ ВАЖНО: Признаки рассчитываются на ВСЕЙ переданной истории (не обрезаются до seq_len!),
        что обеспечивает корректную декомпозицию EMD и стабильные признаки волатильности.
        
        Аргументы:
            df: DataFrame с OHLC данными (минимум min_history_for_features точек)
            reset_state: сбросить состояние фильтра перед прогнозом (для первого шага)
            return_components: вернуть все компоненты для отладки
            ground_truth_available: флаг для внутренней логики (не используется в текущей реализации)
        
        Возвращает:
            Словарь с прогнозом, доверительными интервалами и диагностикой
        """
        # === 1. ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ ===
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")
        
        if len(df) < self.min_history_for_features:
            raise ValueError(
                f"Требуется минимум {self.min_history_for_features} точек для расчёта признаков, "
                f"получено {len(df)}"
            )
        
        # === 2. РАСЧЁТ ПРИЗНАКОВ НА ПОЛНОЙ ИСТОРИИ (КЛЮЧЕВОЕ ИЗМЕНЕНИЕ!) ===
        # ✅ ПРАВИЛЬНО: используем ВСЮ переданную историю для расчёта признаков
        # Это гарантирует корректную декомпозицию EMD и стабильные оценки волатильности
        features_df = self.prepare_features(df, mode='batch')
        
        # Проверка достаточности признаков после расчёта
        if len(features_df) < self.seq_len:
            raise ValueError(
                f"После расчёта признаков осталось {len(features_df)} точек, "
                f"требуется минимум {self.seq_len}"
            )
        
        # === 3. ПОДГОТОВКА ОКНА ДЛЯ МОДЕЛИ ===
        # Берём последние seq_len признаков ТОЛЬКО ПОСЛЕ расчёта всех признаков
        X_window = features_df.tail(self.seq_len).copy()
        
        # Масштабирование признаков
        X_scaled_df = self._scale_features(X_window)
        X_scaled_np = X_scaled_df[self.feature_columns].values.astype(np.float32)
        X_scaled_tensor = tf.convert_to_tensor(
            X_scaled_np.reshape(1, self.seq_len, len(self.feature_columns)),
            dtype=tf.float32
        )
        
        # === 4. ИНИЦИАЛИЗАЦИЯ/ОБНОВЛЕНИЕ СОСТОЯНИЯ ФИЛЬТРА ===
        B = 1
        
        # ✅ КОРРЕКТНАЯ ЛОГИКА УПРАВЛЕНИЯ СОСТОЯНИЕМ (учитывает reset_state)
        if reset_state or not self._state_initialized.numpy():
            # === ИНИЦИАЛИЗАЦИЯ НОВОГО СОСТОЯНИЯ ===
            # ❌ ИСПРАВЛЕНО: двумерная индексация вместо трёхмерной
            # Было: X_scaled_np[0, :min(10, self.seq_len), 0] → ошибка для 2D массива
            # Стало: X_scaled_np[:min(10, self.seq_len), 0] → корректно для [seq_len, n_features]
            window_std = np.std(X_scaled_np[:min(10, self.seq_len), 0])  # ✅ 2 индекса для 2D массива
            
            initial_variance = max(window_std ** 2, 0.05)
            initial_variance = min(initial_variance, 0.5)
            
            # ❌ ИСПРАВЛЕНО: аналогичная ошибка в вычислении initial_state_val
            # Было: X_scaled_np[0, :min(5, self.seq_len), 0] → ошибка
            # Стало: X_scaled_np[:min(5, self.seq_len), 0] → корректно
            initial_state_val = np.mean(X_scaled_np[:min(5, self.seq_len), 0])  # ✅ 2 индекса для 2D массива
            
            initial_state = tf.constant([[initial_state_val]], dtype=tf.float32)
            initial_covariance = tf.constant([[[initial_variance]]], dtype=tf.float32)
            initial_volatility = tf.constant([window_std], dtype=tf.float32)
            
            # Сохраняем состояние для следующего вызова
            self._last_state.assign(tf.squeeze(initial_state, axis=1))
            self._last_P.assign(initial_covariance)
            self._state_initialized.assign(True)
            self._step_counter.assign(0)
            
            if self.debug_mode:
                print(f"🔄 Состояние фильтра ИНИЦИАЛИЗИРОВАНО (reset_state={reset_state})")
                print(f"   initial_state_val={initial_state_val:.6f}, initial_variance={initial_variance:.6f}")
        else:
            # === ИСПОЛЬЗОВАНИЕ СОХРАНЁННОГО СОСТОЯНИЯ ===
            # ✅ ГАРАНТИРОВАННАЯ ОБРАБОТКА ФОРМЫ [1] ДЛЯ _last_state И [1, 1, 1] ДЛЯ _last_P
            # Расширяем скалярное состояние [1] → [B, 1]
            initial_state = tf.tile(
                tf.reshape(self._last_state, [1, self.state_dim]),
                [B, 1]
            )
            # Расширяем ковариацию [1, 1, 1] → [B, 1, 1]
            initial_covariance = tf.tile(
                tf.reshape(self._last_P, [1, self.state_dim, self.state_dim]),
                [B, 1, 1]
            )
            initial_volatility = tf.constant([0.1], dtype=tf.float32)
            
            if self.debug_mode:
                print(f"🔄 Состояние фильтра ЗАГРУЖЕНО из предыдущего вызова")
                print(f"   _last_state={self._last_state.numpy()[0]:.6f}, _last_P={self._last_P.numpy()[0,0,0]:.6f}")
        
        # === 5. ПРОГНОЗ НА СЛЕДУЮЩИЙ ШАГ ===
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
        
        # Сохраняем состояние для следующего вызова (критично для онлайн-режима)
        self._last_state.assign(tf.squeeze(final_state, axis=1))
        self._last_P.assign(final_covariance)
        self._step_counter.assign_add(1)
        
        # === 6. ОБРАТНОЕ ПРЕОБРАЗОВАНИЕ В ИСХОДНЫЙ МАСШТАБ ===
        forecast_original = self.inverse_transform_target(forecast_scaled.numpy())
        ci_lower_original = self.inverse_transform_target(ci_lower_scaled.numpy())
        ci_upper_original = self.inverse_transform_target(ci_upper_scaled.numpy())
        
        # Стандартное отклонение в исходном масштабе
        if self.feature_scalers is not None and 'Y' in self.feature_scalers:
            if hasattr(self.feature_scalers['Y'], 'scale_'):
                scale_factor = self.feature_scalers['Y'].scale_
                std_dev_original = std_dev_scaled.numpy() * scale_factor
            else:
                # Резервный вариант: оценка масштаба из данных
                y_range = np.ptp(df['Close'].values[-self.seq_len:]) if 'Close' in df.columns else 1.0
                std_dev_original = std_dev_scaled.numpy() * (y_range / 2.0)
        else:
            std_dev_original = std_dev_scaled.numpy()
        
        # === 7. ФОРМИРОВАНИЕ РЕЗУЛЬТАТА ===
        # Временная метка последней точки истории (момент прогноза)
        timestamp = df.index[-1] if hasattr(df.index, '__iter__') and len(df.index) > 0 else len(df)
        
        result = {
            'timestamp': timestamp,
            'level_forecast': forecast_original,           # [1] - прогноз level[t] в исходном масштабе
            'level_forecast_scaled': forecast_scaled.numpy(),  # [1] - в масштабированном пространстве
            'std_dev': std_dev_original,                  # [1] - std в исходном масштабе
            'std_dev_scaled': std_dev_scaled.numpy(),     # [1] - std в масштабированном пространстве
            'level_ci_lower': ci_lower_original,          # [1] - нижняя граница ДИ (исходный)
            'level_ci_lower_scaled': ci_lower_scaled.numpy(),  # [1] - нижняя граница ДИ (масштабированный)
            'level_ci_upper': ci_upper_original,          # [1] - верхняя граница ДИ (исходный)
            'level_ci_upper_scaled': ci_upper_scaled.numpy(),  # [1] - верхняя граница ДИ (масштабированный)
			'volatility_level': final_volatility.numpy().item(),   # скаляр - уровень волатильности
			'inflation_factor': final_inflation.numpy().item(),    # скаляр - фактор инфляции
			'confidence': target_coverage.numpy().item(),          # скаляр - целевой уровень покрытия
			'regime': regime_info['regime_assignment'].numpy().item()  # 0=LOW, 1=MID, 2=HIGH
        }
        
        # Дополнительные компоненты для отладки (опционально)
        if return_components:
            result.update({
                'features_used': X_scaled_df,              # последние seq_len масштабированных признаков
                'raw_features': features_df,               # все признаки на полной истории
                'covariance': final_covariance.numpy(),    # ковариационная матрица фильтра
                'state': final_state.numpy(),              # состояние фильтра
                'inflation_state': {
                    'factor': final_inflation.numpy()[0],
                    'remaining_steps': regime_info.get('remaining_steps', tf.constant([0])).numpy()[0],
                    'last_anomaly_time': regime_info.get('last_anomaly_time', tf.constant([0])).numpy()[0]
                }
            })
        
        return result
    
    def evaluate(
        self,
        df: pd.DataFrame,
        plot: bool = False,
        N: int = 300
    ) -> Dict[str, float]:
        """
        Честная оценка модели на тестовых данных с использованием скользящего окна.
        Все признаки рассчитываются на корректной истории (минимум min_history_for_features точек).
        Нет утечки будущего: признаки для прогноза точки t рассчитываются только на истории [t-350 .. t-1].
        
        Аргументы:
            df: DataFrame с колонками ['Open', 'High', 'Low', 'Close']
            plot: флаг построения графиков
            N: количество точек для визуализации
        
        Возвращает:
            Словарь с метриками качества
        """
        print("\n" + "=" * 80)
        print("🔍 НАЧАЛО ЧЕСТНОЙ ОЦЕНКИ МОДЕЛИ (БЕЗ УТЕЧКИ БУДУЩЕГО)")
        print("=" * 80)
        
        # === 1. ВАЛИДАЦИЯ ВХОДНЫХ ДАННЫХ ===
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"❌ Отсутствуют обязательные колонки в данных: {missing_cols}")
        
        window_size = self.min_history_for_features  # 350
        if len(df) < window_size + 1:
            raise ValueError(
                f"❌ Недостаточно данных для оценки. Требуется минимум {window_size + 1} точек, "
                f"получено {len(df)}"
            )
        
        # === 2. СОХРАНЕНИЕ ИСХОДНОГО СОСТОЯНИЯ ФИЛЬТРА (КРИТИЧЕСКИ ВАЖНО: ДО ЛЮБЫХ ИЗМЕНЕНИЙ) ===
        original_state_initialized_val = self._state_initialized.numpy()
        original_last_state_val = self._last_state.numpy().copy() if original_state_initialized_val else None
        original_last_P_val = self._last_P.numpy().copy() if original_state_initialized_val else None
        original_step_counter_val = self._step_counter.numpy()
        
        print(f"💾 Сохранено исходное состояние фильтра: initialized={original_state_initialized_val}")
        
        # === 3. СБРОС СОСТОЯНИЯ ДЛЯ ЧИСТОЙ ОЦЕНКИ ===
        self._state_initialized.assign(False)
        print(f"🔄 Состояние фильтра сброшено перед оценкой (чистый старт)")
        
        # === 4. ИНИЦИАЛИЗАЦИЯ МАССИВОВ ДЛЯ ХРАНЕНИЯ РЕЗУЛЬТАТОВ ===
        timestamps = []
        true_values_original = []   # level[t] в исходном масштабе
        true_values_scaled = []     # level[t] в масштабированном пространстве
        pred_values_original = []   # прогноз в исходном масштабе
        pred_values_scaled = []     # прогноз в масштабированном пространстве
        pi_lower_original = []      # нижняя граница ДИ (исходный масштаб)
        pi_upper_original = []      # верхняя граница ДИ (исходный масштаб)
        pi_lower_scaled = []        # нижняя граница ДИ (масштабированный)
        pi_upper_scaled = []        # верхняя граница ДИ (масштабированный)
        volatility_levels = []      # уровень волатильности
        inflation_factors = []      # фактор инфляции ковариации
        confidences = []            # целевой уровень покрытия ДИ
        regimes = []                # режим фильтра (0=нормальный, 1=высокая волатильность)
        
        # === 5. СКОЛЬЗЯЩЕЕ ОКНО ОЦЕНКИ ===
        # t — индекс прогнозируемой точки level[t]
        total_steps = len(df) - window_size
        print(f"📊 Обработка {total_steps} шагов оценки (окно = {window_size} точек)...")
        
        # Прогресс-бар для отслеживания хода оценки
        progress_bar = tqdm(range(window_size, len(df)), desc="Оценка модели", unit="шаг")
        
        try:
            for t in progress_bar:
                try:
                    # === ШАГ 1: ИСТОРИЯ ДЛЯ РАСЧЁТА ПРИЗНАКОВ (БЕЗ УТЕЧКИ БУДУЩЕГО) ===
                    # История: [t-350 .. t-1] → ровно 350 точек ДО момента прогноза
                    history_features = df.iloc[t - window_size : t].copy()
                    
                    # === ШАГ 2: ПРОГНОЗ ЧЕРЕЗ ИСПРАВЛЕННЫЙ online_predict ===
                    # online_predict рассчитает признаки на ПОЛНОЙ переданной истории (350 точек)
                    # Состояние сохраняется между окнами благодаря: reset_state=(t == window_size)
                    result = self.online_predict(
                        history_features,
                        reset_state=(t == window_size),  # Сбрасываем состояние ТОЛЬКО для первого шага
                        return_components=False,
                        ground_truth_available=False
                    )
                    
                    # === ШАГ 3: ПОЛУЧЕНИЕ ЧЕСТНОГО GROUND TRUTH (level[t]) ===
                    # Расширенная история: [t-350 .. t] → 351 точка (включая прогнозируемую)
                    # КРИТИЧЕСКИ ВАЖНО: расчёт через тот же prepare_features для сохранения контекста!
                    history_gt = df.iloc[t - window_size : t + 1].copy()
                    features_gt = self.prepare_features(history_gt, mode='batch')
                    level_t_true = features_gt.iloc[-1]['level']  # level[t] в исходном масштабе
                    
                    # Масштабируем для точного сравнения в пространстве обучения
                    if self.feature_scalers is not None and 'Y' in self.feature_scalers:
                        level_t_true_scaled = self.feature_scalers['Y'].transform([[level_t_true]])[0, 0]
                    else:
                        level_t_true_scaled = level_t_true
                    
                    # === ШАГ 4: СОХРАНЕНИЕ РЕЗУЛЬТАТОВ ===
                    # Временная метка прогнозируемой точки
                    timestamp = df.index[t] if hasattr(df.index, '__iter__') and len(df.index) > t else t
                    
                    # Агрегация результатов
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
                    confidences.append(result['confidence'])
                    regimes.append(result['regime'])
                    
                except Exception as e:
                    print(f"🔴 Ошибка на шаге t={t}: {str(e)}")
                    print(f"   Тип ошибки: {type(e).__name__}")
                    import traceback
                    traceback.print_exc()  # Полный стек вызовов
                    raise
            
            # === 6. ПРОВЕРКА РЕЗУЛЬТАТОВ ===
            if len(true_values_original) == 0:
                raise RuntimeError("❌ Не удалось выполнить ни одного прогноза для оценки")
            
            print(f"\n✅ Успешно обработано {len(true_values_original)} шагов из {total_steps}")
            
            # === 7. ПРЕОБРАЗОВАНИЕ В NUMPY МАССИВЫ ===
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
            
            # === 8. РАСЧЁТ ОШИБОК ===
            errors_original = pred_values_original - true_values_original
            errors_scaled = pred_values_scaled - true_values_scaled
            
            # === 9. РАСЧЁТ МЕТРИК ===
            metrics = {}
            
            # Базовые метрики (в исходном масштабе)
            metrics['MAE'] = float(np.mean(np.abs(errors_original)))
            metrics['RMSE'] = float(np.sqrt(np.mean(errors_original ** 2)))
            metrics['MAPE'] = float(np.mean(np.abs(errors_original / (np.abs(true_values_original) + 1e-8))))
            metrics['MAPE_median'] = float(np.median(np.abs(errors_original / (np.abs(true_values_original) + 1e-8))))
            
            # R² коэффициент
            ss_res = np.sum(errors_original ** 2)
            ss_tot = np.sum((true_values_original - np.mean(true_values_original)) ** 2)
            metrics['R2'] = float(1 - ss_res / (ss_tot + 1e-8))
            
            # Метрики доверительных интервалов (90% уровень доверия)
            valid_coverage_original = (true_values_original >= pi_lower_original) & (true_values_original <= pi_upper_original)
            metrics['CoverageRatio'] = float(np.mean(valid_coverage_original))
            metrics['CoverageCount'] = int(np.sum(valid_coverage_original))
            metrics['TotalCount'] = len(true_values_original)
            
            # Ширина доверительных интервалов
            pi_widths_original = pi_upper_original - pi_lower_original
            metrics['MeanPIWidth'] = float(np.mean(pi_widths_original))
            metrics['MedianPIWidth'] = float(np.median(pi_widths_original))
            metrics['StdPIWidth'] = float(np.std(pi_widths_original))
            
            # Статистика по волатильности
            metrics['VolatilityMean'] = float(np.nanmean(volatility_levels))
            metrics['VolatilityStd'] = float(np.nanstd(volatility_levels))
            metrics['VolatilityMax'] = float(np.nanmax(volatility_levels))
            metrics['VolatilityMin'] = float(np.nanmin(volatility_levels))
            
            # Статистика по адаптивному inflation
            metrics['InflationMean'] = float(np.nanmean(inflation_factors))
            metrics['InflationStd'] = float(np.nanstd(inflation_factors))
            metrics['InflationMax'] = float(np.nanmax(inflation_factors))
            metrics['InflationMin'] = float(np.nanmin(inflation_factors))
            
            # Статистика по уровню уверенности (покрытию ДИ)
            metrics['ConfidenceMean'] = float(np.nanmean(confidences))
            metrics['ConfidenceStd'] = float(np.nanstd(confidences))
            
            # Дополнительные метрики
            metrics['CalibrationError'] = abs(metrics['CoverageRatio'] - 0.90)  # для 90% доверительного интервала
            if len(errors_original) > 1:
                metrics['DirectionalAccuracy'] = float(
                    np.mean((np.sign(errors_original[:-1]) != np.sign(errors_original[1:])).astype(int))
                )
            else:
                metrics['DirectionalAccuracy'] = 0.0
            
            # === 10. ВЫВОД МЕТРИК ===
            print("\n" + "=" * 60)
            print("📊 РЕЗУЛЬТАТЫ ЧЕСТНОЙ ОЦЕНКИ МОДЕЛИ")
            print("=" * 60)
            print(f"   📈 MAE: {metrics['MAE']:.6f}")
            print(f"   📉 RMSE: {metrics['RMSE']:.6f}")
            print(f"   📊 MAPE (среднее): {metrics['MAPE']:.4%}")
            print(f"   📊 MAPE (медиана): {metrics['MAPE_median']:.4%}")
            print(f"   🎯 R²: {metrics['R2']:.6f}")
            print(f"   🎪 Покрытие PI (90%): {metrics['CoverageRatio']:.2%} "
                  f"({metrics['CoverageCount']}/{metrics['TotalCount']})")
            print(f"   📏 Ширина PI (средняя): {metrics['MeanPIWidth']:.6f}")
            print(f"   📏 Ширина PI (медиана): {metrics['MedianPIWidth']:.6f}")
            print(f"   🔥 Средняя волатильность: {metrics['VolatilityMean']:.4f} ± {metrics['VolatilityStd']:.4f}")
            print(f"   💦 Средний adaptive inflation: {metrics['InflationMean']:.4f} ± {metrics['InflationStd']:.4f}")
            print(f"   ⚖️ Ошибка калибровки: {metrics['CalibrationError']:.4f}")
            print("=" * 60)
            
            # === 11. ОТРИСОВКА ГРАФИКОВ ПРИ НЕОБХОДИМОСТИ ===
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
            # === 12. ВОССТАНОВЛЕНИЕ ИСХОДНОГО СОСТОЯНИЯ ФИЛЬТРА ===
            # Это критически важно для корректной работы после оценки в продакшене
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
        figsize: tuple = (14, 12)
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

        # Создание фигуры и сетки графиков
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, 2, hspace=0.35, wspace=0.2)

        # 1. Основной график: Истинные значения и прогнозы
        ax1 = fig.add_subplot(gs[0, :])
        ax1.plot(plot_indices, true_plot, 'b-', linewidth=2.5, label='Истинное значение', alpha=0.9)
        ax1.plot(plot_indices, pred_plot, 'r--', linewidth=2.5, label='Прогноз', alpha=0.9)
        ax1.fill_between(plot_indices, pi_lower_plot, pi_upper_plot, color='gold', alpha=0.4,
                        label=f"90% доверительный интервал (ширина: {metrics['MeanPIWidth']:.4f})")

        # Настройка оси X для основного графика
        if is_time_index:
            # Получаем оригинальные временные метки для подписей
            time_labels = timestamps[start_idx:]
            # Выбираем 5 ключевых точек для подписей
            tick_positions = np.linspace(0, len(plot_indices)-1, 5, dtype=int)
            tick_labels = [time_labels[i].strftime('%Y-%m-%d') for i in tick_positions]
            ax1.set_xticks(plot_indices[tick_positions])
            ax1.set_xticklabels(tick_labels, rotation=45, ha='right')
            ax1.set_xlabel('Дата', fontsize=12)
        else:
            # Для числовых индексов используем равномерные шаги
            tick_positions = np.linspace(0, len(plot_indices)-1, 5, dtype=int)
            tick_labels = [f"{int(plot_indices[i])}" for i in tick_positions]
            ax1.set_xticks(plot_indices[tick_positions])
            ax1.set_xticklabels(tick_labels)
            ax1.set_xlabel('Индекс наблюдения', fontsize=12)

        ax1.set_ylabel('Значение', fontsize=12)
        ax1.set_title('Прогноз vs Истинные значения с доверительным интервалом', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # 2. График ошибок
        ax2 = fig.add_subplot(gs[1, 0])
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
                     fontsize=13, fontweight='bold')
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, linestyle='--', alpha=0.7)

        # 3. График уровня волатильности
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.plot(plot_indices, volatility_plot, 'b-', linewidth=2, label='Уровень волатильности', alpha=0.9)

        # Горизонтальные линии для зон волатильности
        ax3.axhline(y=0.3, color='g', linestyle='--', alpha=0.5, label='Низкая волатильность (<0.3)')
        ax3.axhline(y=0.7, color='y', linestyle='--', alpha=0.5, label='Высокая волатильность (>0.7)')
        ax3.fill_between(plot_indices, 0, 0.3, color='green', alpha=0.15)
        ax3.fill_between(plot_indices, 0.7, 1.0, color='yellow', alpha=0.15)

        # Настройка оси X
        self._setup_xaxis(ax3, plot_indices, timestamps[start_idx:], is_time_index)
        ax3.set_ylim(0, 1.05)
        ax3.set_ylabel('Уровень волатильности', fontsize=12)
        ax3.set_title(f'Контекстная волатильность (среднее: {metrics["VolatilityMean"]:.4f})',
                     fontsize=13, fontweight='bold')
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, linestyle='--', alpha=0.7)

        # 4. График adaptive inflation
        ax4 = fig.add_subplot(gs[2, 0])
        ax4.plot(plot_indices, inflation_plot, 'm-', linewidth=2, label='Adaptive inflation factor', alpha=0.9)
        ax4.axhline(y=1.0, color='r', linestyle='--', alpha=0.7, label='Базовый уровень (1.0)')

        # Настройка оси X
        self._setup_xaxis(ax4, plot_indices, timestamps[start_idx:], is_time_index)
        ax4.set_ylim(0.95, max(1.5, np.max(inflation_plot) * 1.1))
        ax4.set_ylabel('Inflation factor', fontsize=12)
        ax4.set_title(f'Adaptive inflation (среднее: {metrics["InflationMean"]:.4f})',
                     fontsize=13, fontweight='bold')
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, linestyle='--', alpha=0.7)

        # 5. График покрытия доверительного интервала
        ax5 = fig.add_subplot(gs[2, 1])
        coverage = (true_vals[start_idx:] >= pi_lower_plot) & (true_vals[start_idx:] <= pi_upper_plot)
        coverage_cum = np.cumsum(coverage) / np.arange(1, len(coverage)+1)
        ax5.plot(plot_indices, coverage_cum, 'c-', linewidth=2.5, label='Текущее покрытие', alpha=0.9)
        ax5.axhline(y=0.9, color='r', linestyle='--', alpha=0.7, label='Целевое покрытие (90%)')
        ax5.axhline(y=metrics['CoverageRatio'], color='g', linestyle='-', alpha=0.7,
                   label=f'Фактическое покрытие ({metrics["CoverageRatio"]:.2%})')

        # Настройка оси X
        self._setup_xaxis(ax5, plot_indices, timestamps[start_idx:], is_time_index)
        ax5.set_ylim(0, 1.05)
        ax5.set_ylabel('Покрытие ДИ', fontsize=12)
        ax5.set_title(f'Кумулятивное покрытие 90% ДИ\n(Ошибка калибровки: {metrics["CalibrationError"]:.4f})',
                     fontsize=13, fontweight='bold')
        ax5.legend(loc='best', fontsize=9)
        ax5.grid(True, linestyle='--', alpha=0.7)

        # 6. График уровня уверенности
        ax6 = fig.add_subplot(gs[3, 0])
        ax6.plot(plot_indices, confidence_plot, 'orange', linewidth=2, label='Уровень уверенности', alpha=0.9)
        ax6.axhline(y=np.mean(confidence_plot), color='b', linestyle='--', alpha=0.7,
                   label=f'Среднее: {np.mean(confidence_plot):.4f}')

        # Настройка оси X
        self._setup_xaxis(ax6, plot_indices, timestamps[start_idx:], is_time_index)
        ax6.set_ylim(0, 1.05)
        ax6.set_ylabel('Уровень уверенности', fontsize=12)
        ax6.set_title(f'Уровень уверенности прогнозов\n(среднее: {metrics["ConfidenceMean"]:.4f})',
                     fontsize=13, fontweight='bold')
        ax6.legend(loc='best', fontsize=9)
        ax6.grid(True, linestyle='--', alpha=0.7)

        # 7. Текстовая информация с метриками
        ax7 = fig.add_subplot(gs[3, 1])
        ax7.axis('off')

        # Форматирование метрик для отображения
        metrics_text = (
            f"ОСНОВНЫЕ МЕТРИКИ МОДЕЛИ (последние {N} точек):\n\n"
            f"Прогностическая точность:\n"
            f"   • MAE: {metrics['MAE']:.6f}\n"
            f"   • RMSE: {metrics['RMSE']:.6f}\n"
            f"   • MAPE (медиана): {metrics['MAPE_median']:.4%}\n"
            f"   • R²: {metrics['R2']:.6f}\n\n"
            f"Надежность доверительных интервалов:\n"
            f"   • Покрытие 90% ДИ: {metrics['CoverageRatio']:.2%} ({metrics['CoverageCount']}/{metrics['TotalCount']})\n"
            f"   • Средняя ширина ДИ: {metrics['MeanPIWidth']:.6f}\n"
            f"   • Ошибка калибровки: {metrics['CalibrationError']:.4f}\n\n"
            f"Динамика волатильности:\n"
            f"   • Средняя волатильность: {metrics['VolatilityMean']:.4f} ± {metrics['VolatilityStd']:.4f}\n"
            f"   • Максимальная волатильность: {metrics['VolatilityMax']:.4f}\n"
            f"   • Средний adaptive inflation: {metrics['InflationMean']:.4f}"
        )
        ax7.text(0.02, 0.95, metrics_text, fontsize=10, family='monospace',
                bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='gray'),
                verticalalignment='top', horizontalalignment='left',
                transform=ax7.transAxes)

        # Общий заголовок
        plt.suptitle(f'Результаты оценки модели LSTM-UKF с контекстной волатильностью | '
                    f'{datetime.datetime.now().strftime("%Y-%m-%d %H:%M")}',
                    fontsize=16, fontweight='bold', y=0.98)

        # Сохранение графика
        timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f'evaluation_results_{timestamp_str}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"\nГрафик результатов оценки сохранен: {filename}")
        plt.show()
        
