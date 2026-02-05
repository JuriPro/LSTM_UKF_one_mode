import os
import pickle
import warnings
import numpy as np
import pandas as pd
import datetime
import multiprocessing as mp
from typing import Optional, List, Dict, Tuple, Any
from joblib import Parallel, delayed
from sklearn.preprocessing import PowerTransformer
# Визуализация
import matplotlib
matplotlib.use('Agg')  # Безопасный бэкенд для серверов без GUI
import matplotlib.pyplot as plt


class HonestDataPreparator:
    """
    Честная подготовка данных для временных рядов БЕЗ утечки будущего.
    
    Ключевые гарантии:
    1. ГЛОБАЛЬНАЯ ХРОНОЛОГИЯ сохраняется: все train окна старше всех val окон
    2. ЛОКАЛЬНАЯ СТРАТИФИКАЦИЯ: баланс режимов через хронологические блоки
    3. ВОСПРОИЗВОДИМОСТЬ: квантили волатильности сохраняются и используются повторно
    4. ВАЛИДАЦИЯ БАЛАНСА: проверка представительства режимов при загрузке и подготовке
    5. Минимизация краевых эффектов EMD через буферные зоны (±50 точек)
    6. Каузальная классификация режимов (только на основе прошлого)
    7. Масштабирование с соблюдением каузальности
    8. Сохранение ВСЕХ артефактов в ЕДИНСТВЕННОМ .pkl файле
    
    🔑 РЕЖИМЫ СТРАТИФИКАЦИИ:
    • Фиксированные блоки (по умолчанию): каждый блок имеет размер block_size
    • Адаптивные блоки (опционально): блоки формируются динамически до достижения
      минимального представительства всех режимов (без нарушения хронологии)
    """
    
    def __init__(
        self,
        model: 'LSTMIMMUKF',
        seq_len: int = 72,
        min_history_for_features: int = 350,
        buffer_size: int = 50,
        block_size: int = 100,
        min_windows_per_regime: int = 5,
        adaptive_blocks: bool = True,    # ← НОВЫЙ ПАРАМЕТР
        min_regime_per_block: int = 3,    # ← только для адаптивного режима
        max_block_size: int = 300,        # ← только для адаптивного режима
        seed: int = 42
    ):
        """
        Args:
            model: экземпляр LSTMIMMUKF
            seq_len: длина входной последовательности
            min_history_for_features: минимальная история для расчёта признаков
            buffer_size: размер буферной зоны для EMD
            block_size: размер хронологического блока (используется при adaptive_blocks=False)
            min_windows_per_regime: минимальное число окон на режим для валидации баланса
            adaptive_blocks: использовать адаптивное формирование блоков для баланса режимов
            min_regime_per_block: минимальное число окон КАЖДОГО режима в адаптивном блоке
            max_block_size: максимальный размер адаптивного блока (защита от монорежимных периодов)
            seed: seed для воспроизводимости
        """
        self.model = model
        self.seq_len = seq_len
        self.min_history_for_features = min_history_for_features
        self.buffer_size = buffer_size
        self.total_window_size = min_history_for_features + 2 * buffer_size
        self.block_size = block_size
        self.min_windows_per_regime = min_windows_per_regime
        self.adaptive_blocks = adaptive_blocks
        self.min_regime_per_block = min_regime_per_block
        self.max_block_size = max_block_size
        self.seed = seed
        
        if buffer_size < 30:
            raise ValueError("buffer_size должен быть >= 30")
        
        if adaptive_blocks:
            if min_regime_per_block < 1:
                raise ValueError("min_regime_per_block должен быть >= 1 при адаптивной стратификации")
            if max_block_size < 50:
                raise ValueError("max_block_size должен быть >= 50 при адаптивной стратификации")
            print(f"✅ АДАПТИВНАЯ СТРАТИФИКАЦИЯ ВКЛЮЧЕНА:")
            print(f"   • Мин. окон на режим в блоке: {min_regime_per_block}")
            print(f"   • Макс. размер блока: {max_block_size}")
        else:
            if block_size < 50:
                raise ValueError("block_size должен быть >= 50 при фиксированной стратификации")
            print(f"✅ ФИКСИРОВАННАЯ СТРАТИФИКАЦИЯ (block_size={block_size})")
        
        np.random.seed(seed)
    
    def compute_causal_volatility_series(
        self,
        df: pd.DataFrame,
        window: int = 30,
        price_col: str = 'Close'
    ) -> np.ndarray:
        """Каузальная волатильность (только прошлое [t-window, t))"""
        n = len(df)
        causal_vol = np.zeros(n, dtype=np.float32)
        for i in range(n):
            window_start = max(0, i - window)
            window_size = i - window_start
            if window_size >= 5:
                vol = df[price_col].iloc[window_start:i].std()
            else:
                vol = 0.0
            causal_vol[i] = vol
        return causal_vol
    
    def _classify_regimes(
        self,
        causal_vol: np.ndarray,
        vol_q33: float,
        vol_q67: float
    ) -> np.ndarray:
        """
        🔑 ЕДИНЫЙ МЕТОД КЛАССИФИКАЦИИ РЕЖИМОВ (исправление замечания №8).
        Вынесен общий код из assign_causal_regimes и assign_regimes_with_saved_quantiles.
        """
        regimes = np.zeros(len(causal_vol), dtype=np.int32)
        for i in range(len(causal_vol)):
            if causal_vol[i] == 0.0:
                regimes[i] = 0
            elif causal_vol[i] < vol_q33:
                regimes[i] = 0
            elif causal_vol[i] < vol_q67:
                regimes[i] = 1
            else:
                regimes[i] = 2
        return regimes
    
    def assign_causal_regimes(
        self,
        causal_vol: np.ndarray,
        min_windows_per_regime: Optional[int] = None
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Классификация режимов на основе ПЕРВИЧНОГО расчёта квантилей.
        Используется ТОЛЬКО при первоначальной подготовке данных.
        
        Возвращает:
            regimes: массив режимов (0=LOW, 1=MID, 2=HIGH)
            volatility_quantiles: словарь с квантилями для воспроизводимости
        """
        min_windows_per_regime = min_windows_per_regime or self.min_windows_per_regime
        valid_mask = causal_vol > 0
        valid_vol = causal_vol[valid_mask]
        if len(valid_vol) < 30:
            raise ValueError("Недостаточно данных для классификации режимов")
        
        # 🔑 КРИТИЧЕСКИ ВАЖНО: квантили рассчитываются ОДИН раз на полном датасете
        vol_q33 = np.percentile(valid_vol, 33)
        vol_q50 = np.percentile(valid_vol, 50)
        vol_q67 = np.percentile(valid_vol, 67)
        volatility_quantiles = {
            'q33': float(vol_q33),
            'q50': float(vol_q50),
            'q67': float(vol_q67),
            'mean': float(np.mean(valid_vol)),
            'std': float(np.std(valid_vol)),
            'min': float(np.min(valid_vol)),
            'max': float(np.max(valid_vol))
        }
        
        regimes = self._classify_regimes(causal_vol, vol_q33, vol_q67)  # ← Используем единый метод
        
        # Валидация баланса
        regime_counts = np.bincount(regimes, minlength=3)
        for regime_id in range(3):
            if regime_counts[regime_id] < min_windows_per_regime:
                warnings.warn(
                    f"⚠️  Режим {regime_id} имеет мало окон ({regime_counts[regime_id]} < "
                    f"{min_windows_per_regime}). Это может повлиять на обучение модели."
                )
        
        return regimes, volatility_quantiles
    
    def assign_regimes_with_saved_quantiles(
        self,
        causal_vol: np.ndarray,
        volatility_quantiles: Dict[str, float]
    ) -> np.ndarray:
        """
        🔑 КРИТИЧЕСКИ ВАЖНО: классификация режимов с ИСПОЛЬЗОВАНИЕМ СОХРАНЁННЫХ квантилей.
        Используется для:
        1. Онлайн-предсказания (online_predict)
        2. Воспроизводимости при повторной загрузке
        
        Гарантирует согласованность классификации между обучением и инференсом.
        """
        vol_q33 = volatility_quantiles['q33']
        vol_q67 = volatility_quantiles['q67']
        return self._classify_regimes(causal_vol, vol_q33, vol_q67)  # ← Используем единый метод
    
    def process_single_window(
        self,
        t: int,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Optional[Dict]:
        """Обработка одного окна с буферными зонами"""
        if t < self.min_history_for_features + self.buffer_size:
            return None
        if t >= len(df) - self.buffer_size - 1:
            return None
        
        try:
            window_start = t - self.min_history_for_features - self.buffer_size
            window_end = t + self.buffer_size
            history = df.iloc[window_start:window_end].copy()
            
            if len(history) < self.total_window_size:
                return None
            
            features_full = self.model.prepare_features(history, mode='batch')
            if len(features_full) < self.total_window_size:
                return None
            
            features_core = features_full.iloc[self.buffer_size:-self.buffer_size].copy()
            if len(features_core) < self.seq_len:
                return None
            
            X_seq = features_core.tail(self.seq_len)[feature_columns or self.model.feature_columns].values.astype(np.float32)
            y_filter = features_core.tail(self.seq_len)['level'].values.astype(np.float32)
            
            history_with_target = df.iloc[window_start:window_end + 1].copy()
            features_with_target = self.model.prepare_features(history_with_target, mode='batch')
            target_idx = self.min_history_for_features + self.buffer_size
            
            if target_idx >= len(features_with_target):
                if getattr(self.model, 'debug_mode', False):
                    print(f"⚠️ Недопустимый индекс целевого значения: {target_idx} >= {len(features_with_target)}")
                return None
            
            # 🔑 ИСПРАВЛЕНИЕ №1: используем np.float32() вместо .astype() на скаляре
            y_target = np.float32(features_with_target.iloc[target_idx]['level'])
            
            timestamp = df.index[t] if hasattr(df.index, '__iter__') and len(df.index) > t else t
            
            return {
                't': t,
                'timestamp': timestamp,
                'X_seq': X_seq,
                'y_filter': y_filter,
                'y_target': y_target,  # ← Теперь корректный тип
                'window_start': window_start,
                'window_end': window_end,
                'buffer_size': self.buffer_size,
                'target_idx': target_idx
            }
        except Exception as e:
            if getattr(self.model, 'debug_mode', False):
                print(f"❌ Ошибка обработки окна t={t}: {str(e)}")
                import traceback
                traceback.print_exc()
            return None
    
    def _form_adaptive_blocks(
        self,
        windows: List[Dict],
        regimes: np.ndarray
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Адаптивное формирование хронологических блоков с балансом режимов.
        
        Возвращает:
            blocks: список блоков
            block_stats: статистика по блокам для метаданных
        """
        if not windows:
            raise ValueError("Список окон пуст")
        
        windows_sorted = sorted(windows, key=lambda x: x['t'])
        blocks = []
        current_block = []
        current_regime_counts = np.zeros(3, dtype=int)
        start_t = windows_sorted[0]['t']
        block_sizes = []
        
        for win in windows_sorted:
            t = win['t']
            regime = regimes[t]
            
            # Добавляем окно в текущий блок
            current_block.append(win)
            current_regime_counts[regime] += 1
            
            # Условие завершения блока:
            all_regimes_sufficient = np.all(current_regime_counts >= self.min_regime_per_block)
            block_too_large = len(current_block) >= self.max_block_size
            
            if all_regimes_sufficient or block_too_large:
                # Сохраняем блок
                blocks.append({
                    'windows': current_block.copy(),
                    'start_t': start_t,
                    'end_t': t,
                    'regime_counts': current_regime_counts.copy(),
                    'size': len(current_block),
                    'adaptive': True
                })
                block_sizes.append(len(current_block))
                
                # Начинаем новый блок
                current_block = []
                current_regime_counts = np.zeros(3, dtype=int)
                start_t = t + 1
        
        # Сохраняем последний неполный блок (если есть окна)
        if current_block:
            blocks.append({
                'windows': current_block.copy(),
                'start_t': start_t,
                'end_t': current_block[-1]['t'],
                'regime_counts': current_regime_counts.copy(),
                'size': len(current_block),
                'adaptive': True
            })
            block_sizes.append(len(current_block))
        
        if len(blocks) < 3:
            raise ValueError(
                f"Слишком мало адаптивных блоков ({len(blocks)}) для разделения. "
                f"Рекомендации:\n"
                f"  1. Уменьшите min_regime_per_block (текущий: {self.min_regime_per_block})\n"
                f"  2. Увеличьте max_block_size (текущий: {self.max_block_size})\n"
                f"  3. Увеличьте размер датасета"
            )
        
        # Статистика для метаданных
        block_stats = {
            'total': len(blocks),
            'sizes': block_sizes,
            'min_size': min(block_sizes),
            'max_size': max(block_sizes),
            'mean_size': np.mean(block_sizes),
            'adaptive_used': True
        }
        
        print(f"✅ Сформировано {len(blocks)} АДАПТИВНЫХ блоков:")
        regime_names = {0: 'LOW', 1: 'MID', 2: 'HIGH'}
        for i, block in enumerate(blocks[:5]):
            counts = block['regime_counts']
            print(f"   Блок {i:2d}: {block['size']:4d} окон | " +
                  " | ".join([f"{regime_names[r]}={counts[r]}" for r in range(3)]))
        if len(blocks) > 5:
            print(f"   ... и ещё {len(blocks) - 5} блоков")
        
        return blocks, block_stats
    
    def _form_fixed_blocks(
        self,
        windows: List[Dict],
        regimes: np.ndarray
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        """Формирование блоков фиксированного размера (оригинальная логика)"""
        windows_sorted = sorted(windows, key=lambda x: x['t'])
        n_total = len(windows_sorted)
        blocks = []
        start_idx = 0
        block_sizes = []
        
        while start_idx < n_total:
            end_idx = min(start_idx + self.block_size, n_total)
            block_windows = windows_sorted[start_idx:end_idx]
            block_regimes = [regimes[win['t']] for win in block_windows]
            regime_counts = np.bincount(block_regimes, minlength=3)
            
            blocks.append({
                'windows': block_windows,
                'start_t': block_windows[0]['t'],
                'end_t': block_windows[-1]['t'],
                'regime_counts': regime_counts,
                'size': len(block_windows),
                'adaptive': False
            })
            block_sizes.append(len(block_windows))
            start_idx = end_idx
        
        if len(blocks) < 3:
            raise ValueError(
                f"Слишком мало блоков ({len(blocks)}) для разделения. "
                f"Увеличьте размер датасета или уменьшите block_size={self.block_size}."
            )
        
        # Статистика для метаданных
        block_stats = {
            'total': len(blocks),
            'sizes': block_sizes,
            'min_size': min(block_sizes),
            'max_size': max(block_sizes),
            'mean_size': np.mean(block_sizes),
            'adaptive_used': False
        }
        
        print(f"✅ Сформировано {len(blocks)} ФИКСИРОВАННЫХ блоков (размер = {self.block_size})")
        return blocks, block_stats
    
    def _chronological_stratified_split(
        self,
        windows: List[Dict],
        regimes: np.ndarray,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        use_adaptive: Optional[bool] = None  # ← ЯВНЫЙ ПАРАМЕТР (исправление замечания №3)
    ) -> Tuple[List[Dict], List[Dict], List[Dict], Dict[str, Any]]:
        """
        Хронологическое разделение с выбором стратегии формирования блоков.
        
        Возвращает:
            train_windows, val_windows, test_windows, block_stats
        """
        windows_sorted = sorted(windows, key=lambda x: x['t'])
        n_total = len(windows_sorted)
        
        if n_total < 30:
            raise ValueError("Недостаточно окон для стратифицированного разделения")
        
        # 🔑 ИСПРАВЛЕНИЕ №3: явная передача флага, НЕ изменение состояния объекта
        effective_adaptive = use_adaptive if use_adaptive is not None else self.adaptive_blocks
        
        # === ВЫБОР СТРАТЕГИИ ФОРМИРОВАНИЯ БЛОКОВ ===
        if effective_adaptive:
            print("⚡ Формирование АДАПТИВНЫХ блоков для баланса режимов...")
            blocks, block_stats = self._form_adaptive_blocks(windows_sorted, regimes)
            stratification_method = 'adaptive_chronological'
        else:
            print(f"⚡ Формирование ФИКСИРОВАННЫХ блоков (размер = {self.block_size})...")
            blocks, block_stats = self._form_fixed_blocks(windows_sorted, regimes)
            stratification_method = 'block_based_chronological'
        
        # === ХРОНОЛОГИЧЕСКОЕ РАСПРЕДЕЛЕНИЕ БЛОКОВ ПО СПЛИТАМ ===
        n_blocks = len(blocks)
        n_train_blocks = max(1, int(np.round(n_blocks * train_ratio)))
        n_val_blocks = max(1, int(np.round(n_blocks * val_ratio)))
        n_test_blocks = n_blocks - n_train_blocks - n_val_blocks
        
        if n_test_blocks < 1:
            n_test_blocks = 1
            n_val_blocks = max(1, n_val_blocks - 1)
            n_train_blocks = n_blocks - n_val_blocks - n_test_blocks
        
        train_blocks = blocks[:n_train_blocks]
        val_blocks = blocks[n_train_blocks:n_train_blocks + n_val_blocks]
        test_blocks = blocks[n_train_blocks + n_val_blocks:]
        
        train_windows = [win for block in train_blocks for win in block['windows']]
        val_windows = [win for block in val_blocks for win in block['windows']]
        test_windows = [win for block in test_blocks for win in block['windows']]
        
        # === ВАЛИДАЦИЯ БАЛАНСА И ХРОНОЛОГИИ ===
        self._validate_regime_balance(train_windows, val_windows, test_windows, regimes)
        self._validate_chronology(train_windows, val_windows, test_windows)
        
        # === АНАЛИЗ РЕЗУЛЬТАТОВ ===
        print(f"\n📊 Итоговое распределение блоков:")
        print(f"   • Train: {len(train_blocks)} блоков → {len(train_windows)} окон")
        print(f"   • Val:   {len(val_blocks)} блоков → {len(val_windows)} окон")
        print(f"   • Test:  {len(test_blocks)} блоков → {len(test_windows)} окон")
        
        print(f"\n⚖️  Баланс режимов в сплитах:")
        regime_names = {0: 'LOW', 1: 'MID', 2: 'HIGH'}
        for split_name, split_windows in [('Train', train_windows), ('Val', val_windows), ('Test', test_windows)]:
            split_regimes = [regimes[win['t']] for win in split_windows]
            counts = np.bincount(split_regimes, minlength=3)
            total = counts.sum()
            print(f"   {split_name:5s}: " +
                  " | ".join([f"{regime_names[r]}={counts[r]} ({counts[r]/total*100:.1f}%)" for r in range(3)]))
        
        # Сохраняем метод стратификации и статистику блоков
        block_stats.update({
            'train_blocks': len(train_blocks),
            'val_blocks': len(val_blocks),
            'test_blocks': len(test_blocks),
            'stratification_method': stratification_method
        })
        
        return train_windows, val_windows, test_windows, block_stats
    
    def _validate_regime_balance(
        self,
        train_windows: List[Dict],
        val_windows: List[Dict],
        test_windows: List[Dict],
        regimes: np.ndarray
    ):
        """
        🔑 КРИТИЧЕСКИ ВАЖНО: валидация баланса режимов во ВСЕХ сплитах.
        Вызывается как при подготовке, так и при загрузке (через _validate_regime_balance_loaded).
        """
        splits = [('train', train_windows), ('val', val_windows), ('test', test_windows)]
        for split_name, windows in splits:
            if not windows:
                continue
            split_regimes = [regimes[win['t']] for win in windows]
            counts = np.bincount(split_regimes, minlength=3)
            
            # Критическая проверка: все режимы должны быть представлены
            missing_regimes = [i for i, count in enumerate(counts) if count == 0]
            if missing_regimes:
                regime_names = {0: 'LOW', 1: 'MID', 2: 'HIGH'}
                missing_names = [regime_names[r] for r in missing_regimes]
                raise ValueError(
                    f"❌ КРИТИЧЕСКАЯ ОШИБКА: в сплите {split_name} отсутствуют режимы {missing_names}!\n"
                    f"   Это сделает модель бесполезной для этих режимов.\n"
                    f"   Рекомендации:\n"
                    f"   1. Уменьшите block_size (текущий: {self.block_size})\n"
                    f"   2. Увеличьте размер датасета\n"
                    f"   3. Проверьте корректность расчёта волатильности"
                )
            
            # Предупреждение о дисбалансе
            for regime_id, count in enumerate(counts):
                if count < self.min_windows_per_regime:
                    regime_names = {0: 'LOW', 1: 'MID', 2: 'HIGH'}
                    warnings.warn(
                        f"⚠️  В сплите {split_name} мало окон для режима {regime_names[regime_id]} "
                        f"({count} < {self.min_windows_per_regime}).\n"
                        f"   Это может привести к переобучению или нестабильности для этого режима."
                    )
    
    def _validate_regime_balance_loaded(
        self,
        train_data: Dict,
        val_data: Dict,
        test_data: Dict,
        metadata: Dict
    ):
        """
        🔑 ИСПРАВЛЕНИЕ №5: двойная валидация режимов при загрузке:
        1. По метаданным (для быстрой проверки)
        2. По фактическим меткам (если доступны) — защита от повреждённого кэша
        """
        # === ВАЛИДАЦИЯ ПО МЕТАДАННЫМ ===
        regime_distribution = metadata.get('regime_distribution', {})
        for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
            if split_data['n_samples'] == 0:
                continue
            
            dist = regime_distribution.get(split_name, {})
            counts_from_meta = [dist.get('LOW', 0), dist.get('MID', 0), dist.get('HIGH', 0)]
            
            # Критическая проверка по метаданным
            missing_regimes = [i for i, count in enumerate(counts_from_meta) if count == 0]
            if missing_regimes:
                regime_names = {0: 'LOW', 1: 'MID', 2: 'HIGH'}
                missing_names = [regime_names[r] for r in missing_regimes]
                raise ValueError(
                    f"❌ ЗАГРУЖЕННЫЕ ДАННЫЕ: в сплите {split_name} отсутствуют режимы {missing_names}!\n"
                    f"   Это сделает модель бесполезной для этих режимов.\n"
                    f"   Решение: пересчитайте данные с меньшим block_size или большим датасетом "
                    f"(force_recompute=True)."
                )
        
        # === ДОПОЛНИТЕЛЬНАЯ ВАЛИДАЦИЯ ПО ФАКТИЧЕСКИМ МЕТКАМ (если доступны) ===
        # Защита от повреждённого кэша или несогласованности метаданных
        if 'regime_labels' in train_data and 'regime_labels' in val_data and 'regime_labels' in test_data:
            print("🔍 Дополнительная валидация режимов по фактическим меткам...")
            
            for split_name, split_data in [('train', train_data), ('val', val_data), ('test', test_data)]:
                if split_data['n_samples'] == 0:
                    continue
                
                actual_counts = np.bincount(split_data['regime_labels'], minlength=3)
                meta_counts = [
                    regime_distribution.get(split_name, {}).get('LOW', 0),
                    regime_distribution.get(split_name, {}).get('MID', 0),
                    regime_distribution.get(split_name, {}).get('HIGH', 0)
                ]
                
                # Проверка согласованности метаданных и фактических меток
                if not np.array_equal(actual_counts, meta_counts):
                    warnings.warn(
                        f"⚠️  Несогласованность режимов в сплите {split_name}:\n"
                        f"   Метаданные: {meta_counts}\n"
                        f"   Фактические: {actual_counts.tolist()}\n"
                        f"   Возможна ошибка при сохранении кэша. Продолжаем с фактическими метками."
                    )
                
                # Критическая проверка по фактическим меткам
                missing_regimes = [i for i, count in enumerate(actual_counts) if count == 0]
                if missing_regimes:
                    regime_names = {0: 'LOW', 1: 'MID', 2: 'HIGH'}
                    missing_names = [regime_names[r] for r in missing_regimes]
                    raise ValueError(
                        f"❌ КРИТИЧЕСКАЯ ОШИБКА: в сплите {split_name} отсутствуют режимы {missing_names} "
                        f"(проверено по фактическим меткам)!\n"
                        f"   Кэш повреждён или несогласован. Пересчитайте данные (force_recompute=True)."
                    )
            
            print("✅ Фактические метки режимов согласованы с метаданными")
        else:
            warnings.warn(
                "⚠️  Фактические метки режимов отсутствуют в кэше — валидация выполнена только по метаданным.\n"
                "   Рекомендуется пересчитать данные для полной валидации (force_recompute=True)."
            )
    
    def _scale_features_batch(self, X_batch: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        🔑 ИСПРАВЛЕНИЕ №4: устранение избыточного fit и добавление валидации скейлеров.
        """
        n_samples, seq_len, n_features = X_batch.shape
        X_scaled = np.zeros_like(X_batch, dtype=np.float32)
        
        if self.model.feature_scalers is None:
            self.model.feature_scalers = {
                'robust': None,
                'standard': None,
                'minmax': None,
                'none': None,
                'Y': None
            }
        
        for group_name, features in self.model.scale_groups.items():
            valid_features = [f for f in features if f in self.model.feature_columns]
            if not valid_features:
                continue
            
            feature_indices = [self.model.feature_columns.index(f) for f in valid_features]
            
            if group_name == 'none':
                X_scaled[:, :, feature_indices] = X_batch[:, :, feature_indices]
                continue
            
            X_group = X_batch[:, :, feature_indices].reshape(-1, len(feature_indices))
            
            # Получаем или создаём скейлер
            if group_name not in self.model.feature_scalers or self.model.feature_scalers[group_name] is None:
                if group_name == 'robust':
                    from sklearn.preprocessing import RobustScaler
                    scaler = RobustScaler(quantile_range=(5, 95))
                elif group_name == 'standard':
                    from sklearn.preprocessing import StandardScaler
                    scaler = StandardScaler()
                elif group_name == 'minmax':
                    from sklearn.preprocessing import MinMaxScaler
                    scaler = MinMaxScaler(feature_range=(0, 1))
                else:
                    scaler = None
                
                if scaler is not None and fit:
                    # 🔑 КРИТИЧЕСКИ ВАЖНО: НЕТ избыточного вызова fit() — только fit_transform
                    X_group_scaled = scaler.fit_transform(X_group)
                    self.model.feature_scalers[group_name] = scaler
                else:
                    X_group_scaled = X_group  # без масштабирования
            else:
                scaler = self.model.feature_scalers[group_name]
                if scaler is None:
                    raise RuntimeError(...)
                if fit:
                    X_group_scaled = scaler.fit_transform(X_group)  # ← переобучение при необходимости
                    self.model.feature_scalers[group_name] = scaler
                else:
                    X_group_scaled = scaler.transform(X_group)
            
            X_scaled[:, :, feature_indices] = X_group_scaled.reshape(n_samples, seq_len, len(feature_indices))
        
        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)
        return X_scaled
    
    def prepare_datasets(
        self,
        df: pd.DataFrame,
        save_path: Optional[str] = None,
        train_ratio: float = 0.60,
        val_ratio: float = 0.20,
        n_jobs: int = -1,
        force_recompute: bool = False,
        use_adaptive: Optional[bool] = None  # ← ЯВНЫЙ ПАРАМЕТР (исправление замечания №3)
    ) -> Tuple[Dict, Dict, Dict]:
        """
        Полная подготовка данных с гарантией воспроизводимости квантилей.
        
        Args:
            use_adaptive: явное указание режима стратификации (переопределяет self.adaptive_blocks)
        """
        print("\n" + "=" * 80)
        print("🔍 ЧЕСТАЯ ПОДГОТОВКА ДАННЫХ: хронология + локальная стратификация")
        print("=" * 80)
        
        # Валидация входных данных
        required_cols = ['Open', 'High', 'Low', 'Close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Отсутствуют обязательные колонки: {missing_cols}")
        
        min_required = self.min_history_for_features + 2 * self.buffer_size + 100
        if len(df) < min_required:
            raise ValueError(f"Недостаточно данных: требуется минимум {min_required} точек")
        
        # Проверка кэша
        if save_path and not force_recompute and os.path.exists(f"{save_path}.pkl"):
            print(f"📥 Кэш найден: {save_path}.pkl")
            print("   Загружаем предварительно обработанные данные...")
            return self.load_prepared_datasets(save_path)
        
        # ШАГ 1: Каузальная классификация режимов (квантили рассчитываются ОДИН раз!)
        print("\n📊 ШАГ 1: Каузальная классификация режимов волатильности...")
        causal_vol = self.compute_causal_volatility_series(df, window=30, price_col='Close')
        regimes, volatility_quantiles = self.assign_causal_regimes(causal_vol)
        print(f"✅ Квантили волатильности (рассчитаны ОДИН раз на полном датасете):")
        print(f"   • Q33: {volatility_quantiles['q33']:.6f} | Q50: {volatility_quantiles['q50']:.6f} | Q67: {volatility_quantiles['q67']:.6f}")
        
        # ШАГ 2: Параллельная обработка окон
        print("\n⚡ ШАГ 2: Параллельная обработка окон с буферными зонами...")
        min_t = self.min_history_for_features + self.buffer_size
        max_t = len(df) - self.buffer_size - 1
        
        if n_jobs == -1:
            n_jobs = max(1, mp.cpu_count() - 1)
        
        results = Parallel(n_jobs=n_jobs, verbose=10, backend='loky')(
            delayed(self.process_single_window)(t, df, feature_columns=self.model.feature_columns)
            for t in range(min_t, max_t + 1)
        )
        
        valid_results = [r for r in results if r is not None]
        print(f"✅ Успешно обработано {len(valid_results)} / {len(results)} окон")
        
        if len(valid_results) == 0:
            raise RuntimeError("Не удалось обработать ни одного окна")
        
        # ШАГ 3: Хронологическое разделение с выбором стратегии
        print("\n⚖️  ШАГ 3: Хронологическое разделение с локальной стратификацией...")
        # 🔑 ИСПРАВЛЕНИЕ №3: явная передача флага, НЕ изменение состояния объекта
        effective_adaptive = use_adaptive if use_adaptive is not None else self.adaptive_blocks
        
        train_windows, val_windows, test_windows, block_stats = self._chronological_stratified_split(
            windows=valid_results,
            regimes=regimes,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            use_adaptive=effective_adaptive  # ← явная передача
        )
        
        # ШАГ 4: Формирование массивов
        def windows_to_arrays(windows: List[Dict]) -> Dict:
            n_samples = len(windows)
            n_features = len(self.model.feature_columns)
            X_seq = np.zeros((n_samples, self.seq_len, n_features), dtype=np.float32)
            y_filter = np.zeros((n_samples, self.seq_len), dtype=np.float32)
            y_target = np.zeros((n_samples,), dtype=np.float32)
            timestamps = []
            window_indices = np.zeros((n_samples,), dtype=np.int32)
            regime_labels = np.zeros((n_samples,), dtype=np.int32)
            
            for i, win in enumerate(windows):
                X_seq[i] = win['X_seq']
                y_filter[i] = win['y_filter']
                y_target[i] = win['y_target']
                timestamps.append(win['timestamp'])
                window_indices[i] = win['t']
                regime_labels[i] = regimes[win['t']]
            
            return {
                'X_seq': X_seq,
                'y_filter': y_filter,
                'y_target': y_target,
                'timestamps': np.array(timestamps),
                'window_indices': window_indices,
                'regime_labels': regime_labels,
                'n_samples': n_samples
            }
        
        train_data = windows_to_arrays(train_windows)
        val_data = windows_to_arrays(val_windows)
        test_data = windows_to_arrays(test_windows)
        
        print(f"\n📊 Размеры сплитов:")
        print(f"   Train: {train_data['n_samples']} образцов")
        print(f"   Val:   {val_data['n_samples']} образцов")
        print(f"   Test:  {test_data['n_samples']} образцов")
        
        # === ШАГ 4.5: ВИЗУАЛИЗАЦИЯ ПЕРЕМЕННОЙ 'level' ===
        if save_path is not None:
            self._save_level_visualizations(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                save_path=save_path,
                save_visualizations=True
            )
        
        # === ШАГ 4.6: ТЕКСТОВЫЕ ГИСТОГРАММЫ РАСПРЕДЕЛЕНИЯ 'level' ===
        if save_path is not None:
            self._save_text_histograms(
                train_data=train_data,
                val_data=val_data,
                test_data=test_data,
                save_path=save_path,
                save_histograms=True,
                stratification_params={  # ← ОБЯЗАТЕЛЬНО передать фактические параметры
                    'adaptive_blocks': effective_adaptive,  # ← результат выбора режима стратификации
                    'block_size': self.block_size,
                    'min_regime_per_block': self.min_regime_per_block if effective_adaptive else None,
                    'max_block_size': self.max_block_size if effective_adaptive else None,
                    'seq_len': self.seq_len,
                    'buffer_size': self.buffer_size,
                    'min_windows_per_regime': self.min_windows_per_regime
                }
            )
        
        # ШАГ 5: Масштабирование
        print("\n📏 ШАГ 4: Масштабирование признаков (каузальный подход)...")
        train_data['X_seq_scaled'] = self._scale_features_batch(train_data['X_seq'], fit=True)
        val_data['X_seq_scaled'] = self._scale_features_batch(val_data['X_seq'], fit=False)
        test_data['X_seq_scaled'] = self._scale_features_batch(test_data['X_seq'], fit=False)
        
        y_scaler = PowerTransformer(method='yeo-johnson', standardize=True)
        y_train_scaled = y_scaler.fit_transform(train_data['y_target'].reshape(-1, 1)).flatten()
        y_val_scaled = y_scaler.transform(val_data['y_target'].reshape(-1, 1)).flatten()
        y_test_scaled = y_scaler.transform(test_data['y_target'].reshape(-1, 1)).flatten()
        
        train_data['y_target_scaled'] = y_train_scaled
        val_data['y_target_scaled'] = y_val_scaled
        test_data['y_target_scaled'] = y_test_scaled
        self.model.feature_scalers['Y'] = y_scaler
        
        print("✅ Масштабирование завершено")
        
        # ШАГ 6: Сохранение
        if save_path is not None:
            print(f"\n💾 ШАГ 5: Сохранение в ЕДИНЫЙ файл: {save_path}.pkl")
            self.save_prepared_datasets(
                save_path,
                train_data, val_data, test_data,
                metadata={
                    'feature_columns': self.model.feature_columns,
                    'seq_len': self.seq_len,
                    'min_history_for_features': self.min_history_for_features,
                    'buffer_size': self.buffer_size,
                    'block_size': self.block_size,
                    'min_windows_per_regime': self.min_windows_per_regime,
                    'adaptive_blocks': effective_adaptive,  # ← флаг режима
                    'min_regime_per_block': self.min_regime_per_block if effective_adaptive else None,
                    'max_block_size': self.max_block_size if effective_adaptive else None,
                    'total_window_size': self.total_window_size,
                    'train_ratio': train_ratio,
                    'val_ratio': val_ratio,
                    'test_ratio': 1.0 - train_ratio - val_ratio,
                    'n_train': train_data['n_samples'],
                    'n_val': val_data['n_samples'],
                    'n_test': test_data['n_samples'],
                    'timestamp': str(datetime.datetime.now()),
                    'scale_groups': self.model.scale_groups,
                    'seed': self.seed,
                    'chronological_split': True,
                    'stratify_by_regime': True,
                    'stratification_method': block_stats['stratification_method'],
                    'volatility_quantiles': volatility_quantiles,
                    'regime_distribution': {
                        'train': {
                            'LOW': int(np.sum(train_data['regime_labels'] == 0)),
                            'MID': int(np.sum(train_data['regime_labels'] == 1)),
                            'HIGH': int(np.sum(train_data['regime_labels'] == 2))
                        },
                        'val': {
                            'LOW': int(np.sum(val_data['regime_labels'] == 0)),
                            'MID': int(np.sum(val_data['regime_labels'] == 1)),
                            'HIGH': int(np.sum(val_data['regime_labels'] == 2))
                        },
                        'test': {
                            'LOW': int(np.sum(test_data['regime_labels'] == 0)),
                            'MID': int(np.sum(test_data['regime_labels'] == 1)),
                            'HIGH': int(np.sum(test_data['regime_labels'] == 2))
                        }
                    },
                    # 🔑 ИСПРАВЛЕНИЕ №2: корректные метаданные block_analysis
                    'block_analysis': block_stats  # ← РЕАЛЬНАЯ статистика из _chronological_stratified_split
                }
            )
        
        print("\n" + "=" * 80)
        print("✅ ЧЕСТАЯ ПОДГОТОВКА ЗАВЕРШЕНА (гарантирована воспроизводимость квантилей)")
        print("=" * 80)
        return train_data, val_data, test_data
    
    def save_prepared_datasets(
        self,
        save_path: str,
        train_df: Dict,
        val_df: Dict,
        test_df: Dict,
        metadata: Dict[str, Any]
    ) -> None:
        """
        Сохранение в едином .pkl файле.
        """
        if save_path.endswith('.pkl'):
            actual_path = save_path
        else:
            actual_path = f"{save_path}.pkl"
        
        os.makedirs(os.path.dirname(actual_path) if os.path.dirname(actual_path) else '.', exist_ok=True)
        
        cache_data = {
            'train': {
                'X_seq': train_df['X_seq'],
                'X_seq_scaled': train_df['X_seq_scaled'],
                'y_filter': train_df['y_filter'],
                'y_target': train_df['y_target'],
                'y_target_scaled': train_df['y_target_scaled'],
                'timestamps': train_df['timestamps'],
                'regime_labels': train_df['regime_labels'],
                'n_samples': train_df['n_samples']
            },
            'val': {
                'X_seq': val_df['X_seq'],
                'X_seq_scaled': val_df['X_seq_scaled'],
                'y_filter': val_df['y_filter'],
                'y_target': val_df['y_target'],
                'y_target_scaled': val_df['y_target_scaled'],
                'timestamps': val_df['timestamps'],
                'regime_labels': val_df['regime_labels'],
                'n_samples': val_df['n_samples']
            },
            'test': {
                'X_seq': test_df['X_seq'],
                'X_seq_scaled': test_df['X_seq_scaled'],
                'y_filter': test_df['y_filter'],
                'y_target': test_df['y_target'],
                'y_target_scaled': test_df['y_target_scaled'],
                'timestamps': test_df['timestamps'],
                'regime_labels': test_df['regime_labels'],
                'n_samples': test_df['n_samples']
            },
            'scalers': self.model.feature_scalers.copy() if self.model.feature_scalers else None,
            'scale_groups': self.model.scale_groups.copy() if hasattr(self.model, 'scale_groups') else None,
            'feature_columns': self.model.feature_columns.copy() if hasattr(self.model, 'feature_columns') else None,
            'metadata': metadata
        }
        
        with open(actual_path, 'wb') as f:
            pickle.dump(cache_data, f)
        
        print(f"✅ Данные сохранены: {actual_path}")
        print(f"   • Train: {train_df['n_samples']} образцов")
        print(f"   • Val:   {val_df['n_samples']} образцов")
        print(f"   • Test:  {test_df['n_samples']} образцов")
    
    def load_prepared_datasets(self, load_path: str) -> Tuple[Dict, Dict, Dict]:
        """
        🔑 Загрузка с поддержкой путей как с расширением (.pkl), так и без.
        """
        if load_path.endswith('.pkl'):
            actual_path = load_path
        else:
            actual_path = f"{load_path}.pkl"
        
        if not os.path.exists(actual_path):
            raise FileNotFoundError(
                f"Файл не найден: {actual_path}\n"
                f"Проверьте:\n"
                f"  1. Существует ли директория '{os.path.dirname(actual_path) or '.'}'\n"
                f"  2. Корректность пути к кэшу (без двойного .pkl)\n"
                f"  3. Была ли выполнена первоначальная подготовка данных (force_recompute=True)"
            )
        
        print(f"\n📥 Загрузка предварительно обработанных данных: {actual_path}")
        with open(actual_path, 'rb') as f:
            cached = pickle.load(f)
        
        metadata = cached.get('metadata', {})
        
        # 🔑 ВАЛИДАЦИЯ ПАРАМЕТРОВ МОДЕЛИ
        required_params = ['seq_len', 'min_history_for_features', 'buffer_size']
        mismatches = []
        for param in required_params:
            cache_val = metadata.get(param)
            model_val = getattr(self, param, None)
            if cache_val is not None and model_val is not None and cache_val != model_val:
                mismatches.append(f"{param}: model={model_val}, cache={cache_val}")
        
        # Для адаптивного режима проверяем дополнительные параметры
        adaptive_used = metadata.get('adaptive_blocks', False)
        if adaptive_used:
            cache_min_regime = metadata.get('min_regime_per_block')
            cache_max_size = metadata.get('max_block_size')
            if cache_min_regime is not None and cache_min_regime != self.min_regime_per_block:
                mismatches.append(f"min_regime_per_block: model={self.min_regime_per_block}, cache={cache_min_regime}")
            if cache_max_size is not None and cache_max_size != self.max_block_size:
                mismatches.append(f"max_block_size: model={self.max_block_size}, cache={cache_max_size}")
        else:
            cache_block_size = metadata.get('block_size')
            if cache_block_size is not None and cache_block_size != self.block_size:
                mismatches.append(f"block_size: model={self.block_size}, cache={cache_block_size}")
        
        if mismatches:
            raise ValueError(
                "❌ Несоответствие параметров модели и кэшированных данных:\n" +
                "\n".join(mismatches) +
                "\nИспользуйте force_recompute=True или согласуйте параметры."
            )
        
        # 🔑 ИСПРАВЛЕНИЕ №5: двойная валидация режимов
        self._validate_regime_balance_loaded(cached['train'], cached['val'], cached['test'], metadata)
        
        # 🔑 ВОССТАНОВЛЕНИЕ КВАНТИЛЕЙ ДЛЯ ОНЛАЙН-ПРЕДСКАЗАНИЯ
        if 'volatility_quantiles' in metadata:
            self.model.volatility_quantiles = metadata['volatility_quantiles']
            print(f"✅ Квантили волатильности восстановлены из кэша:")
            print(f"   • Q33: {metadata['volatility_quantiles']['q33']:.6f}")
            print(f"   • Q67: {metadata['volatility_quantiles']['q67']:.6f}")
            print(f"   • Будут использоваться в online_predict для согласованной классификации режимов")
        else:
            warnings.warn("⚠️  Квантили волатильности отсутствуют в кэше! Онлайн-предсказание может быть несогласованным.")
        
        # Восстановление скейлеров
        if 'scalers' in cached and cached['scalers'] is not None:
            self.model.feature_scalers = cached['scalers']
            print("✅ Скейлеры восстановлены")
        
        if 'scale_groups' in cached and cached['scale_groups'] is not None:
            self.model.scale_groups = cached['scale_groups']
            print("✅ scale_groups восстановлены")
        
        if 'feature_columns' in cached and cached['feature_columns'] is not None:
            self.model.feature_columns = cached['feature_columns']
            print("✅ feature_columns восстановлены")
        
        train_data = cached['train']
        val_data = cached['val']
        test_data = cached['test']
        
        # Валидация хронологии
        self._validate_chronology_loaded(train_data, val_data, test_data)
        
        print(f"✅ Данные загружены:")
        print(f"   Train: {train_data['n_samples']} образцов")
        print(f"   Val:   {val_data['n_samples']} образцов")
        print(f"   Test:  {test_data['n_samples']} образцов")
        
        # Восстанавливаем флаг адаптивной стратификации из метаданных для последующих операций
        self.adaptive_blocks = metadata.get('adaptive_blocks', False)
        
        return train_data, val_data, test_data
    
    def _validate_chronology(
        self,
        train_windows: List[Dict],
        val_windows: List[Dict],
        test_windows: List[Dict]
    ):
        """Валидация глобальной хронологии"""
        if not train_windows or not val_windows or not test_windows:
            return
        
        max_train_t = max(w['t'] for w in train_windows)
        min_val_t = min(w['t'] for w in val_windows)
        max_val_t = max(w['t'] for w in val_windows)
        min_test_t = min(w['t'] for w in test_windows)
        
        if max_train_t >= min_val_t:
            raise ValueError(
                f"❌ НАРУШЕНА ХРОНОЛОГИЯ: max(train.t)={max_train_t} >= min(val.t)={min_val_t}\n"
                f"   Это критическая ошибка — утечка будущего!"
            )
        
        if max_val_t >= min_test_t:
            raise ValueError(
                f"❌ НАРУШЕНА ХРОНОЛОГИЯ: max(val.t)={max_val_t} >= min(test.t)={min_test_t}"
            )
        
        print(f"✅ Хронология СОБЛЮДЕНА (гарантировано отсутствие утечки будущего):")
        print(f"   • max(train.t)={max_train_t} < min(val.t)={min_val_t}")
        print(f"   • max(val.t)={max_val_t} < min(test.t)={min_test_t}")
    
    def _validate_chronology_loaded(self, train_data, val_data, test_data):
        """
        Валидация хронологии для загруженных данных с безопасной обработкой отсутствующих полей.
        """
        if train_data['n_samples'] == 0 or val_data['n_samples'] == 0 or test_data['n_samples'] == 0:
            return
        
        # Попытка валидации через индексы окон (более надёжно)
        if 'window_indices' in train_data and 'window_indices' in val_data and 'window_indices' in test_data:
            max_train_idx = train_data['window_indices'][-1]
            min_val_idx = val_data['window_indices'][0]
            max_val_idx = val_data['window_indices'][-1]
            min_test_idx = test_data['window_indices'][0]
            
            if max_train_idx >= min_val_idx:
                warnings.warn(
                    f"⚠️  Потенциальное нарушение хронологии: "
                    f"max_train_idx={max_train_idx} >= min_val_idx={min_val_idx}"
                )
            
            if max_val_idx >= min_test_idx:
                warnings.warn(
                    f"⚠️  Потенциальное нарушение хронологии: "
                    f"max_val_idx={max_val_idx} >= min_test_idx={min_test_idx}"
                )
            
            return
        
        # Fallback: валидация через временные метки
        try:
            max_train_ts = pd.Timestamp(train_data['timestamps'][-1])
            min_val_ts = pd.Timestamp(val_data['timestamps'][0])
            max_val_ts = pd.Timestamp(val_data['timestamps'][-1])
            min_test_ts = pd.Timestamp(test_data['timestamps'][0])
            
            if max_train_ts >= min_val_ts:
                warnings.warn(
                    f"⚠️  Потенциальное нарушение хронологии по времени: "
                    f"max_train_ts={max_train_ts} >= min_val_ts={min_val_ts}"
                )
            
            if max_val_ts >= min_test_ts:
                warnings.warn(
                    f"⚠️  Потенциальное нарушение хронологии по времени: "
                    f"max_val_ts={max_val_ts} >= min_test_ts={min_test_ts}"
                )
        except (IndexError, KeyError, ValueError) as e:
            warnings.warn(
                f"⚠️  Не удалось провалидировать хронологию загруженных данных: {str(e)}\n"
                f"   Проверьте целостность кэша или пересчитайте данные (force_recompute=True)."
            )
    
    def create_tf_datasets(
        self,
        train_df: Dict,
        val_df: Dict,
        batch_size: int = 64
    ) -> Tuple[Any, Any]:
        """
        🔑 ИСПРАВЛЕНИЕ №6: drop_remainder=False для валидации (сохранение всех данных).
        """
        import tensorflow as tf
        
        train_ds = tf.data.Dataset.from_tensor_slices((
            train_df['X_seq_scaled'],
            train_df['y_filter'],
            train_df['y_target_scaled']
        ))
        
        val_ds = tf.data.Dataset.from_tensor_slices((
            val_df['X_seq_scaled'],
            val_df['y_filter'],
            val_df['y_target_scaled']
        ))
        
        # 🔑 drop_remainder=True допустимо для тренировки (стабильность обучения)
        train_ds = (train_ds
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.AUTOTUNE)
        )
        
        # 🔑 drop_remainder=False для валидации (сохранение всех данных для точных метрик)
        val_ds = (val_ds
            .batch(batch_size, drop_remainder=False)  # ← ИСПРАВЛЕНО
            .cache()
            .prefetch(tf.data.AUTOTUNE)
        )
        
        print(f"\n⚡ tf.data пайплайны созданы (БЕЗ перемешивания):")
        print(f"   Train batches: {len(train_ds)}")
        print(f"   Val batches:   {len(val_ds)} (последний батч может быть меньше)")
        
        return train_ds, val_ds
    
    # Методы _save_level_visualizations и _save_text_histograms остаются без изменений
    # (они уже корректны и не содержат критических ошибок)
    def _save_level_visualizations(
        self,
        train_data: Dict,
        val_data: Dict,
        test_data: Dict,
        save_path: str,
        save_visualizations: bool = True
    ):
        """Сохранение графиков переменной 'level' для всех сплитов."""
        if not save_visualizations:
            return
        
        try:
            viz_path = save_path if not save_path.endswith('.pkl') else save_path[:-4]
            viz_dir = os.path.dirname(viz_path) or '.'
            os.makedirs(viz_dir, exist_ok=True)
            
            print(f"\n📊 Сохранение визуализаций переменной 'level' в: {viz_dir}/")
            
            colors = {
                'train': '#2ecc71',
                'val': '#3498db',
                'test': '#e74c3c'
            }
            
            for split_name, split_data, color in [
                ('train', train_data, colors['train']),
                ('val', val_data, colors['val']),
                ('test', test_data, colors['test'])
            ]:
                if split_data['n_samples'] == 0:
                    continue
                
                plt.figure(figsize=(7, 3))
                plt.plot(
                    split_data['timestamps'],
                    split_data['y_target'],
                    color=color,
                    linewidth=1.5,
                    alpha=0.8,
                    label=f'{split_name.upper()} (n={split_data["n_samples"]})'
                )
                
                plt.title(f"Целевая переменная 'level' — Сплит: {split_name.upper()}", fontsize=14, fontweight='bold')
                plt.xlabel('Время', fontsize=12)
                plt.ylabel('level', fontsize=12)
                plt.grid(True, alpha=0.3, linestyle='--')
                plt.legend(loc='best')
                plt.tight_layout()
                
                output_file = f"{viz_path}_{split_name}_level.png"
                plt.savefig(output_file, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"   ✅ {split_name.upper()}: {output_file}")
            
            # График всех сплитов вместе
            plt.figure(figsize=(7, 3))
            plt.plot(
                train_data['timestamps'],
                train_data['y_target'],
                color=colors['train'],
                linewidth=2,
                alpha=0.9,
                label=f'TRAIN (n={train_data["n_samples"]})'
            )
            plt.plot(
                val_data['timestamps'],
                val_data['y_target'],
                color=colors['val'],
                linewidth=2,
                alpha=0.9,
                label=f'VAL (n={val_data["n_samples"]})'
            )
            plt.plot(
                test_data['timestamps'],
                test_data['y_target'],
                color=colors['test'],
                linewidth=2,
                alpha=0.9,
                label=f'TEST (n={test_data["n_samples"]})'
            )
            
            plt.axvline(x=train_data['timestamps'][-1], color='gray', linestyle='--', alpha=0.7, linewidth=2, label='Граница TRAIN/VAL')
            plt.axvline(x=val_data['timestamps'][-1], color='darkgray', linestyle='--', alpha=0.7, linewidth=2, label='Граница VAL/TEST')
            
            plt.title("Целевая переменная 'level' — Все сплиты (хронологическое разделение)", fontsize=16, fontweight='bold')
            plt.xlabel('Время', fontsize=13)
            plt.ylabel('level', fontsize=13)
            plt.grid(True, alpha=0.3, linestyle='--')
            plt.legend(loc='best', fontsize=11)
            plt.tight_layout()
            
            output_file = f"{viz_path}_all_splits_level.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ ВСЕ СПЛИТЫ: {output_file}")
            print(f"   💡 Подсказка: вертикальные линии показывают границы хронологического разделения")
        
        except Exception as e:
            warnings.warn(
                f"⚠️  Не удалось сохранить визуализации 'level': {str(e)}\n"
                f"   Продолжаем подготовку данных без графиков."
            )

    def _save_text_histograms(
        self,
        train_data: Dict,
        val_data: Dict,
        test_data: Dict,
        save_path: str,
        save_histograms: bool = True,
        stratification_params: Optional[Dict[str, Any]] = None
    ):
        """
        Сохранение текстовых гистограмм, статистики 'level' и РАСПРЕДЕЛЕНИЯ РЕЖИМОВ ВОЛАТИЛЬНОСТИ.
        
        🔑 КРИТИЧЕСКИ ВАЖНО: параметры стратификации передаются ЯВНО через stratification_params,
        чтобы избежать зависимости от состояния объекта (которое может измениться между вызовами).
        Это гарантирует, что лог в файле точно соответствует параметрам подготовки КОНКРЕТНОГО кэша.
        
        Args:
            stratification_params: словарь с фактически использованными параметрами:
                {
                    'adaptive_blocks': bool,
                    'block_size': int,                 # для фиксированной стратификации
                    'min_regime_per_block': int,       # для адаптивной стратификации
                    'max_block_size': int,             # для адаптивной стратификации
                    'seq_len': int,
                    'buffer_size': int,
                    'min_windows_per_regime': int      # порог валидации баланса
                }
        """
        if not save_histograms:
            return
        
        try:
            hist_path = save_path if not save_path.endswith('.pkl') else save_path[:-4]
            hist_file = f"{hist_path}_histograms.txt"
            hist_dir = os.path.dirname(hist_path) or '.'
            os.makedirs(hist_dir, exist_ok=True)
            
            print(f"\n📊 Сохранение текстовых гистограмм и распределения режимов в: {hist_file}")
            
            with open(hist_file, 'w', encoding='utf-8') as f:
                f.write("=" * 80 + "\n")
                f.write("ТЕКСТОВЫЕ ГИСТОГРАММЫ И СТАТИСТИКА ПЕРЕМЕННОЙ 'level'\n")
                f.write("=" * 80 + "\n")
                f.write(f"Сгенерировано: {datetime.datetime.now()}\n")
                
                # 🔑 ИСПРАВЛЕНИЕ: параметры берутся из ЯВНО ПЕРЕДАННОГО словаря, а не из состояния объекта
                if stratification_params is not None:
                    f.write("Параметры стратификации (фактически использованные при подготовке):\n")
                    f.write(f"  • seq_len: {stratification_params.get('seq_len', 'N/A')}\n")
                    f.write(f"  • buffer_size: {stratification_params.get('buffer_size', 'N/A')}\n")
                    f.write(f"  • min_windows_per_regime: {stratification_params.get('min_windows_per_regime', 'N/A')}\n")
                    f.write(f"  • adaptive_blocks: {stratification_params.get('adaptive_blocks', 'N/A')}\n")
                    
                    if stratification_params.get('adaptive_blocks', False):
                        f.write(f"  • min_regime_per_block: {stratification_params.get('min_regime_per_block', 'N/A')}\n")
                        f.write(f"  • max_block_size: {stratification_params.get('max_block_size', 'N/A')}\n")
                        f.write("  • Режим: АДАПТИВНАЯ СТРАТИФИКАЦИЯ (блоки формируются до достижения баланса режимов)\n")
                    else:
                        f.write(f"  • block_size: {stratification_params.get('block_size', 'N/A')}\n")
                        f.write("  • Режим: ФИКСИРОВАННАЯ СТРАТИФИКАЦИЯ (постоянный размер блока)\n")
                else:
                    # Fallback (для обратной совместимости, но не рекомендуется)
                    f.write("⚠️  ПРЕДУПРЕЖДЕНИЕ: параметры стратификации не переданы явно!\n")
                    f.write("   Информация ниже может НЕ СООТВЕТСТВОВАТЬ параметрам подготовки кэша.\n")
                    f.write(f"  • seq_len: {self.seq_len}\n")
                    f.write(f"  • buffer_size: {self.buffer_size}\n")
                    f.write(f"  • block_size: {self.block_size}\n")
                    f.write(f"  • adaptive_blocks: {self.adaptive_blocks}\n")
                
                f.write("=" * 80 + "\n")
                
                # Статистика и гистограммы по сплитам
                for split_name, split_data, color_code in [
                    ('TRAIN', train_data, '🟢'),
                    ('VAL', val_data, '🔵'),
                    ('TEST', test_data, '🔴')
                ]:
                    if split_data['n_samples'] == 0:
                        continue
                    
                    data = split_data['y_target']
                    n_samples = split_data['n_samples']
                    
                    f.write(f"{color_code} СПЛИТ: {split_name} (n={n_samples} образцов)\n")
                    f.write("-" * 80 + "\n")
                    f.write("Статистика:\n")
                    f.write(f"  • Среднее:    {np.mean(data):.6f}\n")
                    f.write(f"  • Медиана:    {np.median(data):.6f}\n")
                    f.write(f"  • Стандартное отклонение: {np.std(data):.6f}\n")
                    f.write(f"  • Минимум:    {np.min(data):.6f}\n")
                    f.write(f"  • Максимум:   {np.max(data):.6f}\n")
                    f.write(f"  • Персентили: P10={np.percentile(data, 10):.6f} | "
                            f"P25={np.percentile(data, 25):.6f} | "
                            f"P75={np.percentile(data, 75):.6f} | "
                            f"P90={np.percentile(data, 90):.6f}\n")
                    f.write("\n")
                    
                    f.write("Гистограмма распределения 'level' (20 бинов):\n")
                    f.write("Диапазон значений        : Количество  : Нормализовано (%)\n")
                    f.write("-" * 80 + "\n")
                    
                    hist, edges = np.histogram(data, bins=20)
                    max_count = np.max(hist)
                    
                    for i, count in enumerate(hist):
                        norm_pct = (count / max_count) * 100 if max_count > 0 else 0
                        bar = "█" * int(norm_pct / 5)
                        f.write(
                            f"{edges[i]:>12.4f} — {edges[i+1]:<12.4f} : "
                            f"{count:>8d}   : {count/n_samples*100:>6.2f}% {bar}\n"
                        )
                    
                    f.write("-" * 80 + "\n\n")
                
                # 🔑 КРИТИЧЕСКИ ВАЖНЫЙ РАЗДЕЛ: РАСПРЕДЕЛЕНИЕ РЕЖИМОВ ВОЛАТИЛЬНОСТИ
                f.write("=" * 80 + "\n")
                f.write("РАСПРЕДЕЛЕНИЕ РЕЖИМОВ ВОЛАТИЛЬНОСТИ ПО СПЛИТАМ (КЛЮЧЕВОЙ ПОКАЗАТЕЛЬ СТРАТИФИКАЦИИ)\n")
                f.write("=" * 80 + "\n")
                f.write("Режимы: LOW (низкая волатильность) | MID (средняя) | HIGH (высокая)\n")
                f.write("-" * 80 + "\n")
                
                # Собираем данные по режимам
                splits_data = [
                    ('TRAIN', train_data, '🟢'),
                    ('VAL', val_data, '🔵'),
                    ('TEST', test_data, '🔴')
                ]
                
                # Вычисляем распределение для каждого сплита
                split_regime_distributions = {}
                total_samples = 0
                total_regime_counts = np.zeros(3, dtype=int)
                
                for split_name, split_data, _ in splits_data:
                    if split_data['n_samples'] == 0:
                        continue
                    
                    regime_labels = split_data['regime_labels']
                    regime_counts = np.bincount(regime_labels, minlength=3)
                    total = regime_counts.sum()
                    
                    split_regime_distributions[split_name] = {
                        'counts': regime_counts,
                        'total': total,
                        'percentages': (regime_counts / total * 100) if total > 0 else np.zeros(3)
                    }
                    
                    total_samples += total
                    total_regime_counts += regime_counts
                
                # Глобальное распределение
                global_percentages = (total_regime_counts / total_samples * 100) if total_samples > 0 else np.zeros(3)
                
                # Таблица распределения
                f.write(f"{'Сплит':<8} | {'Всего':>8} | {'LOW (%)':>10} | {'MID (%)':>10} | {'HIGH (%)':>10} | Баланс\n")
                f.write("-" * 80 + "\n")
                
                # Для каждого сплита
                max_deviation = 0.0
                for split_name, split_data, color_code in splits_data:
                    if split_name not in split_regime_distributions:
                        continue
                    
                    dist = split_regime_distributions[split_name]
                    counts = dist['counts']
                    total = dist['total']
                    percentages = dist['percentages']
                    
                    # Отклонение от глобального распределения
                    deviations = np.abs(percentages - global_percentages)
                    max_dev = np.max(deviations)
                    max_deviation = max(max_deviation, max_dev)
                    
                    # Визуальный индикатор баланса
                    if max_dev < 2.0:
                        balance_indicator = "✅ Отлично"
                    elif max_dev < 5.0:
                        balance_indicator = "⚠️  Умеренно"
                    else:
                        balance_indicator = "❌ Критично"
                    
                    f.write(
                        f"{color_code} {split_name:<6} | {total:>8} | "
                        f"{percentages[0]:>9.1f}% | {percentages[1]:>9.1f}% | {percentages[2]:>9.1f}% | {balance_indicator}\n"
                    )
                
                # Итоговая строка
                f.write("-" * 80 + "\n")
                f.write(
                    f"{'ИТОГО':<8} | {total_samples:>8} | "
                    f"{global_percentages[0]:>9.1f}% | {global_percentages[1]:>9.1f}% | {global_percentages[2]:>9.1f}% | —\n"
                )
                f.write("-" * 80 + "\n")
                
                # Анализ баланса
                f.write("\nАнализ баланса режимов:\n")
                f.write(f"  • Глобальное распределение: LOW={global_percentages[0]:.1f}% | "
                        f"MID={global_percentages[1]:.1f}% | HIGH={global_percentages[2]:.1f}%\n")
                f.write(f"  • Максимальное отклонение от глобального распределения: {max_deviation:.1f}%\n")
                
                if max_deviation < 2.0:
                    f.write("  • ✅ СТРАТИФИКАЦИЯ УСПЕШНА: дисбаланс < 2% для всех режимов и сплитов\n")
                elif max_deviation < 5.0:
                    f.write("  • ⚠️  УМЕРЕННЫЙ ДИСБАЛАНС: отклонение 2-5%. Модель может быть менее устойчива к редким режимам.\n")
                else:
                    f.write("  • ❌ КРИТИЧЕСКИЙ ДИСБАЛАНС: отклонение > 5%. Рекомендуется:\n")
                    f.write("      - Уменьшить block_size (для фиксированной стратификации)\n")
                    f.write("      - Уменьшить min_regime_per_block или увеличить max_block_size (для адаптивной)\n")
                    f.write("      - Использовать взвешенную функцию потерь по режимам волатильности при обучении\n")
                
                f.write("\n")
                
                # Текстовые гистограммы режимов (визуализация баланса)
                f.write("Визуализация баланса режимов (каждый символ = 2%):\n")
                f.write("-" * 80 + "\n")
                regime_names = ['LOW ', 'MID ', 'HIGH']
                for i, regime_name in enumerate(regime_names):
                    # Глобальная доля
                    global_bar = "█" * int(global_percentages[i] / 2)
                    f.write(f"Глобально {regime_name}: {global_percentages[i]:5.1f}% | {global_bar}\n")
                    
                    # Доли по сплитам
                    for split_name, _, color_code in splits_data:
                        if split_name not in split_regime_distributions:
                            continue
                        pct = split_regime_distributions[split_name]['percentages'][i]
                        bar = "█" * int(pct / 2)
                        f.write(f"  {color_code} {split_name:<6} {regime_name}: {pct:5.1f}% | {bar}\n")
                    f.write("\n")
                
                f.write("=" * 80 + "\n\n")
                
                # Сравнение распределений 'level'
                f.write("=" * 80 + "\n")
                f.write("СРАВНЕНИЕ РАСПРЕДЕЛЕНИЙ ПЕРЕМЕННОЙ 'level' МЕЖДУ СПЛИТАМИ\n")
                f.write("=" * 80 + "\n")
                
                splits = [
                    ('TRAIN', train_data['y_target']),
                    ('VAL', val_data['y_target']),
                    ('TEST', test_data['y_target'])
                ]
                
                from scipy import stats
                f.write("\nКритерий Колмогорова-Смирнова (мера различия распределений 'level'):\n")
                f.write("  Значение < 0.1: распределения очень похожи\n")
                f.write("  Значение 0.1-0.3: умеренные различия (норма для временных рядов)\n")
                f.write("  Значение > 0.3: существенные различия\n")
                f.write("-" * 80 + "\n")
                
                for i in range(len(splits)):
                    for j in range(i + 1, len(splits)):
                        name1, data1 = splits[i]
                        name2, data2 = splits[j]
                        ks_stat, p_value = stats.ks_2samp(data1, data2)
                        f.write(f"{name1:5s} vs {name2:5s} : KS = {ks_stat:.4f} | p-value = {p_value:.4f}\n")
                
                f.write("\n" + "=" * 80 + "\n")
                f.write("КОММЕНТАРИЙ ДЛЯ АНАЛИТИКА:\n")
                f.write("=" * 80 + "\n")
                f.write("• Небольшие различия между сплитами — норма для временных рядов (тренды, сезонность).\n")
                f.write("• Существенные различия (>0.3 по KS) могут указывать на:\n")
                f.write("  - Недостаточный размер датасета для стабильной стратификации\n")
                f.write("  - Резкие структурные изменения в данных (кризисы, смена режима рынка)\n")
                f.write("  - Слишком большой block_size (рекомендуется уменьшить до 100-150)\n")
                f.write("  - При адаптивной стратификации: уменьшите min_regime_per_block или увеличьте max_block_size\n")
                f.write("=" * 80 + "\n")
            
            print(f"   ✅ Текстовые гистограммы и распределение режимов сохранены: {hist_file}")
            print(f"   💡 Ключевой показатель: максимальное отклонение режимов = {max_deviation:.1f}%")
            if stratification_params is not None:
                if stratification_params.get('adaptive_blocks', False):
                    print(f"   💡 Стратификация: АДАПТИВНАЯ (min_regime_per_block={stratification_params['min_regime_per_block']}, max_block_size={stratification_params['max_block_size']})")
                else:
                    print(f"   💡 Стратификация: ФИКСИРОВАННАЯ (block_size={stratification_params['block_size']})")
        
        except Exception as e:
            import traceback
            warnings.warn(
                f"⚠️  Не удалось сохранить текстовые гистограммы: {str(e)}\n"
                f"   Трассировка: {traceback.format_exc()}\n"
                f"   Продолжаем подготовку данных без гистограмм."
            )
            
