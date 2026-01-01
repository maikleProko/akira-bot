from datetime import datetime

import numpy as np
import pandas as pd
from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.abstracts.indicator import Indicator


class NweBoundsIndicator(Indicator):
    def __init__(self, history_market_parser: HistoryMarketParser):
        super().__init__(history_market_parser)
        self.bounds = {}
        self.candle_count = 0

        # Будут хранить полные исторические значения envelope
        self.history_out = None    # pd.Series или np.array
        self.history_upper = None
        self.history_lower = None

    def get_nadaraya_compute(self, series: pd.Series, h=8.0, mult=3.0):
        try:
            # Входные параметры
            src = series
            coefs = []
            den = 0.0

            # Функция гауссовского распределения
            def gauss(x, h):
                return np.exp(-(x ** 2) / (h * h * 2))

            # Вычисление коэффициентов
            for i in range(500):
                w = gauss(i, h)
                coefs.append(w)

            den = sum(coefs)
            mae = np.zeros(len(src))
            upper = []
            lower = []
            out = np.zeros(len(src))
            src = src.to_numpy()[::-1]

            self.candle_count = len(src) - 500
            for i in range(0, len(src) - 500):
                local_out = 0
                for j in range(0, len(coefs)):
                    local_out += coefs[j] * float(src[i + j])
                local_out /= den
                out[i] = local_out


            src = src[::-1]
            out = out[::-1]

            for i in range(500, len(src)):
                local_mae = 0
                iter = 0
                for j in range(0, 499):
                    if out[i - j] > 0:
                        local_mae += abs(float(src[i - j]) - out[i - j])
                        iter += 1
                local_mae *= mult
                local_mae /= iter
                mae[i] = local_mae

            upper = out + mae
            lower = out - mae


            return out, upper, lower
        except:
            return [], [], []

    def prepare(self, start_time=None, end_time=None):
        """
        Подготавливает индикатор для исторического режима.
        Вычисляет envelope на ВСЁМ history_df и сохраняет полные серии.
        """
        df = self.history_market_parser.history_df
        if df is None or df.empty or 'close' not in df.columns:
            print("[NweBoundsIndicator] history_df пустой или без 'close' — prepare пропущен")
            self.history_out = self.history_upper = self.history_lower = pd.Series()
            return

        close_series = df['close']

        out, upper, lower = self.get_nadaraya_compute(close_series)

        if len(out) == 0:
            self.history_out = self.history_upper = self.history_lower = pd.Series()
            return

        # Приводим к тому же индексу, что и последние len(out) строк history_df
        target_index = close_series.tail(len(out)).index

        self.history_out = pd.Series(out, index=target_index)
        self.history_upper = pd.Series(upper, index=target_index)
        self.history_lower = pd.Series(lower, index=target_index)

        # Для удобства — сколько свечей реально использовано в расчёте
        self.candle_count = len(out)

    def run_realtime(self):
        """
        Для реального времени: берём текущий df и вычисляем последние значения.
        """
        df = self.history_market_parser.df
        if df is None or df.empty or 'close' not in df.columns:
            self.bounds = {'out': np.nan, 'upper': np.nan, 'lower': np.nan}
            return

        out, upper, lower = self.get_nadaraya_compute(df['close'])

        if len(out) > 0:
            self.bounds = {
                'out': float(out[-1]),
                'upper': float(upper[-1]),
                'lower': float(lower[-1]),
            }
        else:
            self.bounds = {'out': np.nan, 'upper': np.nan, 'lower': np.nan}

    def run_historical(self, start_time: datetime, current_time: datetime):
        """
        Для исторического режима: находим последнюю строку, где время <= current_time,
        и берём соответствующие значения из предвычисленных исторических серий.
        """
        if (self.history_out is None or self.history_out.empty or
                self.history_upper is None or self.history_upper.empty or
                self.history_lower is None or self.history_lower.empty):
            self.bounds = {'out': np.nan, 'upper': np.nan, 'lower': np.nan}
            return

        # Предполагаем, что в history_df колонка с временем называется 'time' и это datetime
        time_col = self.history_market_parser.history_df['time']

        # Находим все строки, где время <= current_time
        valid_mask = pd.to_datetime(time_col) <= current_time

        if not valid_mask.any():
            # Ещё нет данных на это время
            self.bounds = {'out': np.nan, 'upper': np.nan, 'lower': np.nan}
            return

        # Берём последнюю доступную строку
        last_valid_index = time_col[valid_mask].index[-1]

        # Проверяем, есть ли эта строка в наших предвычисленных сериях
        if last_valid_index in self.history_out.index:
            self.bounds = {
                'out': float(self.history_out.loc[last_valid_index]),
                'upper': float(self.history_upper.loc[last_valid_index]),
                'lower': float(self.history_lower.loc[last_valid_index]),
            }
        else:
            # Если по какой-то причине индекс не совпадает (например, из-за обрезки в get_nadaraya_compute)
            self.bounds = {'out': np.nan, 'upper': np.nan, 'lower': np.nan}