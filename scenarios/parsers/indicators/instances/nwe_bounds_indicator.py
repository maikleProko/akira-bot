import numpy as np

from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.parsers.indicators.abstracts.indicator import Indicator


class NweBoundsIndicator(Indicator):

    def __init__(self, history_market_parser: HistoryMarketParser):
        super().__init__(history_market_parser)
        self.bounds = {}

    @staticmethod
    def get_nadaraya_compute(dataframe, h=8.0, mult=3.0):
        try:
            # Входные параметры
            src = dataframe
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

            for i in range(0, len(src) - 500):
                local_out = 0
                for j in range(0, len(coefs)):
                    local_out += coefs[j] * float(src[i + j])
                local_out /= den
                out[i] = local_out


            src = src[::-1][-1000:]
            out = out[::-1]

            print(out[-1])

            for i in range(500, len(src)):
                local_mae = 0
                iter = 0
                for j in range(0, 499):
                    if out[i - j] != 0:
                        local_mae += abs(float(src[i - j]) - out[i - j])
                        iter += 1
                local_mae *= mult
                local_mae /= (iter-1)
                mae[i] = local_mae


            upper = out + mae
            lower = out - mae

            return upper, lower
        except:
            return [], []


    def run(self, start_time=None, end_time=None):
        df = self.history_market_parser.df
        if len(df) > 500:
            upper, lower = self.get_nadaraya_compute(df['close'])
            self.bounds = {
                'upper': upper[-1],
                'lower': lower[-1],
            }