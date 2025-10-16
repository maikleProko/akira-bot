import datetime
import numpy as np
from datetime import datetime, timedelta
import os
import json

class MarketProcess:
    def prepare(self):
        pass

    def run(self, start_time=None, current_time=None, end_time=None):
        pass

class FactorAnalyzer(MarketProcess):
    def __init__(self, history_market_parser, orderbook_parser, nwe_bounds_indicator, atr_bounds_indicator):
        self.history_market_parser = history_market_parser
        self.orderbook_parser = orderbook_parser
        self.nwe_bounds_indicator = nwe_bounds_indicator
        self.atr_bounds_indicator = atr_bounds_indicator
        self.previous_cvd = 0  # Для fallback

    def load_historical_orderbooks(self, current_time, num_minutes=5):
        historical_obs = []
        for i in range(num_minutes):
            prev_time = current_time - timedelta(minutes=i)
            year, month, day = prev_time.year, prev_time.month, prev_time.day
            hour, minute = prev_time.hour, prev_time.minute
            path = f'files/orderbook/beautifulsoup/coinglass/beautifulsoup_coinglass_orderbook_BTC-USDT-{year:04d}:{month:02d}:{day:02d}_{hour:02d}:{minute:02d}.json'
            if os.path.exists(path):
                with open(path, 'r') as f:
                    ob = json.load(f)
                    historical_obs.append(ob)
        return historical_obs[::-1]  # От старого к новому

    def calculate_orderbook_score(self):
        current_ob = self.orderbook_parser.orderbook
        if not current_ob or 'bids' not in current_ob or 'asks' not in current_ob:
            return 0

        historical_obs = self.load_historical_orderbooks(datetime.now(), num_minutes=5)
        if len(historical_obs) < 2:
            # Fallback на текущий
            bids = current_ob['bids'][:10]
            asks = current_ob['asks'][:10]
            sum_bid_vol = sum(b['amount'] for b in bids)
            sum_ask_vol = sum(a['amount'] for a in asks)
            ir = (sum_bid_vol - sum_ask_vol) / (sum_bid_vol + sum_ask_vol + 1e-6)
            delta_flow = 0
            spoof_variance = 0
            spoof_factor = 1
            depth_norm = min((sum_bid_vol + sum_ask_vol) / 2 / 1000, 1)
            cvd = sum_bid_vol - sum_ask_vol
        else:
            deltas = []
            for ob in historical_obs:
                bids = ob['bids'][:10]  # Топ-10 уровней, шаг 10 USD - глубина ~100 USD
                asks = ob['asks'][:10]
                sum_bid = sum(b['amount'] for b in bids)
                sum_ask = sum(a['amount'] for a in asks)
                delta = sum_bid - sum_ask
                deltas.append(delta)

            cvd = sum(deltas)  # Cumulative Volume Delta
            prev_avg_delta = np.mean(deltas[:-1])
            delta_flow = deltas[-1] - prev_avg_delta
            spoof_variance = np.var(deltas)
            spoof_factor = 1 if spoof_variance < 500 else 0  # Порог для BTC, adjust по бэктесту

            # IR и depth на последнем (текущем)
            sum_bid_vol = sum(b['amount'] for b in current_ob['bids'][:10])
            sum_ask_vol = sum(a['amount'] for a in current_ob['asks'][:10])
            ir = (sum_bid_vol - sum_ask_vol) / (sum_bid_vol + sum_ask_vol + 1e-6)
            depth_norm = min((sum_bid_vol + sum_ask_vol) / 2 / 1000, 1)

        # Композитный score, включая исторический CVD
        score_ob = (ir * 0.3) + (delta_flow / 1000 * 0.2) + (spoof_factor * 0.2) + (depth_norm * 0.1) + (cvd / 10000 * 0.2)
        return score_ob, cvd

    def detect_sweep_and_manipulation(self, df, is_long=True):
        if len(df) < 3:
            return False, False

        last_candle = df.iloc[-1]
        prev_candle = df.iloc[-2]

        if is_long:
            sweep = last_candle['low'] < prev_candle['low']
            wick_lower = min(last_candle['open'], last_candle['close']) - last_candle['low']
            body = abs(last_candle['close'] - last_candle['open'])
            manipulation = wick_lower > 2 * body and last_candle['close'] > last_candle['open']
        else:
            sweep = last_candle['high'] > prev_candle['high']
            wick_upper = last_candle['high'] - max(last_candle['open'], last_candle['close'])
            body = abs(last_candle['close'] - last_candle['open'])
            manipulation = wick_upper > 2 * body and last_candle['close'] < last_candle['open']

        return sweep, manipulation

    def detect_fvg(self, df, is_long=True):
        if len(df) < 3:
            return False

        candle1 = df.iloc[-3]
        candle3 = df.iloc[-1]
        if is_long:
            fvg = candle1['high'] < candle3['low']
        else:
            fvg = candle1['low'] > candle3['high']
        return fvg

    def calculate_rsi(self, df, period=14):
        if len(df) < period + 1:
            return 50

        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=period).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period).mean()
        rs = gain / (loss + 1e-6)
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1]

    def calculate_ema_crossover(self, df, is_long=True):
        if len(df) < 21:
            return False

        ema9 = df['close'].ewm(span=9, adjust=False).mean()
        ema21 = df['close'].ewm(span=21, adjust=False).mean()
        if is_long:
            crossover = ema9.iloc[-1] > ema21.iloc[-1] and ema9.iloc[-2] <= ema21.iloc[-2]
        else:
            crossover = ema9.iloc[-1] < ema21.iloc[-1] and ema9.iloc[-2] >= ema21.iloc[-2]
        return crossover

    def run(self, start_time=None, current_time=None, end_time=None):
        df = self.history_market_parser.df
        if df.empty:
            return

        current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        current_price = df['close'].iloc[-1]

        nwe = self.nwe_bounds_indicator.bounds
        atr = self.atr_bounds_indicator.bounds

        if not nwe or not atr:
            return

        score_ob, cvd = self.calculate_orderbook_score()

        rsi = self.calculate_rsi(df)

        current_atr_approx = df['high'].iloc[-1] - df['low'].iloc[-1]
        avg_atr = (atr['upper'] - atr['lower']) / 2

        # Long сигнал
        sweep_low, manipulation_long = self.detect_sweep_and_manipulation(df, is_long=True)
        fvg_long = self.detect_fvg(df, is_long=True)
        ema_cross_long = self.calculate_ema_crossover(df, is_long=True)
        print('---')
        print('states:')
        print(score_ob)
        print(cvd)
        print(current_price > nwe['lower'])
        print(sweep_low and manipulation_long)
        print(fvg_long)
        print(rsi)
        print(ema_cross_long)
        print(current_atr_approx < 1.5 * avg_atr)
        print('---')

        if (score_ob > 0.6 and
            cvd > 0 and
            current_price > nwe['lower'] and
            sweep_low and manipulation_long and
            fvg_long and
            45 < rsi < 55 and
            ema_cross_long and
            current_atr_approx < 1.5 * avg_atr):

            print(f"Entering long trade at {current_time_str}")
            with open('/files/decisions.txt', 'a') as f:
                f.write(f"Entering long trade at {current_time_str}\n")

        # Short сигнал
        sweep_high, manipulation_short = self.detect_sweep_and_manipulation(df, is_long=False)
        fvg_short = self.detect_fvg(df, is_long=False)
        ema_cross_short = self.calculate_ema_crossover(df, is_long=False)

        if (score_ob < -0.6 and
            cvd < 0 and
            current_price < nwe['upper'] and
            sweep_high and manipulation_short and
            fvg_short and
            45 < rsi < 55 and
            ema_cross_short and
            current_atr_approx < 1.5 * avg_atr):

            print(f"Entering short trade at {current_time_str}")
            with open('/files/decisions_FactorAnalyzer.txt', 'a') as f:
                f.write(f"Entering short trade at {current_time_str}\n")