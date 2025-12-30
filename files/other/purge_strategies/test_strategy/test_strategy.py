from datetime import datetime
import numpy as np
from collections import deque
import copy

from utils.core.functions import MarketProcess


class TestStrategy(MarketProcess):
    def __init__(self, history_market_parser, orderbook_parser, nwe_bounds_indicator, atr_bounds_indicator):
        self.history_market_parser = history_market_parser
        self.orderbook_parser = orderbook_parser
        self.nwe_bounds_indicator = nwe_bounds_indicator
        self.atr_bounds_indicator = atr_bounds_indicator

        # Adjusted parameters for more signals while maintaining quality
        self.orderbook_depth = 20
        self.imbalance_threshold_buy = 0.52  # Lowered for more long signals
        self.imbalance_threshold_sell = 0.48  # Lowered for more short signals
        self.spread_threshold = 0.001
        self.min_atr_range = 20.0
        self.momentum_period = 3
        self.momentum_threshold = 0.0  # Direction only
        self.risk_reward_ratio = 2.5  # For high winrate
        self.max_position_duration_minutes = 20

        # State
        self.in_trade = False
        self.entry_price = None
        self.position_type = None
        self.entry_time = None
        self.sl = None
        self.tp = None
        self.orderbook_history = deque(maxlen=3)
        self.trade_history = []

    def prepare(self, start_time=None, end_time=None):
        print('Preparing TestStrategy...')
        self.trade_history = []
        self.orderbook_history.clear()

    def calculate_simple_imbalance(self, orderbook):
        try:
            bids = orderbook['bids'][:self.orderbook_depth]
            asks = sorted(orderbook['asks'], key=lambda x: float(x['price']))[:self.orderbook_depth]

            bid_vol = sum(float(b['amount']) for b in bids if 'amount' in b)
            ask_vol = sum(float(a['amount']) for a in asks if 'amount' in a)

            if bid_vol + ask_vol == 0:
                return 0.5
            return bid_vol / (bid_vol + ask_vol)
        except Exception as e:
            print(f"Imbalance calculation error: {e}")
            return 0.5

    def detect_orderbook_aggression(self, orderbook):
        try:
            best_bid = float(orderbook['bids'][0]['price']) if orderbook['bids'] else 0
            best_ask_item = sorted(orderbook['asks'], key=lambda x: float(x['price']), reverse=True)[0] if orderbook[
                'asks'] else None
            best_ask = float(best_ask_item['price']) if best_ask_item else 0

            spread = (best_ask - best_bid) / best_bid if best_bid > 0 else 0

            top_bid_vol = float(orderbook['bids'][0]['amount']) if orderbook['bids'] else 0
            top_ask_vol = float(best_ask_item['amount']) if best_ask_item else 0

            return {
                'spread': spread,
                'top_bid_pressure': top_bid_vol,
                'top_ask_pressure': top_ask_vol,
                'bid_ask_ratio_top': top_bid_vol / (top_bid_vol + top_ask_vol + 1e-8)
            }
        except Exception as e:
            print(f"Orderbook aggression error: {e}")
            return {'spread': 0, 'top_bid_pressure': 0, 'top_ask_pressure': 0, 'bid_ask_ratio_top': 0.5}

    def calculate_momentum(self, df, period):
        if len(df) < period:
            return 0
        recent_closes = df.tail(period)['close'].astype(float)
        return (recent_closes.iloc[-1] - recent_closes.iloc[0]) / recent_closes.iloc[0]

    def is_green_candle(self, df):
        last_candle = df.iloc[-1]
        return float(last_candle['close']) > float(last_candle['open'])

    def is_red_candle(self, df):
        last_candle = df.iloc[-1]
        return float(last_candle['close']) < float(last_candle['open'])

    def run_historical(self, start_time, current_time):
        if self.history_market_parser.df.empty:
            print(f"{current_time}: Empty dataframe, skipping")
            return

        df = self.history_market_parser.df
        current_price = float(df.iloc[-1]['close'])

        # ATR range check
        atr_bounds = self.atr_bounds_indicator.bounds
        atr_range = atr_bounds['upper'] - atr_bounds['lower']
        if atr_range < self.min_atr_range:
            print(f"{current_time}: ATR range {atr_range:.2f} < {self.min_atr_range}, skipping")
            return

        # Orderbook validation
        current_orderbook = self.orderbook_parser.orderbook
        if 'bids' not in current_orderbook or 'asks' not in current_orderbook or not current_orderbook['bids'] or not \
        current_orderbook['asks']:
            print(f"{current_time}: Invalid orderbook data, skipping")
            return

        self.orderbook_history.append(copy.deepcopy(current_orderbook))

        imbalance = self.calculate_simple_imbalance(current_orderbook)
        aggression = self.detect_orderbook_aggression(current_orderbook)

        momentum = self.calculate_momentum(df, self.momentum_period)

        # Position management
        if self.in_trade:
            time_in_trade = (current_time - self.entry_time).total_seconds() / 60 if self.entry_time else 0
            if time_in_trade > self.max_position_duration_minutes:
                self.close_position(current_time, current_price, 'TIME_LIMIT')
                return

            if (self.position_type == 'long' and (current_price <= self.sl or current_price >= self.tp)) or \
                    (self.position_type == 'short' and (current_price >= self.sl or current_price <= self.tp)):
                reason = 'SL' if (self.position_type == 'long' and current_price <= self.sl) or \
                                 (self.position_type == 'short' and current_price >= self.sl) else 'TP'
                self.close_position(current_time, current_price, reason)
                return
            return

        # Signal conditions
        abs_spread = abs(aggression['spread'])

        long_signal = (
                (imbalance > self.imbalance_threshold_buy or aggression['bid_ask_ratio_top'] > 0.6) and
                abs_spread < self.spread_threshold and
                momentum > self.momentum_threshold and
                self.is_green_candle(df)
        )

        short_signal = (
                (imbalance < self.imbalance_threshold_sell or aggression['bid_ask_ratio_top'] < 0.4) and
                abs_spread < self.spread_threshold and
                momentum < -self.momentum_threshold and
                self.is_red_candle(df)
        )

        # Debug logging
        if not long_signal and not short_signal:
            print(f"At {current_time}, Price: {current_price:.2f}, Imbalance: {imbalance:.3f}, "
                  f"Spread: {aggression['spread']:.4f}, Momentum: {momentum:.4f}, "
                  f"Green: {self.is_green_candle(df)}, Red: {self.is_red_candle(df)}, "
                  f"Long: {long_signal} (Imb: {imbalance > self.imbalance_threshold_buy}, Ratio: {aggression['bid_ask_ratio_top'] > 0.6}, "
                  f"Spread: {abs_spread < self.spread_threshold}, Mom: {momentum > self.momentum_threshold}), "
                  f"Short: {short_signal} (Imb: {imbalance < self.imbalance_threshold_sell}, Ratio: {aggression['bid_ask_ratio_top'] < 0.4}, "
                  f"Mom: {momentum < -self.momentum_threshold})")

        if long_signal:
            sl_distance = atr_range * 0.4
            self.sl = current_price - sl_distance
            self.tp = current_price + (sl_distance * self.risk_reward_ratio)
            self.enter_position('long', current_time, current_price, {
                'imbalance': imbalance,
                'spread': aggression['spread'],
                'momentum': momentum
            })
            return

        if short_signal:
            sl_distance = atr_range * 0.4
            self.sl = current_price + sl_distance
            self.tp = current_price - (sl_distance * self.risk_reward_ratio)
            self.enter_position('short', current_time, current_price, {
                'imbalance': imbalance,
                'spread': aggression['spread'],
                'momentum': momentum
            })
            return

    def enter_position(self, position_type, current_time, current_price, signals):
        self.in_trade = True
        self.position_type = position_type
        self.entry_price = current_price
        self.entry_time = current_time

        print(f"🚀 ENTER {position_type.upper()} at {current_price:.2f} | "
              f"Imbalance: {signals['imbalance']:.3f}, Momentum: {signals['momentum']:.4f}, "
              f"SL: {self.sl:.2f}, TP: {self.tp:.2f}")

        with open('files/decisions/decisions_HistoryTestStrategy.txt', 'a') as f:
            f.write(f"{current_time}, ENTER {position_type}, price={current_price}, "
                    f"signals={signals}, sl={self.sl}, tp={self.tp}\n")

    def close_position(self, current_time, current_price, reason):
        pnl = ((current_price - self.entry_price) / self.entry_price if self.position_type == 'long'
               else (self.entry_price - current_price) / self.entry_price)

        self.trade_history.append({
            'entry_time': self.entry_time, 'exit_time': current_time,
            'type': self.position_type, 'entry_price': self.entry_price,
            'exit_price': current_price, 'pnl': pnl, 'reason': reason
        })

        print(f"💰 EXIT {self.position_type.upper()} at {current_price:.2f} | "
              f"PnL: {pnl * 100:+.2f}% | Reason: {reason} | "
              f"Winrate: {self.calculate_winrate():.1%} ({len(self.trade_history)} trades)")

        self.in_trade = False
        self.entry_price = None
        self.position_type = None
        self.entry_time = None
        self.sl = None
        self.tp = None

    def calculate_winrate(self):
        if not self.trade_history:
            return 0
        wins = sum(1 for trade in self.trade_history if trade['pnl'] > 0)
        return wins / len(self.trade_history)