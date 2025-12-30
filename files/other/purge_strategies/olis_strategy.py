from collections import deque
import copy
import datetime
import numpy as np
from utils.core.functions import MarketProcess


class OlisStrategy(MarketProcess):
    def __init__(self, history_market_parser, orderbook_parser, nwe_bounds_indicator, atr_bounds_indicator):
        self.history_market_parser = history_market_parser
        self.orderbook_parser = orderbook_parser
        self.nwe_bounds_indicator = nwe_bounds_indicator
        self.atr_bounds_indicator = atr_bounds_indicator
        self.price_history = deque(maxlen=15)  # Last 15 seconds prices for grab detection
        self.orderbook_history = deque(maxlen=15)  # Last 15 orderbooks for sweep detection
        self.atr_history = deque(maxlen=1800)  # 30 min * 60 sec for average ATR

    def prepare(self, start_time=None, end_time=None):
        pass

    def run_realtime(self):
        # Get current data
        if self.orderbook_parser.orderbook is None:
            print('[OlisStrategy]  orderbook is None ')
            return  # Not ready yet

        bids = self.orderbook_parser.orderbook['bids']
        asks = self.orderbook_parser.orderbook['asks']

        # Current mid price as proxy for current price
        if not bids or not asks:
            print('[OlisStrategy]  no bids or asks ')
            return
        current_price = (bids[0]['price'] + asks[0]['price']) / 2

        # Append to histories
        self.price_history.append(current_price)
        self.orderbook_history.append(copy.deepcopy(self.orderbook_parser.orderbook))

        # ATR width as proxy for volatility
        atr = self.atr_bounds_indicator.bounds
        atr_width = atr['upper'] - atr['lower']
        self.atr_history.append(atr_width)

        # NWE bounds
        nwe = self.nwe_bounds_indicator.bounds

        # Skip if not enough history for analysis
        if len(self.price_history) < 10 or len(self.atr_history) < 60:
            print('[OlisStrategy]  not enough history ')
            return

        # Volatility filter: current ATR > 1.2x average ATR over last 30 min
        avg_atr = np.mean(self.atr_history)
        cond1 = atr_width > 1.2 * avg_atr

        # Lower bound for support
        lower_bound = min(nwe['lower'], atr['lower'])

        # Detect liquidity grab for long position
        prices = list(self.price_history)
        min_price_idx = np.argmin(prices[-10:]) + (len(prices) - 10)  # Index in full history, last 10 sec
        min_price = prices[min_price_idx]
        bounce_percentage = (current_price - min_price) / min_price * 100

        # Check for dip to near lower_bound and subsequent bounce (>0.1% dip, >0.05% bounce)
        cond2 = abs(current_price - min_price) / current_price >= 0.001 and min_price <= lower_bound * 1.001 and bounce_percentage > 0.05

        # Check for sweep: volume decrease at the dipped level (>20% drop)
        pre_dip_idx = max(0, min_price_idx - 5)  # 5 sec before dip
        pre_dip_ob = self.orderbook_history[pre_dip_idx]
        dip_ob = self.orderbook_history[min_price_idx]

        # Find closest bid level to min_price in pre_dip orderbook
        pre_bids = pre_dip_ob['bids']
        closest_level = min(pre_bids, key=lambda b: abs(b['price'] - min_price))
        pre_amount = closest_level['amount']

        # Check same level in dip_ob
        dip_bids = [b for b in dip_ob['bids'] if abs(b['price'] - closest_level['price']) < 1e-6]
        dip_amount = dip_bids[0]['amount'] if dip_bids else 0

        cond3 = dip_amount < pre_amount * 0.8

        # Compute imbalance and delta on current orderbook (top 10 levels)
        top_bid_sum = sum(b['amount'] for b in bids[:10])
        top_ask_sum = sum(a['amount'] for a in asks[:10])
        imbalance = top_bid_sum / top_ask_sum if top_ask_sum > 0 else 0
        delta = top_bid_sum - top_ask_sum

        cond4 = imbalance > 2
        cond5 = delta > 50

        print(f'[OlisStrategy]  conditions fit [{str(cond1)},{str(cond2)},{str(cond3)},{str(cond4)},{str(cond5)}] ')

        # Thresholds: at least 4 out of 5
        passed = sum([cond1, cond2, cond3, cond4, cond5])
        if passed >= 4:
            # Conditions met: signal entry
            current_time = datetime.datetime.now()
            message = f"Входим в сделку at {current_time}"
            print(message)
            with open('files/decisions/decisions_OlisStrategy.txt', 'a') as f:
                f.write(message + '\n')