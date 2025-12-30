from datetime import timedelta, datetime
import numpy as np
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN

from files.other.purge_strategies.level_strategy.instances.history_market_level_evaluator import \
    HistoryMarketLevelEvaluator
from files.other.purge_strategies.level_strategy.instances.orderbook_level_momentum_evaluator import \
    OrderbookLevelMomentumEvaluator
from files.other.purge_strategies.level_strategy.instances.orderbook_level_simple_range_evaluator import \
    OrderbookLevelSimpleRangeEvaluator
from utils.core.functions import MarketProcess


class SmartLevelStrategy(MarketProcess):
    def __init__(
        self,
        history_market_parser,
        orderbook_parser,
        nwe_bounds_indicator,
        atr_bounds_indicator,
    ):
        self.history_market = history_market_parser
        self.orderbook = orderbook_parser
        self.nwe = nwe_bounds_indicator
        self.atr = atr_bounds_indicator
        self.symbol = history_market_parser.slash_symbol
        self.orderbook_level_momentum_evaluator = OrderbookLevelMomentumEvaluator(orderbook_parser)
        self.orderbook_level_simple_range_evaluator = OrderbookLevelSimpleRangeEvaluator(self.orderbook_level_momentum_evaluator)
        self.history_market_level_evaluator = HistoryMarketLevelEvaluator(self.history_market)
        self.historical_levels = deque(maxlen=100)  # Self-overwriting array for historical final levels
        self.historical_obis = deque(maxlen=100)  # Buffer for historical Order Book Imbalance
        self.entry_points = []  # Array to store fixed entry points
        self.ml_model = RandomForestClassifier(n_estimators=50, random_state=42)  # Simple ML for rebound prediction
        self.train_ml_model()  # Assume training on historical data (mocked here)

    def train_ml_model(self):
        # Mock training: In real, use historical orderbook data
        # Features: [imbalance, depth, volume_spike, variance]
        # Labels: [rebound_happened: 0/1]
        X = np.random.rand(100, 4)  # Mock features
        y = np.random.randint(0, 2, 100)  # Mock labels
        self.ml_model.fit(X, y)

    def calculate_obi(self, snapshot):
        # Calculate Order Book Imbalance
        bids = snapshot.get('bids', [])
        asks = snapshot.get('asks', [])
        if not bids or not asks:
            return 0.0
        bid_vol = sum(b['amount'] for b in bids[:10])  # Top 10 levels
        ask_vol = sum(a['amount'] for a in asks[:10])
        imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-6)
        self.historical_obis.append(imbalance)
        return imbalance

    def calculate_vwap(self):
        # Volume Weighted Average Price from history_df
        df = self.history_market.history_df.tail(100)
        if df.empty:
            return 0.0
        vwap = (df['close'] * df['volume']).sum() / df['volume'].sum()
        return vwap

    def detect_spoofing(self, snapshot):
        # Simple spoof detection: if max amount > 70% total in cluster, suspect spoof
        bids = snapshot.get('bids', [])
        for bid in bids:
            if bid['amount'] > 0.7 * bid['total']:
                return True
        return False

    def predict_rebound(self, features):
        # ML prediction: probability of rebound
        features = np.array(features).reshape(1, -1)
        return self.ml_model.predict_proba(features)[0][1]  # Prob of class 1 (rebound)

    def get_final_level(self, valid_levels):
        if not valid_levels:
            final_level = {'price1': 0.0, 'price2': 0.0, 'persistence': 0.0}
        else:
            # Step 2: Find the intersection (conjunction) of all valid levels
            overall_price1 = max(l['price1'] for l in valid_levels)
            overall_price2 = min(l['price2'] for l in valid_levels)

            if overall_price1 <= overall_price2:
                persistences = [l.get('quality', l.get('persistence', 0)) for l in valid_levels]
                persistence = min(persistences) if persistences else 0.0
                final_level = {
                    'price1': overall_price1,
                    'price2': overall_price2,
                    'persistence': persistence
                }
            else:
                pairwise_clusters = []
                for i in range(len(valid_levels)):
                    for j in range(i + 1, len(valid_levels)):
                        l1 = valid_levels[i]
                        l2 = valid_levels[j]
                        inter_price1 = max(l1['price1'], l2['price1'])
                        inter_price2 = min(l1['price2'], l2['price2'])
                        if inter_price1 <= inter_price2:
                            p1 = l1.get('quality', l1.get('persistence', 0))
                            p2 = l2.get('quality', l2.get('persistence', 0))
                            persistence = min(p1, p2)
                            pairwise_clusters.append({
                                'price1': inter_price1,
                                'price2': inter_price2,
                                'persistence': persistence,
                                'num_levels': 2
                            })

                if pairwise_clusters:
                    best_cluster = max(pairwise_clusters, key=lambda c: c['persistence'])
                    final_level = {
                        'price1': best_cluster['price1'],
                        'price2': best_cluster['price2'],
                        'persistence': best_cluster['persistence']
                    }
                else:
                    persistences = [l.get('quality', l.get('persistence', 0)) for l in valid_levels]
                    best_idx = np.argmax(persistences)
                    best_level = valid_levels[best_idx]
                    final_level = {
                        'price1': best_level['price1'],
                        'price2': best_level['price2'],
                        'persistence': persistences[best_idx]
                    }

        return final_level


    def process(self, start_time, end_time):
        print('[SmartLevelStrategy] Analyzing...')
        snapshot = self.orderbook.get_by_time(end_time)  # Assume get_by_time returns orderbook
        if self.detect_spoofing(snapshot):
            print("Spoofing detected! But continue.")

        obi = self.calculate_obi(snapshot)
        vwap = self.calculate_vwap()

        level1 = self.orderbook_level_momentum_evaluator.evaluate_support(end_time)
        level2 = self.orderbook_level_simple_range_evaluator.evaluate_support(end_time)
        level3 = self.history_market_level_evaluator.evaluate_support()

        # Step 1: Filter valid levels
        levels = [level1, level2, level3]
        valid_levels = [l for l in levels if l['price1'] > 0 and l['price2'] > 0]
        final_level = self.get_final_level(valid_levels)


        # Fix the final level with time
        final_level_with_time = {'time': end_time, **final_level}
        self.historical_levels.append(final_level_with_time)
        print(f"Final Level: {final_level_with_time}")

        # Adjust final_level with OBI and VWAP
        if obi > 0.2:  # Strong buy imbalance, shift level up
            final_level['price1'] = max(final_level['price1'], vwap * 0.99)
            final_level['price2'] = min(final_level['price2'], vwap * 1.01)

        # Add to historical_levels
        final_level_with_time = {'time': end_time, **final_level}
        self.historical_levels.append(final_level_with_time)
        print(f"Final Level: {final_level_with_time}")

        # Stability analysis
        if len(self.historical_levels) >= 5:
            midpoints = np.array([(l['price1'] + l['price2']) / 2 for l in self.historical_levels])
            times = np.array([(l['time'] - self.historical_levels[0]['time']).total_seconds() for l in self.historical_levels])
            persistences = np.array([l['persistence'] for l in self.historical_levels])
            obis = np.array(list(self.historical_obis)[-5:])  # Last 5 OBIs

            # Features for ML: [obi_mean, std_midpoints, mean_persistence, variance_obis]
            features = [np.mean(obis), np.std(midpoints), np.mean(persistences), np.var(obis)]
            rebound_prob = self.predict_rebound(features)

            # Linear regression for trend
            reg = LinearRegression()
            reg.fit(times.reshape(-1, 1), midpoints)
            slope = reg.coef_[0]

            # DBSCAN for clustering
            db = DBSCAN(eps=0.005 * np.mean(midpoints), min_samples=3)
            clusters = db.fit_predict(midpoints.reshape(-1, 1))
            num_clusters = len(np.unique(clusters)) - (1 if -1 in clusters else 0)  # Exclude noise

            # Check stability
            if abs(slope) < 0.001 and num_clusters == 1 and rebound_prob > 0.7:
                entry_point = {'time': end_time, 'level': final_level}
                self.entry_points.append(entry_point)
                print(f'STABLE LEVEL FIXED! ENTER TRADE at {end_time}: {final_level} (Rebound Prob: {rebound_prob:.2f})')
                self.historical_levels.clear()  # Reset buffer after fixation

        return [final_level_with_time]

    def prepare(self, start_time=None, end_time=None):
        pass

    def run_historical(self, start_time, current_time):
        self.process(current_time - timedelta(hours=4), current_time)


    def run_realtime(self):
        current_time = datetime.now()
        self.process(current_time - timedelta(hours=4), current_time)