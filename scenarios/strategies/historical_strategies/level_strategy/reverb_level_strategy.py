from asyncio import sleep
from datetime import timedelta
import numpy as np

from scenarios.strategies.historical_strategies.level_strategy.instances.history_market_level_evaluator import \
    HistoryMarketLevelEvaluator
from scenarios.strategies.historical_strategies.level_strategy.instances.orderbook_level_momentum_evaluator import \
    OrderbookLevelMomentumEvaluator
from scenarios.strategies.historical_strategies.level_strategy.instances.orderbook_level_simple_range_evaluator import \
    OrderbookLevelSimpleRangeEvaluator
from utils.core.functions import MarketProcess


class ReverbLevelStrategy(MarketProcess):
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
        self.reverberated_level = {'price1': 0.0, 'price2': 0.0, 'persistence': -1.0, 'depth': 1}

    def get_level(self, start_time, end_time):
        print('[ReverbLevelStrategy] Analyzing...')
        level1 = self.orderbook_level_momentum_evaluator.evaluate_support(end_time)
        level2 = self.orderbook_level_simple_range_evaluator.evaluate_support(end_time)
        level3 = self.history_market_level_evaluator.evaluate_support()

        print('----')
        print(end_time.strftime('%Y:%m:%d_%H:%M'))
        print(f"Level1 (Orderbook Momentum): {level1}")
        print(f"Level2 (Orderbook Range): {level2}")
        print(f"Level3 (History Market): {level3}")
        print('----')

        # Step 1: Filter valid levels
        levels = [level1, level2, level3]
        valid_levels = [l for l in levels if l['price1'] > 0 and l['price2'] > 0]

        if not valid_levels:
            return [{'price1': 0.0, 'price2': 0.0, 'persistence': 0.0}]

        # Step 2: Find the intersection (conjunction) of all valid levels
        overall_price1 = max(l['price1'] for l in valid_levels)
        overall_price2 = min(l['price2'] for l in valid_levels)

        if overall_price1 <= overall_price2:
            # There is a common intersection
            persistences = [l.get('quality', l.get('persistence', 0)) for l in valid_levels]
            persistence = min(persistences) if persistences else 0.0
            result = [{
                'price1': overall_price1,
                'price2': overall_price2,
                'persistence': persistence
            }]
        else:
            # No full intersection, find pairwise intersections
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
                # Select the pairwise intersection with highest persistence
                best_cluster = max(pairwise_clusters, key=lambda c: c['persistence'])
                result = [{
                    'price1': best_cluster['price1'],
                    'price2': best_cluster['price2'],
                    'persistence': best_cluster['persistence']
                }]
            else:
                # No intersections, select the level with highest persistence
                persistences = [l.get('quality', l.get('persistence', 0)) for l in valid_levels]
                best_idx = np.argmax(persistences)
                best_level = valid_levels[best_idx]
                result = [{
                    'price1': best_level['price1'],
                    'price2': best_level['price2'],
                    'persistence': persistences[best_idx]
                }]

        print(f"Calculated Level: {result}")
        return result

    def process(self, start_time, end_time):
        # Get current price from history_df
        if self.history_market.history_df.empty or 'close' not in self.history_market.history_df.columns:
            print("Warning: history_df is empty or missing 'close' column")
            course = 0.0
        else:
            course = float(self.history_market.history_df['close'].iloc[-1])

        # Check if current price is within the reverberated level
        if (self.reverberated_level['persistence'] != -1 and
            self.reverberated_level['price1'] < course < self.reverberated_level['price2']):
            print('CAAATCH!!!!!!!!!!!!!!!!!!!!!!')
            self.reverberated_level = {'price1': 0.0, 'price2': 0.0, 'persistence': -1.0, 'depth': 1}
            return [{'price1': 0.0, 'price2': 0.0, 'persistence': 0.0}]

        # Calculate new level
        new_level = self.get_level(start_time, end_time)[0]  # Extract the first (and only) level from list

        if new_level['price1'] == 0.0 and new_level['price2'] == 0.0:
            print("Warning: Invalid new level, returning default")
            return [{'price1': 0.0, 'price2': 0.0, 'persistence': 0.0}]

        # Initialize or update reverberated level
        if self.reverberated_level['persistence'] == -1:
            self.reverberated_level = {
                'price1': new_level['price1'],
                'price2': new_level['price2'],
                'persistence': new_level['persistence'],
                'depth': 1
            }
        else:
            print('Updating reverberated level...')
            reverb_coeff = 1 / self.reverberated_level['depth'] if self.reverberated_level['depth'] != 0 else 0.5
            self.reverberated_level = {
                'price1': (self.reverberated_level['price1'] + new_level['price1']) / 2,
                'price2': (self.reverberated_level['price2'] + new_level['price2']) / 2,
                'persistence': (self.reverberated_level['persistence'] + new_level['persistence']) / 2,
                'depth': self.reverberated_level['depth'] + 1
            }

        print(f"Reverberated Level: {self.reverberated_level}")
        return [{
            'price1': self.reverberated_level['price1'],
            'price2': self.reverberated_level['price2'],
            'persistence': self.reverberated_level['persistence']
        }]

    def prepare(self, start_time=None, end_time=None):
        pass

    def run_historical(self, start_time, current_time):
        return self.process(current_time - timedelta(hours=4), current_time)