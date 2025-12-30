from datetime import timedelta
import numpy as np

from files.other.purge_strategies.level_strategy.instances.history_market_level_evaluator import \
    HistoryMarketLevelEvaluator
from files.other.purge_strategies.level_strategy.instances.orderbook_level_momentum_evaluator import \
    OrderbookLevelMomentumEvaluator
from files.other.purge_strategies.level_strategy.instances.orderbook_level_simple_range_evaluator import \
    OrderbookLevelSimpleRangeEvaluator
from utils.core.functions import MarketProcess


class SimpleLevelStrategy(MarketProcess):
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

    def process(self, start_time, end_time):
        print('[SimpleLevelStrategy] Analyzing...')
        level1 = self.orderbook_level_momentum_evaluator.evaluate_support(end_time)
        level2 = self.orderbook_level_simple_range_evaluator.evaluate_support(end_time)
        level3 = self.history_market_level_evaluator.evaluate_support()

        # Step 1: Filter valid levels
        levels = [level1, level2, level3]
        valid_levels = [l for l in levels if l['price1'] > 0 and l['price2'] > 0]

        if not valid_levels:
            return [{'price1': 0.0, 'price2': 0.0, 'persistence': 0.0}]

        # Step 2: Find the intersection (conjunction) of all valid levels
        # Conjunction means the overlapping range where all levels agree
        overall_price1 = max(l['price1'] for l in valid_levels)
        overall_price2 = min(l['price2'] for l in valid_levels)

        if overall_price1 <= overall_price2:
            # There is a common intersection
            # Persistence: minimum of the persistences/qualities (strict conjunction)
            persistences = [l.get('quality', l.get('persistence', 0)) for l in valid_levels]
            persistence = min(persistences) if persistences else 0.0
            result = [{
                'price1': overall_price1,
                'price2': overall_price2,
                'persistence': persistence
            }]
        else:
            # No full intersection, find pairwise intersections or select the one with highest persistence
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
                # No intersections at all, select the level with highest persistence
                persistences = [l.get('quality', l.get('persistence', 0)) for l in valid_levels]
                best_idx = np.argmax(persistences)
                best_level = valid_levels[best_idx]
                result = [{
                    'price1': best_level['price1'],
                    'price2': best_level['price2'],
                    'persistence': persistences[best_idx]
                }]

        print(f"Final Level: {result}")

        if (result['price1'] < float(self.history_market.history_df['close'].iloc[-1]) < result['price2']):
            print('CAAATCH!!!!!!!!!!!!!!!!!!!!!!')

        return result

    def prepare(self, start_time=None, end_time=None):
        pass

    def run_historical(self, start_time, current_time):
        self.process(current_time - timedelta(hours=4), current_time)