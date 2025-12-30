from datetime import timedelta
import numpy as np

from files.other.purge_strategies.level_strategy.instances.orderbook_level_momentum_evaluator import \
    OrderbookLevelMomentumEvaluator


class OrderbookLevelSimpleRangeEvaluator:
    def __init__(self, orderbook_level_momentum_evaluator: OrderbookLevelMomentumEvaluator):
        super().__init__()
        self.orderbook_level_momentum_evaluator = orderbook_level_momentum_evaluator

    def evaluate_support(self, current_time, time_range=100):

        # Step 1: Generate timestamps for the last 'time_range' minutes
        times = [current_time - timedelta(minutes=i) for i in range(1, time_range + 1)]
        times.sort()  # Ensure chronological order (oldest first)

        # Step 2: Compute individual support levels for each timestamp
        historical_supports = []
        for time in times:
            try:
                support = self.orderbook_level_momentum_evaluator.evaluate_support(time)
                if support['quality'] > 0.3:  # Filter low-quality supports
                    historical_supports.append({
                        'time': time,
                        'price1': support['price1'],
                        'price2': support['price2'],
                        'quality': support['quality']
                    })
            except Exception as e:
                continue

        # Handle edge case: insufficient valid supports
        if not historical_supports:
            return {'price1': 0.0, 'price2': 0.0, 'persistence': 0.0}

        # Step 3: Cluster historical ranges based on overlap
        clusters = []
        for support in historical_supports:
            if not support:  # Skip invalid entries
                continue
            assigned = False
            for cluster in clusters:
                # Check for overlap with cluster's range
                cluster_min = min(c['price1'] for c in cluster['supports'])
                cluster_max = max(c['price2'] for c in cluster['supports'])
                overlap = (support['price1'] <= cluster_max and support['price2'] >= cluster_min)
                if overlap:
                    cluster['supports'].append(support)
                    assigned = True
                    break
            if not assigned:
                clusters.append({'supports': [support]})

        # Step 4: Compute metrics for each cluster
        for cluster in clusters:
            supports = cluster['supports']
            # Duration: fraction of time range the cluster covers
            duration = len(supports) / time_range
            # Weighted average prices
            qualities = np.array([s['quality'] for s in supports])
            price1s = np.array([s['price1'] for s in supports])
            price2s = np.array([s['price2'] for s in supports])
            midpoints = (price1s + price2s) / 2
            # Trimmed mean for price1 and price2 (exclude top/bottom 10%)
            trim = int(0.1 * len(supports)) if len(supports) > 10 else 0
            sorted_price1s = np.sort(price1s)[trim:len(price1s) - trim] if trim > 0 else price1s
            sorted_price2s = np.sort(price2s)[trim:len(price2s) - trim] if trim > 0 else price2s
            avg_price1 = np.average(sorted_price1s, weights=qualities[trim:len(qualities) - trim] if trim > 0 else qualities) if sorted_price1s.size > 0 else np.mean(price1s)
            avg_price2 = np.average(sorted_price2s, weights=qualities[trim:len(qualities) - trim] if trim > 0 else qualities) if sorted_price2s.size > 0 else np.mean(price2s)
            # Stability: inverse of standard deviation of midpoints
            stability = 1 / (np.std(midpoints) + 1e-6)
            # Average quality
            avg_quality = np.mean(qualities)
            # Composite persistence score
            cluster['persistence'] = avg_quality * duration * stability
            cluster['price1'] = avg_price1
            cluster['price2'] = avg_price2

        # Step 5: Select the best cluster
        if not clusters:
            return {'price1': 0.0, 'price2': 0.0, 'persistence': 0.0}

        best_cluster = max(clusters, key=lambda c: c['persistence'])

        # Step 6: Normalize persistence to [0, 1]
        max_persistence = max(c['persistence'] for c in clusters)
        persistence_normalized = best_cluster['persistence'] / max_persistence if max_persistence > 0 else 0.0
        # Cap persistence based on duration
        persistence_normalized = min(persistence_normalized, len(best_cluster['supports']) / time_range)

        result = {
            'price1': best_cluster['price1'],
            'price2': best_cluster['price2'],
            'persistence': persistence_normalized
        }


        return result