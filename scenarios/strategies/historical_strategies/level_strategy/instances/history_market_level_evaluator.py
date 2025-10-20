from scenarios.parsers.history_market_parser.abstracts.history_market_parser import HistoryMarketParser
from scenarios.strategies.historical_strategies.level_strategy.abstracts.level_evaluator import LevelEvaluator


class HistoryMarketLevelEvaluator(LevelEvaluator):
    def __init__(self, history_market_parser: HistoryMarketParser):
        super().__init__()
        self.history_market_parser = history_market_parser

    def evaluate_support(self, time_range=100):
        import numpy as np
        from scipy.stats import gaussian_kde
        from scipy.signal import find_peaks

        # Get the last 'time_range' records from history_df
        df = self.history_market_parser.history_df.tail(time_range)
        if df.empty or len(df) < 3:
            return {'price1': 0.0, 'price2': 0.0, 'persistence': 0.0}

        lows = df['low'].values
        volumes = df['volume'].values

        # Step 1: Use KDE to estimate density of low prices
        try:
            kde = gaussian_kde(lows)
        except Exception as e:
            print(f"Warning: KDE failed: {e}")
            return {'price1': 0.0, 'price2': 0.0, 'persistence': 0.0}

        x = np.linspace(min(lows), max(lows), 1000)
        density = kde(x)

        # Step 2: Find peaks in density (candidate support levels)
        peaks, properties = find_peaks(density, height=0.3 * np.max(density), distance=10)
        if len(peaks) == 0:
            return {'price1': 0.0, 'price2': 0.0, 'persistence': 0.0}

        # Step 3: Form clusters around density peaks
        clusters = []
        for peak_idx in peaks:
            peak_price = x[peak_idx]
            # Define cluster range as where density > 0.2 * peak_height
            peak_height = properties['peak_heights'][peaks.tolist().index(peak_idx)]
            cluster_mask = density > 0.2 * peak_height
            cluster_indices = np.where(cluster_mask)[0]
            if len(cluster_indices) < 3:
                continue  # Small cluster, skip
            cluster_prices = x[cluster_indices]
            min_p = np.min(cluster_prices)
            max_p = np.max(cluster_prices)
            price_range = max_p - min_p + 1e-6

            # Step 4: Analyze historical lows in this range
            hist_mask = (lows >= min_p) & (lows <= max_p)
            num_touches = np.sum(hist_mask)
            if num_touches < 3:
                continue  # Require at least 3 touches for robustness
            avg_volume = np.mean(volumes[hist_mask])
            density_avg = np.mean(density[cluster_indices])
            variance = np.var(cluster_prices)
            if variance < 1e-6:
                continue  # Too narrow, possible noise

            # Step 5: Calculate persistence score
            # Score = density * num_touches * avg_volume / sqrt(variance)
            score = density_avg * num_touches * avg_volume / np.sqrt(variance + 1e-6)

            clusters.append({
                'price1': min_p,
                'price2': max_p,
                'persistence': score
            })

        # Step 6: Select the best cluster
        if not clusters:
            return {'price1': 0.0, 'price2': 0.0, 'persistence': 0.0}

        best_cluster = max(clusters, key=lambda c: c['persistence'])
        max_persistence = max(c['persistence'] for c in clusters)
        persistence_normalized = best_cluster['persistence'] / max_persistence if max_persistence > 0 else 0.0
        # Cap persistence based on number of touches relative to time_range
        persistence_normalized = min(persistence_normalized, np.sum((lows >= best_cluster['price1']) & (lows <= best_cluster['price2'])) / time_range)

        return {
            'price1': best_cluster['price1'],
            'price2': best_cluster['price2'],
            'persistence': persistence_normalized
        }