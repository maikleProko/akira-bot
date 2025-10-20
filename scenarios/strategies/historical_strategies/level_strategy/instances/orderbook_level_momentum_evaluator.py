import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

class OrderbookLevelMomentumEvaluator:
    def __init__(self, orderbook_parser):
        super().__init__()
        self.orderbook_parser = orderbook_parser

    def evaluate_support(self, time):
        snapshot = self.orderbook_parser.get_by_time(time)
        # Extract bids (support levels)
        bids = snapshot.get('bids', [])
        if not bids:
            return {'price1': 0.0, 'price2': 0.0, 'quality': 0.0}

        # Sort bids by price descending (highest price first for top bids)
        bids_sorted = sorted(bids, key=lambda x: x['price'], reverse=True)
        prices = np.array([bid['price'] for bid in bids_sorted])
        amounts = np.array([bid['amount'] for bid in bids_sorted])
        totals = np.array([bid['total'] for bid in bids_sorted])

        # Normalize prices for relative calculations (optional, kept for consistency)
        max_price = np.max(prices)
        normalized_prices = (prices / max_price) if max_price != 0 else prices

        # Step 1: Smooth the amounts to reduce noise from single bins
        sigma = max(1, len(amounts) // 20)  # Adaptive sigma based on depth
        smoothed_amounts = gaussian_filter1d(amounts, sigma=sigma)

        # Step 2: Find peaks in smoothed amounts (candidate support points)
        peaks, properties = find_peaks(smoothed_amounts, height=0.3 * np.max(smoothed_amounts), distance=3)
        if len(peaks) == 0:
            return {'price1': 0.0, 'price2': 0.0, 'quality': 0.0}

        # Step 3: Manual clustering around peaks to form areas (without sklearn)
        # Group consecutive levels around each peak if they are close and have sufficient amount
        clusters = []
        visited = np.zeros(len(prices), dtype=bool)

        for peak_idx in peaks:
            if visited[peak_idx]:
                continue

            # Start cluster from peak
            cluster_indices = [peak_idx]
            visited[peak_idx] = True

            # Expand left (higher indices, lower prices)
            i = peak_idx + 1
            while i < len(prices) and not visited[i]:
                price_diff = abs(prices[i] - prices[cluster_indices[-1]]) / prices[
                    cluster_indices[-1]] < 0.005  # 0.5% proximity
                amount_sufficient = smoothed_amounts[i] > 0.2 * smoothed_amounts[peak_idx]
                if price_diff and amount_sufficient:
                    cluster_indices.append(i)
                    visited[i] = True
                    i += 1
                else:
                    break

            # Expand right (lower indices, higher prices)
            i = peak_idx - 1
            while i >= 0 and not visited[i]:
                price_diff = abs(prices[i] - prices[cluster_indices[0]]) / prices[cluster_indices[0]] < 0.005
                amount_sufficient = smoothed_amounts[i] > 0.2 * smoothed_amounts[peak_idx]
                if price_diff and amount_sufficient:
                    cluster_indices.insert(0, i)
                    visited[i] = True
                    i -= 1
                else:
                    break

            # Filter small clusters (anti-manipulation: require at least 3 levels)
            if len(cluster_indices) < 3:
                continue

            cluster_prices = prices[cluster_indices]
            cluster_amounts = amounts[cluster_indices]
            cluster_totals = totals[cluster_indices]

            # Additional anti-spoofing: if one level dominates >70% of volume, skip
            if np.max(cluster_amounts) > 0.7 * np.sum(cluster_amounts):
                continue

            # Calculate cluster metrics
            min_p = np.min(cluster_prices)
            max_p = np.max(cluster_prices)
            price_range = max_p - min_p + 1e-6
            total_volume = np.sum(cluster_amounts)
            density = total_volume / price_range
            variance = np.var(cluster_prices)
            if variance < 1e-6:  # Too narrow, possible spoof
                continue
            num_levels = len(cluster_indices)

            # Cumulative factor: proportion of book depth covered by this cluster's max total
            cumulative_factor = np.max(cluster_totals) / np.max(totals) if np.max(totals) > 0 else 1.0

            # Closeness to top: smaller mean index = closer to best bid (favor stronger, nearer support)
            mean_index = np.mean(cluster_indices) / len(prices)
            closeness_factor = 1 / (1 + mean_index)  # Values closer to 1 for top clusters

            # Improved quality score based on criteria: total_volume * density * num_levels / price_range * cumulative * closeness
            # This rewards high volume, dense clusters with multiple levels, not too wide, deeper cumulative, closer to top
            # No normalization across clusters - absolute score for comparability across snapshots
            score = (total_volume * density * num_levels / price_range) * cumulative_factor * closeness_factor

            clusters.append({
                'price1': min_p,
                'price2': max_p,
                'quality': score,
                'indices': cluster_indices
            })

        # Step 4: Select the best cluster (highest quality)
        if not clusters:
            return {'price1': 0.0, 'price2': 0.0, 'quality': 0.0}

        best_cluster = max(clusters, key=lambda c: c['quality'])

        result = {
            'price1': best_cluster['price1'],
            'price2': best_cluster['price2'],
            'quality': best_cluster['quality']
        }

        return result