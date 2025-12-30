import math
from datetime import datetime
from statistics import median
from typing import List, Dict, Any, Optional

# ---------- константы ----------
Z_WILSON = 1.96
P0_BAYES = 0.8

ALPHA = 0.6  # вес Wilson
BETA = 0.25  # вес log(count)
GAMMA = 0.15  # вес size

DEFAULT_MIN_TRADES = 40
DEFAULT_WILSON_MIN = 0.7
DEFAULT_K_MIN = 4

try:
    from scipy.special import betainc  # regularized incomplete beta I_x(a,b)
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


class LeaderboardAnalyzer:
    """
    Класс для расчёта метрик/скоров по traders.
    Инициализация: TraderScorer(traders, alpha=..., beta=..., gamma=...)
    Вызов: scorer.process() -> {"scored": [...], "collective": {...}}
    """

    def __init__(self,
                 traders: List[Dict[str, Any]],
                 alpha: float = ALPHA,
                 beta: float = BETA,
                 gamma: float = GAMMA):
        self.traders = traders
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    # ---------- низкоуровневые утилиты ----------
    def _parse_dt(self, s: str) -> datetime:
        return datetime.fromisoformat(s) if s else None

    def _closed_trades(self, trades: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        return [t for t in trades if t.get("close_date")]

    def _compute_n_k(self, closed: List[Dict[str, Any]]) -> (int, int):
        n = len(closed)
        k = sum(1 for t in closed if t.get("is_profit"))
        return n, k

    def _compute_winrate(self, provided: Optional[float], k: int, n: int) -> float:
        if provided is not None:
            return provided
        return (k / n) if n > 0 else 0.0

    def _median_abs_pnl(self, closed: List[Dict[str, Any]]) -> float:
        vals = [abs(t.get("pnl", 0.0)) for t in closed if t.get("pnl") is not None]
        return median(vals) if vals else 0.0

    def _days_active(self, closed: List[Dict[str, Any]]) -> int:
        if not closed:
            return 0
        opens = [self._parse_dt(t["open_date"]) for t in closed]
        closes = [self._parse_dt(t["close_date"]) for t in closed]
        first, last = min(opens), max(closes)
        days = max(1, (last - first).days or 1)
        return days

    # ---------- статистические функции ----------
    def _wilson_lower(self, k: int, n: int, z: float = Z_WILSON) -> float:
        if n == 0:
            return 0.0
        phat = k / n
        denom = 1 + z * z / n
        num = phat + z * z / (2 * n) - z * math.sqrt(phat * (1 - phat) / n + z * z / (4 * n * n))
        return num / denom

    def _bayes_p_ge(self, p0: float, k: int, n: int, a: float = 1.0, b: float = 1.0) -> Optional[float]:
        if not SCIPY_AVAILABLE:
            return None
        a_post = a + k
        b_post = b + (n - k)
        cdf = betainc(a_post, b_post, p0)
        return 1.0 - cdf

    # ---------- вычисление метрик по одному трейдеру ----------
    def compute_trader_metrics(self, trader: Dict[str, Any]) -> Dict[str, Any]:
        trades = trader.get("trades", [])
        closed = self._closed_trades(trades)
        n, k = self._compute_n_k(closed)
        winrate = self._compute_winrate(trader.get("winrate"), k, n)
        med_abs = self._median_abs_pnl(closed)
        days = self._days_active(closed)
        wilson = self._wilson_lower(k, n)
        bayes_p80 = self._bayes_p_ge(P0_BAYES, k, n)
        return {
            "id": trader.get("url"),
            "n": n,
            "k": k,
            "winrate": winrate,
            "median_abs_pnl": med_abs,
            "days_active": days,
            "wilson_lb": wilson,
            "bayes_p80": bayes_p80,
            "raw": trader,
        }

    # ---------- нормализация и скоринг ----------
    def _compute_normalizers(self, metrics: List[Dict[str, Any]]) -> Dict[str, float]:
        max_n = max((m["n"] for m in metrics), default=1)
        max_med = max((m["median_abs_pnl"] for m in metrics), default=1.0)
        denom_log = math.log(1 + max_n) if max_n > 0 else 1.0
        return {"max_n": max_n, "max_med": max_med, "denom_log": denom_log}

    def _score_one(self, m: Dict[str, Any], norms: Dict[str, float]) -> Dict[str, Any]:
        wilson_norm = m["wilson_lb"]
        log_count = math.log(1 + m["n"]) / norms["denom_log"] if norms["denom_log"] > 0 else 0.0
        size_norm = (m["median_abs_pnl"] / norms["max_med"]) if norms["max_med"] > 0 else 0.0
        score = self.alpha * wilson_norm + self.beta * log_count + self.gamma * size_norm
        out = dict(m)
        out.update({"wilson_norm": wilson_norm,
                    "log_count_norm": log_count,
                    "size_norm": size_norm,
                    "trader_score": score})
        return out

    def compute_scores(self) -> List[Dict[str, Any]]:
        metrics = [self.compute_trader_metrics(t) for t in self.traders]
        norms = self._compute_normalizers(metrics)
        scored = [self._score_one(m, norms) for m in metrics]
        scored.sort(key=lambda x: x["trader_score"], reverse=True)
        return scored

    # ---------- агрегирование collective ----------
    def aggregate_collective(self,
                             scored_traders: List[Dict[str, Any]],
                             min_trades: int = DEFAULT_MIN_TRADES,
                             wilson_min: float = DEFAULT_WILSON_MIN,
                             k_min: int = DEFAULT_K_MIN) -> Dict[str, Any]:
        qualified = [t for t in scored_traders if t["n"] >= min_trades and t["wilson_lb"] >= wilson_min]
        total_score = sum(t["trader_score"] for t in qualified)
        return {
            "qualified_count": len(qualified),
            "qualified_traders": qualified,
            "collective_score": total_score,
            "meets_k_min": len(qualified) >= k_min
        }

    # ---------- основной вход/выход ----------
    def process(self,
                min_trades: int = DEFAULT_MIN_TRADES,
                wilson_min: float = DEFAULT_WILSON_MIN,
                k_min: int = DEFAULT_K_MIN) -> Dict[str, Any]:
        scored = self.compute_scores()
        collective = self.aggregate_collective(scored, min_trades=min_trades, wilson_min=wilson_min, k_min=k_min)
        return {"scored": scored, "collective": collective}