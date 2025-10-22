import os
import json
import gc
import logging
from typing import List, Dict, Any, Optional, Tuple, NamedTuple
import numpy as np
import pandas as pd
from math import inf
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import joblib
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data.sampler import WeightedRandomSampler
from sklearn.metrics import roc_auc_score
import pywt
import psutil

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

n_feat = None

# Configuration with validation
class Config:
    def __init__(self):
        self.atr_period: int = 14
        self.zz_mult: float = 0.8  # Multiplier for zigzag threshold, tuned empirically for volatility
        self.min_bars: int = 3  # Minimum bars between swings to avoid noise
        self.min_atr: float = 1e-6  # Minimum ATR to prevent division by zero
        self.min_channel_pairs: int = 3  # Minimum swing pairs for a valid channel
        self.r2_thresh: float = 0.2  # R2 threshold for line fit quality
        self.cv_width_thresh: float = 0.45  # CV threshold for channel width consistency
        self.tolerance: float = 0.05  # Tolerance for ascending/descending checks
        self.wavelet_level: int = 3  # Wavelet decomposition level for noise reduction
        self.seq_len: int = 256  # Sequence length for model input
        self.epochs: int = 100  # Maximum training epochs
        self.batch_size: int = 128  # Batch size for training
        self.lr: float = 5e-4  # Initial learning rate
        self.patience: int = 10  # Patience for early stopping
        self.min_delta: float = 0.0005  # Minimum delta for early stopping
        self.d_model: int = 128  # Transformer model dimension
        self.n_heads: int = 4  # Number of attention heads
        self.n_encoder_layers: int = 3  # Number of encoder layers
        self.dim_feedforward: int = 512  # Feedforward dimension
        self.dropout: float = 0.1  # Dropout rate

CONFIG = Config()

# Utilities and Indicators
def compute_indicators(df: pd.DataFrame, atr_period: int = CONFIG.atr_period, wavelet_level: int = CONFIG.wavelet_level) -> pd.DataFrame:
    """
    Computes technical indicators and wavelet decomposition for OHLCV DataFrame.

    :param df: Input DataFrame with OHLCV columns.
    :param atr_period: Period for ATR calculation.
    :param wavelet_level: Level for wavelet decomposition.
    :return: DataFrame with added indicators.
    :raises ValueError: If required columns are missing.
    """
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"DataFrame must contain {required_cols} columns.")

    # ATR (in-place operations where possible)
    df['hl'] = df['high'] - df['low']
    df['hc'] = (df['high'] - df['close'].shift(1)).abs()
    df['lc'] = (df['low'] - df['close'].shift(1)).abs()
    df['tr'] = df[['hl', 'hc', 'lc']].max(axis=1)
    df['atr'] = df['tr'].rolling(window=atr_period, min_periods=1).mean()  # Use rolling mean for exact Wilder ATR

    # SMA
    df['sma20'] = df['close'].rolling(20, min_periods=1).mean()
    df['sma50'] = df['close'].rolling(50, min_periods=1).mean()

    # Momentum
    df['ret1'] = df['close'].pct_change().fillna(0)
    df['roc5'] = df['close'].pct_change(5).fillna(0)

    # Exact Wilder RSI using rolling
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14, min_periods=1).mean()
    roll_down = down.rolling(14, min_periods=1).mean()
    rs = roll_up / (roll_down + 1e-9)
    df['rsi14'] = 100 - 100 / (1 + rs)

    # Volume (normalized later per window)
    df['vol20'] = df['volume'].rolling(20, min_periods=1).mean()

    # Wavelet decomposition on close
    coeffs = pywt.wavedec(df['close'].values, 'db4', level=wavelet_level)
    approx = pywt.waverec([coeffs[0]] + [np.zeros_like(c) for c in coeffs[1:]], 'db4')
    details = df['close'].values - approx
    df['wavelet_approx'] = approx
    df['wavelet_detail'] = details

    # Cleanup
    df.drop(columns=['hl', 'hc', 'lc', 'tr'], inplace=True)

    # Fill NaNs
    df = df.ffill().bfill()  # Forward and backward fill for consistency

    return df

# Swing namedtuple
Swing = NamedTuple('Swing', [('idx', int), ('price', float), ('type', str)])

def zigzag_by_atr(df: pd.DataFrame, atr_col: str = 'atr', mult: float = CONFIG.zz_mult,
                  min_bars: int = CONFIG.min_bars, min_atr: float = CONFIG.min_atr) -> List[Swing]:
    """
    Computes ZigZag swings based on ATR.

    :param df: DataFrame with high, low, atr.
    :param atr_col: Column name for ATR.
    :param mult: Multiplier for threshold.
    :param min_bars: Min bars between swings.
    :param min_atr: Min ATR value.
    :return: List of Swing namedtuples.
    """
    highs = df['high'].values
    lows = df['low'].values
    atrs = df[atr_col].values
    N = len(highs)
    swings = []
    if N < min_bars:
        return swings

    # Detect initial direction
    if highs[1] > highs[0]:
        last_ext_price = highs[0]
        last_ext_type = 'high'
    else:
        last_ext_price = lows[0]
        last_ext_type = 'low'
    last_ext_idx = 0
    swings.append(Swing(last_ext_idx, last_ext_price, last_ext_type))
    last_atr = atrs[0] if not np.isnan(atrs[0]) else min_atr

    for i in range(1, N):
        atr = atrs[i] if not np.isnan(atrs[i]) else last_atr
        atr = max(atr, min_atr)
        last_atr = atr
        thr = mult * atr

        if last_ext_type == 'high':
            if lows[i] < last_ext_price - thr and (i - last_ext_idx) >= min_bars:
                last_ext_type = 'low'
                last_ext_price = lows[i]
                last_ext_idx = i
                swings.append(Swing(last_ext_idx, last_ext_price, last_ext_type))
            elif highs[i] > last_ext_price:
                last_ext_price = highs[i]
                last_ext_idx = i
                swings[-1] = Swing(last_ext_idx, last_ext_price, last_ext_type)
        else:
            if highs[i] > last_ext_price + thr and (i - last_ext_idx) >= min_bars:
                last_ext_type = 'high'
                last_ext_price = highs[i]
                last_ext_idx = i
                swings.append(Swing(last_ext_idx, last_ext_price, last_ext_type))
            elif lows[i] < last_ext_price:
                last_ext_price = lows[i]
                last_ext_idx = i
                swings[-1] = Swing(last_ext_idx, last_ext_price, last_ext_type)

    return swings

# Channel namedtuple
Channel = NamedTuple('Channel', [('start_idx', int), ('end_idx', int), ('lows_idx', List[int]), ('highs_idx', List[int]),
                                 ('slope_low', float), ('slope_high', float), ('r2_low', float), ('r2_high', float),
                                 ('widths', List[float])])

def find_channels_from_swings(swings: List[Swing], min_pairs: int = CONFIG.min_channel_pairs,
                              r2_thresh: float = CONFIG.r2_thresh, cv_width_thresh: float = CONFIG.cv_width_thresh,
                              tolerance: float = CONFIG.tolerance, channel_type: str = 'ascending') -> List[Channel]:
    """
    Finds channels from swings.

    :param swings: List of Swing.
    :param min_pairs: Min pairs for channel.
    :param r2_thresh: R2 threshold.
    :param cv_width_thresh: CV threshold.
    :param tolerance: Tolerance for trend.
    :param channel_type: 'ascending', 'descending', 'horizontal'.
    :return: List of Channel namedtuples.
    """
    if len(swings) < 2 * min_pairs:
        return []

    channels = []
    # Pair consecutive highs and lows
    highs = [s for s in swings if s.type == 'high']
    lows = [s for s in swings if s.type == 'low']
    num_pairs = min(len(highs), len(lows))

    if num_pairs < min_pairs:
        return []

    for start in range(num_pairs - min_pairs + 1):
        for length in range(min_pairs, num_pairs - start + 1):
            lows_w = [lows[start + i].price for i in range(length)]
            highs_w = [highs[start + i].price for i in range(length)]
            xs = np.array([lows[start + i].idx for i in range(length)])

            # Trend checks
            if channel_type == 'ascending':
                if not all(lows_w[i] >= lows_w[i-1] - tolerance * lows_w[i-1] for i in range(1, length)):
                    continue
                if not all(highs_w[i] >= highs_w[i-1] - tolerance * highs_w[i-1] for i in range(1, length)):
                    continue
            elif channel_type == 'descending':
                if not all(lows_w[i] <= lows_w[i-1] + tolerance * lows_w[i-1] for i in range(1, length)):
                    continue
                if not all(highs_w[i] <= highs_w[i-1] + tolerance * highs_w[i-1] for i in range(1, length)):
                    continue
            elif channel_type == 'horizontal':
                slope_thresh = 1e-4  # Small slope for horizontal
                # Compute slopes first
            else:
                raise ValueError(f"Unknown channel_type: {channel_type}")

            try:
                coeffs_low = np.polyfit(xs, lows_w, 1)
                ypred_low = np.polyval(coeffs_low, xs)
                ss_res_low = np.sum((np.array(lows_w) - ypred_low) ** 2)
                ss_tot_low = np.sum((np.array(lows_w) - np.mean(lows_w)) ** 2)
                r2_low = 1 - ss_res_low / ss_tot_low if ss_tot_low != 0 else 0.0

                coeffs_high = np.polyfit(xs, highs_w, 1)
                ypred_high = np.polyval(coeffs_high, xs)
                ss_res_high = np.sum((np.array(highs_w) - ypred_high) ** 2)
                ss_tot_high = np.sum((np.array(highs_w) - np.mean(highs_w)) ** 2)
                r2_high = 1 - ss_res_high / ss_tot_high if ss_tot_high != 0 else 0.0

                if channel_type == 'horizontal' and (abs(coeffs_low[0]) > slope_thresh or abs(coeffs_high[0]) > slope_thresh):
                    continue

                widths = np.array(highs_w) - np.array(lows_w)
                mean_w = np.mean(widths)
                if mean_w == 0:
                    continue
                cv = np.std(widths) / mean_w

                if r2_low >= r2_thresh and r2_high >= r2_thresh and cv <= cv_width_thresh:
                    lows_idx = [swings.index(lows[start + i]) for i in range(length)]
                    highs_idx = [swings.index(highs[start + i]) for i in range(length)]
                    start_idx = min(lows[start].idx, highs[start].idx)
                    end_idx = max(lows[start + length - 1].idx, highs[start + length - 1].idx)
                    ch = Channel(start_idx, end_idx, lows_idx, highs_idx, coeffs_low[0], coeffs_high[0], r2_low, r2_high, widths.tolist())
                    channels.append(ch)
            except Exception as e:
                logger.warning(f"Error in channel fit: {e}")
                continue

    channels.sort(key=lambda c: (c.r2_low + c.r2_high) / 2 + (c.end_idx - c.start_idx) / len(swings), reverse=True)  # Prioritize quality and length
    return channels

# Dataset
class SequenceDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]

# Model
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 512):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class CorridorTransformer(nn.Module):
    def __init__(self, n_features: int, d_model: int = CONFIG.d_model, n_heads: int = CONFIG.n_heads,
                 n_encoder_layers: int = CONFIG.n_encoder_layers, dim_feedforward: int = CONFIG.dim_feedforward,
                 dropout: float = CONFIG.dropout, max_seq_len: int = CONFIG.seq_len):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(n_features, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=dim_feedforward,
                                                 dropout=dropout, activation='gelu', batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_encoder_layers)
        self.fc = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor, src_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.embedding(x) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        x = self.pos_encoder(x)
        out = self.transformer_encoder(x, src_key_padding_mask=src_mask)
        pooled = out.mean(dim=1)
        return self.fc(pooled).squeeze(1)

# Trainer
class CorridorModelTrainer:
    def __init__(self, model_name: str, pair: str, model_dir: str = 'models', device: Optional[str] = None):
        self.model_name = model_name
        self.pair = pair
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        self.save_path = os.path.join(self.model_dir, f"model_{self.model_name}_{self.pair}")
        os.makedirs(self.save_path, exist_ok=True)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = CONFIG
        logger.info(f"Initialized trainer for {model_name}_{self.pair} on {self.device}")

    def fit(self, history_df: pd.DataFrame) -> Dict[str, Any]:
        cfg = self.config
        def log_memory_usage(stage: str):
            process = psutil.Process(os.getpid())
            mem_rss = process.memory_info().rss / (1024 ** 2)
            logger.debug(f"[{stage}] Memory usage: Process RSS: {mem_rss:.2f} MB")

        logger.info("Starting fit method")
        log_memory_usage("Start of fit")

        if not all(col in history_df.columns for col in ['open', 'high', 'low', 'close', 'volume']):
            raise ValueError("history_df must have OHLCV columns")

        df = history_df.drop_duplicates().sort_index().reset_index(drop=True)
        log_memory_usage("After preparing DF")

        # Compute indicators once
        logger.info("Computing indicators on full DF")
        df = compute_indicators(df, atr_period=cfg.atr_period, wavelet_level=cfg.wavelet_level)
        log_memory_usage("After computing indicators")

        feature_cols = ['close', 'sma20', 'sma50', 'atr', 'ret1', 'roc5', 'rsi14', 'vol20', 'wavelet_approx', 'wavelet_detail']

        # Prepare windows using generator to save memory
        X_windows = []
        y_windows = []
        for end_idx in range(cfg.seq_len - 1, len(df)):
            start_idx = end_idx - cfg.seq_len + 1
            window_df = df.iloc[start_idx:end_idx + 1].reset_index(drop=True)

            # Per window winsorizing
            for col in feature_cols:
                q_low = window_df[col].quantile(0.01)
                q_high = window_df[col].quantile(0.99)
                window_df[col] = window_df[col].clip(q_low, q_high)

            # Normalization per window
            first_close = window_df['close'].iloc[0]
            if first_close == 0:
                first_close = 1e-9
            for col in ['close', 'sma20', 'sma50', 'wavelet_approx']:
                window_df[col] = (window_df[col] / first_close - 1) * 100
            for col in ['atr', 'wavelet_detail']:
                window_df[col] = (window_df[col] / first_close) * 100
            mean_vol = window_df['vol20'].mean()
            window_df['vol20'] = window_df['vol20'] / mean_vol if mean_vol != 0 else 0

            swings = zigzag_by_atr(window_df, mult=cfg.zz_mult, min_bars=cfg.min_bars, min_atr=cfg.min_atr)
            channels = find_channels_from_swings(swings, min_pairs=cfg.min_channel_pairs, r2_thresh=cfg.r2_thresh,
                                                 cv_width_thresh=cfg.cv_width_thresh, tolerance=cfg.tolerance)
            in_channel = any(ch.start_idx <= len(window_df) - 1 <= ch.end_idx for ch in channels)
            y_windows.append(1 if in_channel else 0)

            X_windows.append(window_df[feature_cols].values.astype(np.float32))
            gc.collect()  # Collect garbage per iteration

        if not X_windows:
            raise ValueError("No windows prepared")

        X = np.stack(X_windows)
        y = np.array(y_windows, dtype=np.float32)
        logger.info(f"Prepared {len(X_windows)} windows with {X.shape[2]} features each")
        logger.info(f"Positive labels: {np.sum(y)} out of {len(y)}")
        log_memory_usage("After preparing windows")

        n_feat = X.shape[2]

        # Time-series split (no shuffle)
        n = len(X)
        split = int(n * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
        logger.info(f"Split data: Train {len(X_train)}, Val {len(X_val)}")
        log_memory_usage("After splitting data")

        # Scaler fit only on train
        logger.info("Fitting scaler on train")
        scaler = StandardScaler()
        X_train_flat = X_train.reshape(-1, n_feat)
        scaler.fit(X_train_flat)
        joblib.dump(scaler, os.path.join(self.save_path, 'scaler.pkl'))
        X_train_scaled = scaler.transform(X_train_flat).reshape(X_train.shape).astype(np.float32)
        X_val_flat = X_val.reshape(-1, n_feat)
        X_val_scaled = scaler.transform(X_val_flat).reshape(X_val.shape).astype(np.float32)
        del X, X_train, X_val, X_train_flat, X_val_flat, X_windows
        gc.collect()
        log_memory_usage("After scaling")

        # Weights
        num_pos = np.sum(y_train == 1)
        num_neg = np.sum(y_train == 0)
        if num_pos == 0:
            raise RuntimeError("No positives in train set.")
        pos_weight = torch.tensor([num_neg / num_pos], dtype=torch.float32).to(self.device)

        # Datasets and loaders
        logger.info("Creating datasets and loaders")
        train_ds = SequenceDataset(X_train_scaled, y_train)
        val_ds = SequenceDataset(X_val_scaled, y_val)
        train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)  # Shuffle for better training
        val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
        log_memory_usage("After creating loaders")

        # Model
        logger.info("Initializing model")
        model = CorridorTransformer(n_features=n_feat).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
        scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=cfg.patience // 2)  # On AUC
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        log_memory_usage("After initializing model")

        best_auc = 0.0
        history = {'train_loss': [], 'val_loss': [], 'val_auc': []}
        no_improve = 0

        logger.info("Starting training loop")
        for ep in range(cfg.epochs):
            model.train()
            train_losses = []
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                logits = model(xb)
                loss = loss_fn(logits, yb)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                opt.step()
                train_losses.append(loss.item())
            avg_train = np.mean(train_losses)
            history['train_loss'].append(avg_train)

            model.eval()
            val_losses = []
            val_logits = []
            with torch.no_grad():
                for xb, yb in val_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
                    logits = model(xb)
                    loss = loss_fn(logits, yb)
                    val_losses.append(loss.item())
                    val_logits.extend(torch.sigmoid(logits).cpu().numpy())
            avg_val = np.mean(val_losses)
            val_auc = roc_auc_score(y_val, val_logits) if len(np.unique(y_val)) > 1 else 0.5
            history['val_loss'].append(avg_val)
            history['val_auc'].append(val_auc)
            scheduler.step(val_auc)  # Step on AUC

            logger.info(f"Epoch {ep + 1}/{cfg.epochs} train={avg_train:.4f} val_loss={avg_val:.4f} val_auc={val_auc:.4f} lr={opt.param_groups[0]['lr']:.6f}")

            if val_auc > best_auc + cfg.min_delta:
                best_auc = val_auc
                torch.save(model.state_dict(), os.path.join(self.save_path, 'model.pt'))
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= cfg.patience:
                    logger.info(f"Early stopping at {ep + 1}")
                    break

        meta = {
            'model_name': self.model_name,
            'pair': self.pair,
            'feature_cols': feature_cols,
            'config': vars(cfg),
            'device': self.device,
            'train_history': history,
            'n_positive': int(np.sum(y == 1)),
            'n_samples': int(len(y)),
        }
        with open(os.path.join(self.save_path, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)
        logger.info(f"Model saved to {self.save_path}")
        log_memory_usage("End of fit")
        return meta

# Predictor
class CorridorModelPredictor:
    def __init__(self, model_name: str, pair: str, model_dir: str = 'models', device: Optional[str] = None):
        self.model_name = model_name
        self.pair = pair
        self.model_dir = model_dir
        self.base_path = os.path.join(self.model_dir, f"model_{self.model_name}_{self.pair}")
        if not os.path.exists(self.base_path):
            raise FileNotFoundError(f"Model not found: {self.base_path}")
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        with open(os.path.join(self.base_path, 'meta.json'), 'r', encoding='utf-8') as f:
            self.meta = json.load(f)
        self.config = Config()  # Reinit from class
        for k, v in self.meta.get('config', {}).items():
            setattr(self.config, k, v)
        self.scaler = joblib.load(os.path.join(self.base_path, 'scaler.pkl'))
        self.seq_len = self.config.seq_len
        self.feature_cols = self.meta['feature_cols']
        n_feat = len(self.feature_cols)
        self.model = CorridorTransformer(n_features=n_feat, max_seq_len=self.seq_len).to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(self.base_path, 'model.pt'), map_location=self.device, weights_only=True))
        self.model.eval()
        logger.info(f"Loaded predictor for {model_name}_{self.pair} on {self.device}")

    def _make_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_feat = compute_indicators(df)
        return df_feat

    def predict(self, day_df: pd.DataFrame, try_windows: Optional[List[int]] = None,
                aggregate: str = 'weighted_mean') -> Dict[str, Any]:
        cfg = self.config
        df = day_df.drop_duplicates().sort_index().reset_index(drop=True)
        if len(df) < 16:
            return {'probability_pct': 0.0, 'raw_probs': [], 'chosen_window': None,
                    'explanation': 'Not enough data.'}
        df_feat = self._make_features(df)
        n = len(df_feat)
        if try_windows is None:
            try_windows = [self.seq_len // (2 ** i) for i in range(5) if self.seq_len // (2 ** i) >= 16]

        probs = []
        details = []
        for w in sorted(try_windows, reverse=True):
            if n < w:
                continue
            window = df_feat.iloc[-w:].reset_index(drop=True)

            # Per window processing
            for col in self.feature_cols:
                q_low = window[col].quantile(0.01)
                q_high = window[col].quantile(0.99)
                window[col] = window[col].clip(q_low, q_high)

            first_close = window['close'].iloc[0]
            if first_close == 0:
                first_close = 1e-9
            for col in ['close', 'sma20', 'sma50', 'wavelet_approx']:
                window[col] = (window[col] / first_close - 1) * 100
            for col in ['atr', 'wavelet_detail']:
                window[col] = (window[col] / first_close) * 100
            mean_vol = window['vol20'].mean()
            window['vol20'] = window['vol20'] / mean_vol if mean_vol != 0 else 0

            Xw = window[self.feature_cols].values.astype(np.float32)
            Xw_scaled = self.scaler.transform(Xw)

            # Pad to seq_len
            pad_len = self.seq_len - w
            if pad_len > 0:
                pad = np.zeros((pad_len, n_feat), dtype=np.float32)
                Xw_scaled = np.vstack([pad, Xw_scaled])
                mask = torch.ones(1, self.seq_len, dtype=torch.bool)
                mask[:, :pad_len] = False
            else:
                Xw_scaled = Xw_scaled[-self.seq_len:]
                mask = None

            xb = torch.from_numpy(Xw_scaled).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logit = self.model(xb, src_mask=mask)
                prob = torch.sigmoid(logit).item()
            probs.append(prob)
            details.append({'window': w, 'prob': prob})

        if not probs:
            return {'probability_pct': 0.0, 'raw_probs': [], 'chosen_window': None,
                    'explanation': 'Not enough data.'}

        if aggregate == 'max':
            p = max(probs)
        elif aggregate == 'mean':
            p = np.mean(probs)
        elif aggregate == 'weighted_mean':
            weights = [d['window'] for d in details]
            p = np.average(probs, weights=weights)
        else:
            raise ValueError(f"Unknown aggregate: {aggregate}")

        best_detail = max(details, key=lambda d: d['prob'])
        chosen_w = best_detail['window']
        last_window = df_feat.iloc[-chosen_w:].reset_index(drop=True)
        swings = zigzag_by_atr(last_window, mult=cfg.zz_mult, min_bars=cfg.min_bars, min_atr=cfg.min_atr)
        channels = find_channels_from_swings(swings, min_pairs=cfg.min_channel_pairs, r2_thresh=cfg.r2_thresh,
                                             cv_width_thresh=cfg.cv_width_thresh, tolerance=cfg.tolerance)
        in_channel = False
        pos_in_channel = None
        channel_info = None
        last_bar_idx = len(last_window) - 1
        for ch in channels:
            if ch.start_idx <= last_bar_idx <= ch.end_idx:
                in_channel = True
                lows = [swings[i].price for i in ch.lows_idx]
                highs = [swings[i].price for i in ch.highs_idx]
                if len(lows) >= 2 and len(highs) >= 2:
                    xs = np.array([swings[i].idx for i in ch.lows_idx])
                    coeff_low = np.polyfit(xs, lows, 1)
                    coeff_high = np.polyfit(xs, highs, 1)
                    low_line = np.polyval(coeff_low, last_bar_idx)
                    high_line = np.polyval(coeff_high, last_bar_idx)
                    last_price = last_window['close'].iloc[-1]
                    width = high_line - low_line if high_line - low_line != 0 else 1e-9
                    pos_in_channel = (last_price - low_line) / width
                    channel_info = {'low_line': low_line, 'high_line': high_line, 'pos': pos_in_channel,
                                    'r2_low': ch.r2_low, 'r2_high': ch.r2_high, 'width_mean': np.mean(ch.widths)}
                break

        ret = {
            'probability_pct': p * 100.0,
            'raw_probs': details,
            'chosen_window': best_detail,
            'in_channel_heuristic': in_channel,
            'pos_in_channel': pos_in_channel,
            'channel_info': channel_info,
            'explanation': f"Aggregated over {len(details)} windows with '{aggregate}'."
        }

        return ret