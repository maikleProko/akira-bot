
import pandas as pd

def plot_kama(
    df,
    length=14,
    fastLength=2,
    slowLength=30,
    highlight=True
):
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt

    # --- подготовка ---
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'])
    df.set_index('time', inplace=True)

    src = df['close']

    # --- расчет ER ---
    change = src.diff()
    mom = (src - src.shift(length)).abs()
    volatility = change.abs().rolling(length).sum()
    er = np.where(volatility != 0, mom / volatility, 0)

    # --- alpha ---
    fastAlpha = 2 / (fastLength + 1)
    slowAlpha = 2 / (slowLength + 1)
    alpha = (er * (fastAlpha - slowAlpha) + slowAlpha) ** 2

    # --- KAMA (рекурсивно, как в Pine) ---
    kama = np.zeros(len(df))
    kama[0] = src.iloc[0]

    for i in range(1, len(df)):
        if np.isnan(alpha[i]):
            kama[i] = kama[i - 1]
        else:
            kama[i] = alpha[i] * src.iloc[i] + (1 - alpha[i]) * kama[i - 1]

    df['KAMA'] = kama

    # --- цвета линии ---
    colors = np.where(
        df['KAMA'] > df['KAMA'].shift(1),
        'green',
        'red'
    )
    colors[0] = 'red'

    # --- график ---
    plt.figure(figsize=(14, 7))

    plt.plot(df.index, src, color='gray', alpha=0.4, label='Close')

    for i in range(1, len(df)):
        plt.plot(
            df.index[i-1:i+1],
            df['KAMA'].iloc[i-1:i+1],
            color=colors[i],
            linewidth=2
        )

    plt.title('KAMA (Pine Script → Python)')
    plt.legend()
    plt.grid(True)
    plt.show()

    return df[['close', 'KAMA']]


df = pd.read_csv('files/history_data/binance_BTCUSDT_1m_20260101_to_20260102.csv')
plot_kama(df)