import os
import time
import requests
import pandas as pd
from datetime import datetime, timedelta, timezone

def ms(ts: datetime) -> int:
    """Вернуть unix ms для datetime (UTC)."""
    return int(ts.replace(tzinfo=timezone.utc).timestamp() * 1000)

def _sanitize_symbol(symbol: str) -> str:
    return symbol.replace("/", "").replace("-", "").upper()

def fetch_klines_binance(symbol: str, interval: str, start_time_ms: int, limit: int = 1000):
    """
    Возвращает ответ от /api/v3/klines (список свечей) начиная с start_time_ms (если есть).
    limit не более 1000.
    """
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_time_ms,
        "limit": limit
    }
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()

def download_binance_1m_year(symbol: str = 'BTCUSDT',
                            out_dir: str = "files/history_data",
                            filename_template: str = "fulldata_current_history_data_binance_{symbol}.csv",
                            days_back: int = 365,
                            sleep_between_requests: float = 0.15):
    """
    Скачивает 1m свечи Binance для symbol за последние days_back дней и сохраняет в CSV.
    - symbol: строка символа, например "BTCUSDT" или "BTC/USDT"
    - out_dir: директория для сохранения
    - filename_template: шаблон имени файла с {symbol}
    - days_back: сколько дней назад (по умолчанию 365)
    - sleep_between_requests: пауза между HTTP запросами (сек)
    """
    symbol_clean = _sanitize_symbol(symbol)
    end_dt = datetime.now(timezone.utc).replace(second=0, microsecond=0)  # текущий момент (UTC)
    start_dt = end_dt - timedelta(days=days_back)

    all_rows = []
    cur_day = start_dt.date()
    last_day = end_dt.date()

    print(f"Начинаю загрузку {symbol_clean} с {start_dt.isoformat()} по {end_dt.isoformat()}")

    while cur_day <= last_day:
        day_start_dt = datetime(cur_day.year, cur_day.month, cur_day.day, tzinfo=timezone.utc)
        day_end_dt = day_start_dt + timedelta(days=1) - timedelta(minutes=1)  # включительно до 23:59
        start_ms = ms(day_start_dt)
        day_end_ms = ms(day_end_dt)

        # Внутренний цикл: т.к. limit=1000 < 1440, делаем несколько запросов за день
        fetch_start_ms = start_ms
        day_rows = []
        while fetch_start_ms <= day_end_ms:
            try:
                klines = fetch_klines_binance(symbol_clean, "1m", fetch_start_ms, limit=1000)
            except Exception as e:
                print(f"Ошибка при запросе {symbol_clean} {datetime.fromtimestamp(fetch_start_ms/1000, tz=timezone.utc)}: {e}")
                # простая логика: подождать и повторить
                time.sleep(5)
                continue

            if not klines:
                # нет данных дальше
                break

            # Добавляем все свечи, но фильтруем по дню (иногда может прилететь свечи за следующий день)
            for k in klines:
                open_time = int(k[0])
                if open_time < start_ms or open_time > day_end_ms:
                    continue
                # поля: open_time, open, high, low, close, volume, close_time, quote_asset_volume,
                # number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore
                day_rows.append({
                    "time": datetime.fromtimestamp(open_time/1000, tz=timezone.utc),
                    "open": float(k[1]),
                    "high": float(k[2]),
                    "low": float(k[3]),
                    "close": float(k[4]),
                    "volume": float(k[5]),
                    "close_time": datetime.fromtimestamp(int(k[6])/1000, tz=timezone.utc),
                    "quote_asset_volume": float(k[7]),
                    "num_trades": int(k[8]),
                    "taker_buy_base": float(k[9]),
                    "taker_buy_quote": float(k[10])
                })

            # определить следующий старт: берем последний open_time из полученных свечей + 1 мин
            last_open_time = int(klines[-1][0])
            # если последний открывашийся тик уже совпадает или не продвигается, чтобы избежать бесконечного цикла, увеличим:
            next_start = last_open_time + 60_000
            if next_start <= fetch_start_ms:
                next_start = fetch_start_ms + 60_000
            fetch_start_ms = next_start

            time.sleep(sleep_between_requests)

        if day_rows:
            # отсортируем и присоединим
            day_rows.sort(key=lambda r: r["time"])
            all_rows.extend(day_rows)
            print(f"День {cur_day} скачан, свечей добавлено: {len(day_rows)}")
        else:
            print(f"День {cur_day} — данных не найдено.")

        cur_day = cur_day + timedelta(days=1)

    # Создаём DataFrame и сохраняем
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    df = pd.DataFrame(all_rows)
    if df.empty:
        print("Данных не найдено за запрошенный период.")
        return

    # Включаем индекс временной метки в отдельный столбец (возможно удобнее сохранить как строку ISO)
    df["open_time_iso"] = df["time"].dt.strftime("%Y-%m-%d %H:%M:%S%z")
    # Сохраняем CSV
    filename = filename_template.format(symbol=symbol_clean)
    fullpath = os.path.join(out_dir, filename)
    df.to_csv(fullpath, index=False)
    print(f"Сохранено {len(df)} строк в {fullpath}")
    return fullpath

# Пример вызова:
# download_binance_1m_year("BTCUSDT")
# или за год до определённой даты: изменить days_back, например days_back=365