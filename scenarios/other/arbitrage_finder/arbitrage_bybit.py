import json

import requests
from collections import defaultdict
import time
from pybit.unified_trading import HTTP

# По умолчанию основной публичный хост Bybit v5; для тестнета используйте "https://api-testnet.bybit.com"
BYBIT_API = "https://api.bybit.com"

# ---------- HTTP helper ----------
def get_json(url, params=None, timeout=10):
    session = HTTP(testnet=True)

    tickers = session.get_tickers(
        category="inverse",
        symbol="BTCUSD",
    )

    return tickers

# ---------- Получение инструментов и тикеров (Bybit v5) ----------
def fetch_instruments(category="spot"):
    """
    Возвращает список инструментов (список словарей) из /v5/history_market_parser/instrument.
    Учитывает разные обёртки ответа (result.list / result.data и т.п.).
    """
    url = BYBIT_API + "/v5/history_market_parser/instrument"
    params = {"category": category}
    j = get_json(url, params=params)
    res = j.get("result", j)
    # возможные контейнеры
    if isinstance(res, dict):
        for key in ("list", "data", "rows"):
            if key in res and isinstance(res[key], list):
                return res[key]
        # если result сам список — обработаем ниже
    if isinstance(res, list):
        return res
    # fallback: если в j есть 'list' на верхнем уровне
    if "list" in j and isinstance(j["list"], list):
        return j["list"]
    return []

def fetch_tickers(category="spot"):
    """
    Возвращает список тикеров из /v5/history_market_parser/tickers (result.list или result.data).
    """
    url = BYBIT_API + "/v5/history_market_parser/tickers"
    params = {"category": category}
    j = get_json(url, params=params)
    res = j.get("result", j)
    if isinstance(res, dict):
        for key in ("list", "data", "rows"):
            if key in res and isinstance(res[key], list):
                return res[key]
        # если внутри result нет явного списка — попытка найти любой список в значениях
        for v in res.values():
            if isinstance(v, list):
                return v
    if isinstance(res, list):
        return res
    if "list" in j and isinstance(j["list"], list):
        return j["list"]
    return []

# ---------- Построение графа и маппингов ----------
def build_graph_and_prices(category="spot"):
    """
    Возвращает:
      - out_edges: dict asset -> set(assets reachable за 1 сделку)
      - symbol_map: dict (base,quote) -> symbol
      - price_map: dict symbol -> {'bid': float, 'ask': float, 'base':..., 'quote':...}
    Парсит разные возможные имена полей в ответах Bybit.
    """
    instruments = fetch_instruments(category=category)
    tickers = fetch_tickers(category=category)

    # маппинг символьных имен инструмента -> instrument info
    instr_map = {}
    for it in instruments:
        # возможные имена полей: 'symbol', 'name', 'instId', 'symbolName'
        sym = it.get("symbol") or it.get("symbolName") or it.get("instId") or it.get("pair")
        # base/quote поля тоже могут называться по-разному
        base = it.get("baseCoin") or it.get("base") or it.get("baseCurrency") or it.get("baseToken")
        quote = it.get("quoteCoin") or it.get("quote") or it.get("quoteCurrency") or it.get("quoteToken")
        if not (sym and base and quote):
            continue
        instr_map[sym] = {"symbol": sym, "base": base, "quote": quote}

    # ticker map: symbol -> ticker dict
    ticker_map = {}
    for t in tickers:
        sym = t.get("symbol") or t.get("s") or t.get("instId")
        if not sym:
            continue
        ticker_map[sym] = t

    price_map = {}
    symbol_map = {}
    out_edges = defaultdict(set)

    for sym, it in instr_map.items():
        base = it["base"]
        quote = it["quote"]
        t = ticker_map.get(sym)
        if not t:
            # тикер отсутствует — пропускаем
            continue

        # Попытки извлечь лучшие bid/ask из нескольких возможных полей (Bybit example: bid1Price/ask1Price)
        bid = None
        ask = None
        for k in ("bid1Price", "bidPrice", "bid", "bestBidPrice", "bid1"):
            if k in t:
                try:
                    bid = float(t[k])
                    break
                except Exception:
                    bid = None
        for k in ("ask1Price", "askPrice", "ask", "bestAskPrice", "ask1"):
            if k in t:
                try:
                    ask = float(t[k])
                    break
                except Exception:
                    ask = None

        # иногда цены лежат во вложенном объекте 'tick' или 'data'
        if (bid is None or ask is None):
            for container in ("data", "tick", "info", "instrument"):
                sub = t.get(container)
                if isinstance(sub, dict):
                    if bid is None:
                        for k in ("bid1Price","bidPrice","bid","bestBidPrice"):
                            if k in sub:
                                try:
                                    bid = float(sub[k]); break
                                except Exception:
                                    bid = None
                    if ask is None:
                        for k in ("ask1Price","askPrice","ask","bestAskPrice"):
                            if k in sub:
                                try:
                                    ask = float(sub[k]); break
                                except Exception:
                                    ask = None

        # если не нашли или нулевая/отрицательная цена — пропускаем
        if bid is None or ask is None:
            continue
        if bid <= 0 or ask <= 0:
            continue

        price_map[sym] = {"bid": bid, "ask": ask, "base": base, "quote": quote}
        symbol_map[(base, quote)] = sym

        # направление дуг: продаём base -> получаем quote (base -> quote)
        out_edges[base].add(quote)
        # покупаем base за quote => quote -> base
        out_edges[quote].add(base)

    return out_edges, symbol_map, price_map

# ---------- Конвертация суммы между активами ----------
def convert_amount(amount, from_asset, to_asset, symbol_map, price_map, fee_rate):
    """
    Преобразует amount из from_asset в to_asset, используя прямую или обратную пару.
    Возвращает (new_amount, used_symbol, direction) или (None, None, None)
    direction: 'sell' если использована пара FROM/TO (продажа from по bid),
               'buy'  если использована пара TO/FROM (покупка to за from по ask).
    """
    # прямая пара: FROM/TO (продаём from -> получаем quote по bid)
    sym_direct = symbol_map.get((from_asset, to_asset))
    if sym_direct:
        p = price_map.get(sym_direct)
        if not p:
            return None, None, None
        bid = p["bid"]
        new_amount = amount * bid * (1 - fee_rate)
        return new_amount, sym_direct, "sell"

    # обратная пара: TO/FROM (покупаем to за from по ask)
    sym_rev = symbol_map.get((to_asset, from_asset))
    if sym_rev:
        p = price_map.get(sym_rev)
        if not p:
            return None, None, None
        ask = p["ask"]
        new_amount = (amount / ask) * (1 - fee_rate)
        return new_amount, sym_rev, "buy"

    return None, None, None

# ---------- Нормализация цикла ----------
def normalize_cycle(nodes):
    """
    Нормализует цикл (список вершин без дублирующего последнего элемента) для удаления ротаций.
    Возвращает tuple минимальной ротации (лексикографически).
    Направление сохраняется.
    """
    if not nodes:
        return tuple()
    n = len(nodes)
    rotations = []
    for i in range(n):
        rot = tuple(nodes[i:] + nodes[:i])
        rotations.append(rot)
    return min(rotations)

# ---------- Поиск циклов (DFS), длина 3..max_cycle_len ----------
def find_arbitrage_cycles_bybit(fee_rate=0.001, min_profit=0.0005, start_amount=1.0,
                                max_cycles=200000, max_cycle_len=4, category="spot"):
    """
    Ищет циклы длины 3..max_cycle_len на Bybit.
    Параметры аналогичны описанным ранее.
    Возвращает список возможностей:
      {'path':[A,B,C,A,...], 'start_asset':A, 'start_amount':..., 'end_amount':..., 'profit_perc':..., 'trades':[...]}
    """
    if max_cycle_len < 3:
        raise ValueError("max_cycle_len must be >= 3")

    out_edges, symbol_map, price_map = build_graph_and_prices(category=category)

    opportunities = []
    checked = 0
    seen_cycles = set()
    assets = list(out_edges.keys())
    stop_flag = False

    def dfs(start, path):
        nonlocal checked, stop_flag
        if stop_flag:
            return
        current = path[-1]

        # если нашли возврат к start и длина >=3 — цикл найден
        if len(path) >= 3 and start in out_edges[current]:
            cycle_nodes = path[:]  # e.g. [A,B,C]
            norm = normalize_cycle(cycle_nodes)
            if norm not in seen_cycles:
                seen_cycles.add(norm)
                # симуляция сделок
                amt = start_amount
                trades = []
                valid = True
                for i in range(len(cycle_nodes)):
                    frm = cycle_nodes[i]
                    to = cycle_nodes[(i + 1) % len(cycle_nodes)]
                    new_amt, sym, direction = convert_amount(amt, frm, to, symbol_map, price_map, fee_rate)
                    if new_amt is None:
                        valid = False
                        break
                    trades.append((frm, to, sym, direction, new_amt))
                    amt = new_amt
                if valid:
                    profit = amt - start_amount
                    profit_perc = profit / start_amount
                    if profit_perc > min_profit:
                        opportunities.append({
                            "path": cycle_nodes + [cycle_nodes[0]],
                            "start_asset": cycle_nodes[0],
                            "start_amount": start_amount,
                            "end_amount": amt,
                            "profit": profit,
                            "profit_perc": profit_perc,
                            "trades": trades
                        })
            checked += 1
            if checked > max_cycles:
                stop_flag = True
                return

        # ограничение по длине пути
        if len(path) >= max_cycle_len:
            return

        for nb in out_edges[current]:
            if stop_flag:
                return
            # не повторяем вершины в пути
            if nb in path:
                continue
            # избегаем преждевременного возврата (обрабатываем выше)
            if nb == start:
                continue
            checked += 1
            if checked > max_cycles:
                stop_flag = True
                return
            path.append(nb)
            dfs(start, path)
            path.pop()

    # запуск DFS от каждой вершины
    for a in assets:
        if stop_flag:
            break
        # быстрый фильтр: чтобы образовать цикл длиной >=3 вершина должна иметь хотя бы 1 соседей
        if len(out_edges[a]) < 1:
            continue
        dfs(a, [a])

    # сортируем по убыванию прибыли %
    opportunities.sort(key=lambda x: x["profit_perc"], reverse=True)
    return opportunities

def save_to_file(array, file_name):
    # сохранить в файл
    out_path = f'files/{file_name}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(array, f, ensure_ascii=False, indent=2)

    print(f'fast reverberation result saved to {out_path}')


def arbitrage_bybit():
    fee = 0.001  # 0.1% комиссия (пример)
    min_profit = 0.00001  # минимальная прибыль 0.1%
    start = 1.0
    max_len = 4  # ищем циклы длиной 3 и 4
    category = "spot"  # можно 'spot', 'inverse', 'linear' и т.д. (в зависимости от нужного рынка)

    print("Запрашиваю данные с Bybit... (может занять несколько секунд)")
    ops = find_arbitrage_cycles_bybit(fee_rate=fee, min_profit=min_profit,
                                      start_amount=start, max_cycles=300000,
                                      max_cycle_len=max_len, category=category)

    if not ops:
        print("Арбитражных возможностей не найдено (по заданному порогу).")
    else:
        for o in ops[:30]:
            print("Путь:", " -> ".join(o["path"]),
                  f"Начало {o['start_amount']}{o['start_asset']} -> Конец {o['end_amount']:.8f}{o['start_asset']}",
                  f"Прибыль {o['profit_perc'] * 100:.4f}%")
            for t in o["trades"]:
                frm, to, sym, dirc, amt = t
                print(f"  {frm} -> {to} via {sym} ({dirc}) => {amt:.8f} {to}")
            print("----")
