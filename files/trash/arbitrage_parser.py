import json
from utils.core.functions import MarketProcess
from datetime import datetime

class ArbitrageParser(MarketProcess):
    def run_realtime(self):
        fee = 0.001      # 0.1% комиссия
        min_profit = 0.000001  # 0.1% порог
        start = 1.0
        max_len = 4  # ищем циклы длины 3 и 4

        ops = self.find_arbitrage_cycles(fee_rate=fee, min_profit=min_profit, start_amount=start,
                                    max_cycles=200000, max_cycle_len=max_len)

        if not ops:
            print("Арбитражных возможностей не найдено (по заданному порогу).")
        else:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

            for o in ops[:50]:
                print("Путь:", " -> ".join(o['path']), f"Начало {o['start_amount']}{o['start_asset']} -> Конец {o['end_amount']:.8f}{o['start_asset']}", f"Прибыль {o['profit_perc']*100:.4f}%")
                print("Calculation steps:")
                amt = o['start_amount']
                for t in o['trades']:
                    frm, to, sym, dirc, new_amt, price = t
                    if dirc == 'sell':
                        print(f"  Selling {amt:.8f} {frm} for {to} using pair {sym} at bid price {price:.8f} ({to} per {frm})")
                        print(f"  Amount received: {amt:.8f} * {price:.8f} * (1 - {fee}) = {new_amt:.8f} {to}")
                    else:
                        print(f"  Buying {new_amt:.8f} {to} with {amt:.8f} {frm} using pair {sym} at ask price {price:.8f} ({frm} per {to})")
                        print(f"  Amount bought: ({amt:.8f} / {price:.8f}) * (1 - {fee}) = {new_amt:.8f} {to}")
                    amt = new_amt
                print("----")


    def build_graph_and_prices(self):
        return [], [], []

    def find_arbitrage_cycles(self, fee_rate=0.001, min_profit=0.0001, start_amount=1.0,
                              max_cycles=200000, max_cycle_len=4):
        """
        Ищет циклы длины 3..max_cycle_len.
        Параметры:
          - fee_rate: комиссия за сделку (например 0.001 = 0.1%)
          - min_profit: минимальная относительная прибыль для отчёта (например 0.001 = 0.1%)
          - start_amount: стартовый объём для симуляции
          - max_cycles: лимит проверяемых путей (защита от долгой работы)
          - max_cycle_len: максимальное число разных активов в цикле (>=3)
        Возвращает список возможностей вида:
          {'path': [A,B,C,A], 'start_asset':A, 'start_amount':..., 'end_amount':..., 'profit_perc':..., 'trades': [...]}
        """
        if max_cycle_len < 3:
            raise ValueError("max_cycle_len must be >= 3")

        out_edges, symbol_map, price_map = self.build_graph_and_prices()
        opportunities = []
        checked = 0
        seen_cycles = set()  # для удаления ротационных дубликатов

        assets = list(out_edges.keys())

        stop_flag = False

        def dfs(start, current_path):
            nonlocal checked, stop_flag

            if stop_flag:
                return

            current = current_path[-1]

            # Если длина >=3 и есть обратная дуга к старту -> нашли цикл
            if len(current_path) >= 3 and start in out_edges[current]:
                # Нормализуем цикл без повторного конца
                cycle_nodes = current_path[:]  # e.g. [A, B, C]
                norm = self.normalize_cycle(cycle_nodes)
                if norm not in seen_cycles:
                    seen_cycles.add(norm)
                    # Проверяем и симулируем сделки
                    trades = []
                    amt = start_amount
                    valid = True
                    for i in range(len(cycle_nodes)):
                        frm = cycle_nodes[i]
                        to = cycle_nodes[(i + 1) % len(cycle_nodes)]
                        new_amt, sym, direction, price = self.convert_amount(amt, frm, to, symbol_map, price_map, fee_rate)
                        if new_amt is None:
                            valid = False
                            break
                        trades.append((frm, to, sym, direction, new_amt, price))
                        amt = new_amt
                    if valid:
                        profit = amt - start_amount
                        profit_perc = profit / start_amount
                        if profit_perc > min_profit:
                            # путь с повтором начального узла для удобства
                            path_with_return = list(cycle_nodes) + [cycle_nodes[0]]
                            opportunities.append({
                                'path': path_with_return,
                                'start_asset': cycle_nodes[0],
                                'start_amount': start_amount,
                                'end_amount': amt,
                                'profit': profit,
                                'profit_perc': profit_perc,
                                'trades': trades
                            })

                checked += 1
                if checked > max_cycles:
                    stop_flag = True
                    return

            # Продолжаем DFS, если не достигли предельной длины
            if len(current_path) >= max_cycle_len:
                return

            for nb in out_edges[current]:
                if stop_flag:
                    return
                # Не возвращаемся в старт раньше времени и не повторяем вершины
                if nb == start:
                    # Мы уже обрабатываем закрытие цикла выше при len>=3, поэтому здесь пропускаем
                    continue
                if nb in current_path:
                    continue
                # Ограничитель проверяемых путей
                checked += 1
                if checked > max_cycles:
                    stop_flag = True
                    return
                current_path.append(nb)
                dfs(start, current_path)
                current_path.pop()

        # Запускаем DFS от каждой вершины
        for a in assets:
            if stop_flag:
                break
            # Примитивный оптимизационный фильтр: если вершина имеет степень <2, маловероятно образовать цикл >=3
            if len(out_edges[a]) < 1:
                continue
            dfs(a, [a])

        # Сортируем по убыванию прибыли %
        opportunities.sort(key=lambda x: x['profit_perc'], reverse=True)
        return opportunities

    def convert_amount(self, amount, from_asset, to_asset, symbol_map, price_map, fee_rate):
        """
        Преобразует amount из from_asset в to_asset используя доступные пары.
        Возвращает (new_amount, used_symbol, direction, price) или (None, None, None, None) если конвертация невозможна.
        direction = 'sell' если использована пара FROM/TO (продажа from для получения to at bid)
                  = 'buy'  если использована пара TO/FROM (покупка to за from at ask)
        """
        # Прямая пара FROM/TO: продаём from -> получаем quote
        sym_direct = symbol_map.get((from_asset, to_asset))
        if sym_direct:
            p = price_map.get(sym_direct)
            if not p:
                return None, None, None, None
            bid = p['bid']
            new_amount = amount * bid * (1 - fee_rate)
            return new_amount, sym_direct, 'sell', bid

        # Обратная пара TO/FROM: покупаем to за from по ask
        sym_rev = symbol_map.get((to_asset, from_asset))
        if sym_rev:
            p = price_map.get(sym_rev)
            if not p:
                return None, None, None, None
            ask = p['ask']
            # можем купить amount / ask единиц to
            new_amount = (amount / ask) * (1 - fee_rate)
            return new_amount, sym_rev, 'buy', ask

        return None, None, None, None

    def normalize_cycle(self, nodes):
        """
        Нормализует цикл nodes (без повторного стартового элемента) для удаления ротационных дубликатов.
        Возвращает кортеж с минимальной ротацией (лексикографически).
        Пример: ['B','C','A'] -> кортеж ('A','B','C')
        NOTE: учитывается порядок (направление цикла). Обратная последовательность считается другим циклом.
        """
        if not nodes:
            return tuple()
        n = len(nodes)
        rotations = []
        for i in range(n):
            rot = tuple(nodes[i:] + nodes[:i])
            rotations.append(rot)
        return min(rotations)

    def save_to_file(self, array, file_name):
        # сохранить в файл
        out_path = f'files/{file_name}.json'
        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(array, f, ensure_ascii=False, indent=2)

        print(f'fast reverberation result saved to {out_path}')