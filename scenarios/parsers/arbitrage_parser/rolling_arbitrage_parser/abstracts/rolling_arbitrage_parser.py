# rolling_arbitrage_parser.py
import asyncio
import time
from typing import Dict, List

from scenarios.parsers.arbitrage_parser.rolling_arbitrage_parser.abstracts.abstract_arbitrage_parser import \
    AbstractArbitrageParser
from collections import defaultdict
class RollingArbitrageParser(AbstractArbitrageParser):
    async def async_init_chain(self, pos, portion, path, balances):
        start_coin = path[0]
        target_coin = path[pos]
        current_amount = portion
        current_amount = await self.async_perform_trade(start_coin, target_coin, current_amount, balances)
        return current_amount  # Возвращаем, хотя не используется
    async def async_unwind_chain(self, start_idx, path, balances):
        start_coin = path[0]
        current_coin = path[start_idx]
        amount = balances.get(current_coin, 0.0) if not self.production else await self.async_get_balance(current_coin)
        if amount <= 0:
            return
        await self.async_perform_trade(current_coin, start_coin, amount, balances)
        # В prod добавляем небольшую задержку для обновления баланса
        if self.production:
            await asyncio.sleep(0.2)  # Уменьшили задержку
    def _print_balances(self, balances: Dict[str, float], path: List[str]):
        balance_str = "Balances: " + ", ".join(f"{coin}: {balances.get(coin, 0.0):.4f}" for coin in path)
        self.logger.print_message(balance_str)
    async def start_trade(self, path_data: dict):
        """
        Реализация перекатного арбитража.
        """

        if self.is_testing_only_once_in_cycle and self.is_tested_only_once_in_cycle:
            return
        path = path_data['path'][:-1] # Цикл без замыкания
        N = len(path)
        start_coin = path[0]
        # Инициализация балансов
        if self.production:
            balance_tasks = [self.async_get_balance(coin) for coin in path]
            balance_results = await asyncio.gather(*balance_tasks)
            balances = {coin: balance_results[i] for i, coin in enumerate(path)}
            total = balances[start_coin]
        else:
            total = self.deposit
            balances = defaultdict(float)
            balances[start_coin] = total
        self.logger.print_message(f"Starting rolling arbitrage with total {total} {start_coin}, N={N}")
        portion = total / N

        # Инициализация: распределение капитала по позициям асинхронно
        init_tasks = []
        for pos in range(1, N):
            init_tasks.append(self.async_init_chain(pos, portion, path, balances))
        if init_tasks:
            await asyncio.gather(*init_tasks)
        self.logger.print_message("Initialization complete, capital distributed")
        # Цикл переката
        last_good_time = time.time()
        rolled_once = False
        if self.is_testing_only_once_in_cycle:
            self.logger.print_message("TESTING ONLY ONCE (1/2)")
            self._print_balances(balances, path)
        while True:
            self._fetch_specific_prices(path) # Обновить только нужные цены
            # Проверка профитности пути
            multiplier = 1.0
            valid = True
            for i in range(N):
                frm = path[i]
                to = path[(i + 1) % N]
                rate = self._get_rate(frm, to)
                if rate is None:
                    valid = False
                    break
                multiplier *= rate
            if valid and multiplier > 1.0:
                last_good_time = time.time()
                self.logger.print_message(f"Path still profitable: {multiplier - 1:.4%}")
                # Перекат: N асинхронных сделок
                trades = []
                for j in range(N):
                    from_coin = path[j]
                    to_coin = path[(j + 1) % N]
                    amount = balances[from_coin] if not self.production else await self.async_get_balance(from_coin)
                    if amount <= 0:
                        continue
                    trades.append(self.async_perform_trade(from_coin, to_coin, amount, balances))
                # Запуск асинхронно и ожидание всех
                if trades:
                    results = await asyncio.gather(*trades, return_exceptions=True)
                    # Обработка результатов
                    for res in results:
                        if isinstance(res, Exception):
                            self.logger.print_message(f"Trade error: {res}")
                    if not self.production:
                        pass # balances updated in sim
                self.logger.print_message("Roll completed")
                if self.is_testing_only_once_in_cycle:
                    count = N
                    self.is_testing_only_once_in_cycle_counter += 1

                    if self.is_testing_only_once_in_cycle_counter > count:
                        self.is_tested_only_once_in_cycle = True
                        break
                    self.logger.print_message(f"TESTING ONLY ONCE ({self.is_testing_only_once_in_cycle_counter + 1}/{count + 1})")
                    self._print_balances(balances, path)
                    if not self.is_testing_only_once_in_cycle:
                        break
            else:
                if time.time() - last_good_time > 6: # 6 секунд
                    self.logger.print_message("Path no longer profitable for 6 sec, stopping")
                    break
                self.logger.print_message("Path not profitable, waiting...")
            await asyncio.sleep(1)
            if self.is_testing_only_once_in_cycle and rolled_once:
                break
        # Фаза unwind: конвертация обратно в start_coin
        self.logger.print_message(f"Unwinding positions back to {start_coin}")
        unwind_tasks = [self.async_unwind_chain(pos, path, balances) for pos in range(1, N)]
        await asyncio.gather(*unwind_tasks)
        self.logger.print_message("Unwind complete, all capital back to {start_coin}")