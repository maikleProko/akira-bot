# rolling_arbitrage_parser.py
import asyncio
import time
from typing import Dict, List

from scenarios.parsers.arbitrage_parser.rolling_arbitrage_parser.core.abstract_arbitrage_parser import \
    AbstractArbitrageParser
from collections import defaultdict
class RollingArbitrageParser(AbstractArbitrageParser):
    async def async_init_chain(self, pos, portion, path, balances):
        current_amount = portion
        current_coin = path[0]
        for step in range(pos):
            next_coin = path[step + 1]
            current_amount = await self.async_perform_trade(current_coin, next_coin, current_amount, balances)
            current_coin = next_coin
        return current_amount  # Возвращаем, хотя не используется
    async def async_unwind_chain(self, start_idx, path, balances):
        N = len(path)
        current_idx = start_idx
        current_coin = path[current_idx]
        for _ in range(N - start_idx):
            next_coin = path[(current_idx + 1) % N]
            amount = balances.get(current_coin, 0.0) if not self.production else self.get_balance(current_coin)
            if amount <= 0:
                break
            await self.async_perform_trade(current_coin, next_coin, amount, balances)
            # В prod добавляем небольшую задержку для обновления баланса
            if self.production:
                await asyncio.sleep(1)  # 1 сек задержки для синхронизации
            current_coin = next_coin
            current_idx = (current_idx + 1) % N
    def _print_balances(self, balances: Dict[str, float], path: List[str]):
        balance_str = "Balances: " + ", ".join(f"{coin}: {balances.get(coin, 0.0):.4f}" for coin in path)
        self.logger.print_message(balance_str)
    async def start_trade(self, path_data: dict):
        """
        Реализация перекатного арбитража.
        """
        path = path_data['path'][:-1] # Цикл без замыкания
        N = len(path)
        start_coin = path[0]
        # Инициализация балансов
        if self.production:
            balances = {coin: self.get_balance(coin) for coin in path}
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
        if self.is_testing_only_once:
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
                    amount = balances[from_coin]
                    trades.append(self.async_perform_trade(from_coin, to_coin, amount, balances))
                # Запуск асинхронно и ожидание всех
                results = await asyncio.gather(*trades, return_exceptions=True)
                # Обработка результатов
                for res in results:
                    if isinstance(res, Exception):
                        self.logger.print_message(f"Trade error: {res}")
                if not self.production:
                    for j in range(N):
                        if not isinstance(results[j], Exception):
                            # balances[path[(j + 1) % N]] = results[j] # Уже обновлено в async_perform_trade
                            pass # Нет нужды сбрасывать, balances уже обновлены
                self.logger.print_message("Roll completed")
                if self.is_testing_only_once:
                    rolled_once = True
                    self.logger.print_message("TESTING ONLY ONCE (2/2)")
                    self._print_balances(balances, path)
                    break
            else:
                if time.time() - last_good_time > 180: # 3 минуты
                    self.logger.print_message("Path no longer profitable for 3 min, stopping")
                    break
                self.logger.print_message("Path not profitable, waiting...")
            await asyncio.sleep(1)
            if self.is_testing_only_once and rolled_once:
                break
        # Фаза unwind: конвертация обратно в start_coin
        self.logger.print_message(f"Unwinding positions back to {start_coin}")
        unwind_tasks = [self.async_unwind_chain(pos, path, balances) for pos in range(1, N)]
        await asyncio.gather(*unwind_tasks)
        self.logger.print_message("Unwind complete, all capital back to {start_coin}")