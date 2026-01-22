from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from scenarios.constants import *
from utils.core.decorators import periodic, fast_periodic


class MarketProcessor(ABC):
    """Абстрактный базовый класс для процессоров рынка"""

    @abstractmethod
    def prepare(self):
        """Подготовка процессора"""
        pass

    @abstractmethod
    def run(self):
        """Запуск процессора"""
        pass


class RealtimeProcessor(MarketProcessor):
    """Процессор для работы в реальном времени"""

    def prepare(self):
        """Подготовка для реального времени"""
        print('Preparing realtime...')
        self._prepare_market_processes()

    @fast_periodic()
    def run(self):
        """Запуск в реальном времени"""
        self._execute_market_processes()

    def _prepare_market_processes(self):
        """Подготовка всех рыночных процессов"""
        for market_process in MARKET_PROCESSES:
            market_process.prepare()
            market_process.prepocess_realtime()

    def _execute_market_processes(self):
        """Выполнение рыночных процессов в реальном времени"""
        for market_process in MARKET_PROCESSES:
            market_process.run()


from datetime import datetime, timedelta

class HistoricalProcessor(MarketProcessor):
    """Процессор для исторической обработки"""

    def __init__(self, start_time_string, end_time_string, minutes_interval=1, is_printed_ticks=True):
        self.minutes_interval = minutes_interval
        self.start_time = datetime.strptime(start_time_string, "%Y/%m/%d %H:%M")
        self.end_time = datetime.strptime(end_time_string, "%Y/%m/%d %H:%M")
        self.is_printed_ticks = is_printed_ticks

    def prepare(self):
        """Подготовка для исторической обработки"""
        print('[Processor] Preparing history...')
        self._prepare_market_processes()

    def run(self):
        """Запуск исторической обработки"""
        current_time = self.start_time
        current_date = current_time.date()  # Дата текущей итерации
        previous_date = None                # Дата предыдущей итерации

        # Выводим начало первого дня сразу
        print(f'[Processor] === День: {current_date.strftime("%Y-%m-%d")} ===')

        while current_time <= self.end_time:
            current_date = current_time.date()

            # Проверяем, сменился ли день
            if previous_date is not None and current_date != previous_date:
                print(f'[Processor] === День: {current_date.strftime("%Y-%m-%d")} ===')

            if self.is_printed_ticks:
                print('[Processor] Tick: ' + current_time.strftime('%Y/%m/%d_%H:%M'))

            try:
                self._execute_market_processes(current_time)
            except Exception as e:
                if self.is_printed_ticks:
                    print('[Processor] running error: ' + str(e))

            previous_date = current_date
            current_time += timedelta(minutes=self.minutes_interval)

        for process in MARKET_PROCESSES:
            process.finalize()

    def _prepare_market_processes(self):
        """Подготовка рыночных процессов для исторического режима"""
        for market_process in MARKET_PROCESSES:
            market_process.prepare(self.start_time, self.end_time)

    def _execute_market_processes(self, current_time):
        """Выполнение рыночных процессов для конкретного времени"""
        for market_process in MARKET_PROCESSES:
            market_process.run(self.start_time, current_time)


class MainProcessor:
    """Главный процессор, управляющий выбором режима работы"""

    def __init__(self, realtime=False, start_time_string=None, end_time_string=None, minutes_interval=1, is_printed_ticks=True):
        """
        :param realtime: True для реального времени, False для исторической обработки
        :param start_time_string: формат "YYYY/MM/DD HH:MM" для исторической обработки
        :param end_time_string: формат "YYYY/MM/DD HH:MM" для исторической обработки
        :param minutes_interval: интервал в минутах для исторической обработки
        """
        self.realtime = realtime
        self.minutes_interval = minutes_interval
        self.is_printed_ticks = is_printed_ticks

        # Валидация параметров
        if not realtime and (start_time_string is None or end_time_string is None):
            raise ValueError("Для исторической обработки обязательны start_time_string и end_time_string")

        # Инициализация сервисов
        self._initialize_services(start_time_string, end_time_string)
        self.prepare()
        self.run()

    def _initialize_services(self, start_time_string, end_time_string):
        """Инициализация нужного процессора в зависимости от режима"""

        if self.realtime:
            self.processor = RealtimeProcessor()
            print("MainProcessor: Инициализирован RealtimeProcessor")
        else:
            self.processor = HistoricalProcessor(
                start_time_string,
                end_time_string,
                self.minutes_interval,
                self.is_printed_ticks
            )
            print("MainProcessor: Инициализирован HistoricalProcessor")

    def prepare(self):
        """Подготовка процессора"""
        self.processor.prepare()

    def run(self):
        """Запуск процессора"""
        self.processor.run()

    def get_processor_type(self):
        """Получить тип используемого процессора"""
        return type(self.processor).__name__


MainProcessor(
    realtime=realtime,
    start_time_string=start_time_string,
    end_time_string=end_time_string,
    is_printed_ticks=is_printed_ticks
)