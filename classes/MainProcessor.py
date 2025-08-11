import os
import time
import json
from datetime import datetime
from collections import Counter
from copy import deepcopy

from classes.selenium.instances.ChartGetter import ChartGetter
from classes.ocr.instances.ChartParser import ChartParser


class MainProcessor:
    def __init__(self):
        # при создании запускаем run (в run мы стартуем периодическую запись)
        self.run()

    def get_pane_data(self, iterations=7):
        pane_datas = []
        for i in range(iterations):
            try:
                chart_getter = ChartGetter()
                chart_parser = ChartParser()
                pd = getattr(chart_parser, 'pane_data', None)
                pane_datas.append(deepcopy(pd))
            except Exception:
                pane_datas.append(None)

        valid = [pd for pd in pane_datas if pd is not None]

        if not valid:
            return None

        serials = [json.dumps(pd, sort_keys=True, separators=(',', ':'), ensure_ascii=False) for pd in valid]
        counts = Counter(serials)
        most_common_serial, _ = counts.most_common(1)[0]

        for pd in valid:
            if json.dumps(pd, sort_keys=True, separators=(',', ':'), ensure_ascii=False) == most_common_serial:
                return deepcopy(pd)

        return pane_datas[0]

    def dump_pane_data_periodically(self, interval_seconds=60, out_dir='./test_pane_datas/'):
        """
        Бесконечно каждые interval_seconds вычисляет get_pane_data и
        записывает результат в файл out_dir/YYYY-MM-DD_HH-MM.txt
        Скрипт работает пока пользователь не прервет процесс (Ctrl+C).
        """
        os.makedirs(out_dir, exist_ok=True)

        try:
            while True:
                # Получаем данные и формируем имя файла по текущему времени (до минуты)
                try:
                    pane_data = self.get_pane_data()
                except Exception as e:
                    print(f"[{datetime.now()}] Ошибка при получении pane_data: {e}")
                    pane_data = None

                now = datetime.now()
                fname = now.strftime('%Y-%m-%d_%H-%M') + '.txt'
                fpath = os.path.join(out_dir, fname)

                try:
                    with open(fpath, 'w', encoding='utf-8') as f:
                        json.dump(pane_data, f, ensure_ascii=False, indent=2)
                    print(f"[{datetime.now()}] Записан файл: {fpath}")
                except Exception as e:
                    print(f"[{datetime.now()}] Ошибка при записи файла {fpath}: {e}")

                # Вычисляем время до следующей полной минуты и спим
                now = datetime.now()
                # секунды до следующей минуты (с учётом микросекунд)
                secs = 60 - now.second - now.microsecond / 1_000_000
                # Если заданый interval_seconds отличается от 60, используем его
                if interval_seconds != 60:
                    time.sleep(interval_seconds)
                else:
                    # спим до следующей минуты (чтобы имена файлов соответствовали минутам)
                    time.sleep(secs)
        except KeyboardInterrupt:
            print("Periodic dump stopped by user (KeyboardInterrupt).")
        except Exception as e:
            print(f"Stopped due to unexpected error: {e}")

    def run(self):
        # Запускаем периодическую запись каждые 60 секунд (по минутам)
        self.dump_pane_data_periodically(interval_seconds=60, out_dir='./test_pane_datas/')