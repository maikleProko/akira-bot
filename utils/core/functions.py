from datetime import datetime
import json


def load_json(file_path: str) -> dict:
    """
    Читает содержимое JSON файла и возвращает его как словарь Python.

    Args:
        file_path (str): Путь к JSON файлу

    Returns:
        dict: Словарь с данными из JSON файла

    Raises:
        FileNotFoundError: Если файл не существует
        json.JSONDecodeError: Если файл содержит некорректный JSON
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"Файл {file_path} не найден")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Ошибка при разборке JSON в файле {file_path}: {str(e)}")

def log(text):
    print(f"{datetime.now()} {str(text)}")


class MarketProcess:
    def prepare(self, start_time=None, end_time=None):
        pass

    def run_historical(self, start_time, current_time):
        pass

    def run_realtime(self):
        pass

    def run(self, start_time=None, current_time=None):
        if start_time is None and current_time is None:
            self.run_realtime()
        else:
            self.run_historical(start_time, current_time)

    def finalize(self):
        pass
