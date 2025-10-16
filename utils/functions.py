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


class MarketProcess:
    def prepare(self):
        pass

    def run(self, start_time=None, current_time=None, end_time=None):
        pass
