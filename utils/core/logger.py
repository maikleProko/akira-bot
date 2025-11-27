from datetime import datetime


class Logger:
    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def get_timestamp(self):
        return datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]

    def print_message(self, message):
        timestamp = self.get_timestamp()
        print(f"{timestamp}: {message}")

    def error(self, message):
        self.print_message(message)

    def write_to_file(self, timestamp, message):
        with open('files/decisions/final_decisions.txt', 'a', encoding='utf-8') as f:
            f.write(f"{timestamp}: {message}\n")

    def log_message(self, message):
        timestamp = self.get_timestamp()
        print(f"{timestamp}: {message}")
        self.write_to_file(timestamp, message)