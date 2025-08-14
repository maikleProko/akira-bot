import time
from datetime import datetime
from functools import wraps

def doing_periodical_per_1_minute(func):

    @wraps(func)
    def wrapper():
        interval_seconds = 60

        try:
            while True:
                func()
                now = datetime.now()
                secs = 60 - now.second - now.microsecond / 1_000_000
                if interval_seconds != 60:
                    time.sleep(interval_seconds)
                else:
                    time.sleep(secs)
        except KeyboardInterrupt:
            print("Periodic dump stopped by user (KeyboardInterrupt).")
        except Exception as e:
            print(f"Stopped due to unexpected error: {e}")

    return wrapper()
