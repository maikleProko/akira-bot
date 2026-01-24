from functools import wraps
from apscheduler.schedulers.blocking import BlockingScheduler
import threading
import time
import logging
from datetime import datetime


def periodic(trigger='cron', **trigger_args):
    """
    Декоратор: превращает функцию в периодическую задачу APScheduler.
    Декорируемая функция должна принимать те же аргументы, что и вызов starter-а.
    Пример использования: @periodic(second=0) -> запускать каждую минуту в начале минуты.
    """
    def decorator(func):
        @wraps(func)
        def starter(*args, **kwargs):
            scheduler = BlockingScheduler()

            def job_wrapper():
                try:
                    print(f' ')
                    #print(f'[run] TICK RUN AT {datetime.now()}')
                    func(*args, **kwargs)
                except Exception as e:
                    # логируем ошибку, но не останавливаем планировщик
                    print(f'[{func.__name__}] Error in scheduled job: {e}', flush=True)

            job_id = f'periodic_{func.__name__}'
            scheduler.add_job(job_wrapper, trigger, id=job_id, replace_existing=True,
                              coalesce=True, max_instances=1, **trigger_args)
            print(f'[{func.__name__}] Scheduler started: trigger={trigger}, args={trigger_args}')
            try:
                scheduler.start()
            except (KeyboardInterrupt, SystemExit):
                print(f'[{func.__name__}] Scheduler stopped by user.')
                scheduler.shutdown()
        return starter
    return decorator


def fast_periodic():
    """
    Декоратор: превращает функцию в непрерывно выполняющуюся задачу без пауз.
    Декорируемая функция должна принимать те же аргументы, что и вызов starter-а.
    """
    def decorator(func):
        @wraps(func)
        def starter(*args, **kwargs):
            print(f'[{func.__name__}] Continuous runner started')
            try:
                while True:
                    try:
                        #print(f'[run] TICK RUN AT {datetime.now()}')
                        func(*args, **kwargs)
                    except Exception as e:
                        # логируем ошибку, но не останавливаем цикл
                        print(f'[{func.__name__}] Error in job: {e}', flush=True)
            except (KeyboardInterrupt, SystemExit):
                print(f'[{func.__name__}] Continuous runner stopped by user.')
        return starter
    return decorator


def every_second(interval=1.0, daemon=True):
    """
    Декоратор: при вызове декорированной функции начинается фоновый цикл,
    который вызывает функцию каждые `interval` секунд.
    Если функция выполняется дольше интервала, следующий запуск будет
    ждать завершения текущего и сразу стартует.
    """
    def decorator(func):
        def starter(*args, **kwargs):
            if getattr(starter, "_running", False):
                raise RuntimeError("Already running")
            starter._stop_event = threading.Event()

            def runner():
                while not starter._stop_event.is_set():
                    start_time = time.monotonic()
                    try:
                        func(*args, **kwargs)
                    except Exception:
                        logging.exception("Exception in repeated function")
                    elapsed = time.monotonic() - start_time
                    wait = interval - elapsed
                    if wait > 0:
                        # ждать можно прерываемо через event.wait
                        starter._stop_event.wait(wait)
                starter._running = False

            starter._thread = threading.Thread(target=runner, daemon=daemon)
            starter._running = True
            starter._thread.start()

        def stop(wait=True):
            """Остановить цикл. Если wait=True — дождаться завершения потока."""
            if not getattr(starter, "_running", False):
                return
            starter._stop_event.set()
            if wait:
                starter._thread.join()

        # Экспортируем методы/флаги на сам starter
        starter.stop = stop
        starter.is_running = lambda: getattr(starter, "_running", False)
        return starter
    return decorator
