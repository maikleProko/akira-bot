import time
from datetime import datetime
from functools import wraps

from apscheduler.schedulers.blocking import BlockingScheduler


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
