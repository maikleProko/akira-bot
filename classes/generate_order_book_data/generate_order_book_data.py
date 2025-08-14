import os
import json
from datetime import datetime
from collections import Counter
from copy import deepcopy

from classes.generate_order_book_data.selenium.instances.OrderBookGetter import OrderBookGetter
from classes.generate_order_book_data.ocr.instances.OrderBookParser import OrderBookParser


def get_order_book_data(iterations=7):
    order_book_datas = []
    for i in range(iterations):
        try:
            order_book_getter = OrderBookGetter()
            order_book_parser = OrderBookParser()
            pd = getattr(order_book_parser, 'order_book_data', None)
            order_book_datas.append(deepcopy(pd))
        except Exception as e:
            order_book_datas.append(None)

    valid = [pd for pd in order_book_datas if pd is not None]

    if not valid:
        return None

    serials = [json.dumps(pd, sort_keys=True, separators=(',', ':'), ensure_ascii=False) for pd in valid]
    counts = Counter(serials)
    most_common_serial, _ = counts.most_common(1)[0]

    for pd in valid:
        if json.dumps(pd, sort_keys=True, separators=(',', ':'), ensure_ascii=False) == most_common_serial:
            return deepcopy(pd)

    return order_book_datas[0]


def generate_order_book_data(iterations=7, out_dir='./test_order_book_datas/'):
    os.makedirs(out_dir, exist_ok=True)

    try:
        order_book_data = get_order_book_data()
    except Exception as e:
        print(f"[{datetime.now()}] Ошибка при получении order_book_data: {e}")
        order_book_data = None

    now = datetime.now()
    fname = now.strftime('%Y-%m-%d_%H-%M') + '.txt'
    fpath = os.path.join(out_dir, fname)

    try:
        with open(fpath, 'w', encoding='utf-8') as f:
            json.dump(order_book_data, f, ensure_ascii=False, indent=2)
        print(f"[{datetime.now()}] Записан файл: {fpath}")
    except Exception as e:
        print(f"[{datetime.now()}] Ошибка при записи файла {fpath}: {e}")