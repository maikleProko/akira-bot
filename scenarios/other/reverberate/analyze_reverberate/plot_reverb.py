import json, ast
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime
import numpy as np

def plot_reverb(path="files/reverb_full_08.json"):
    try:
        from dateutil.parser import parse as dparse
    except Exception:
        dparse = None
    # Чтение и разбор
    with open(path, "r", encoding="utf-8") as f:
        s = f.read()
    try:
        data = json.loads(s)
    except Exception:
        data = ast.literal_eval(s)

    if not isinstance(data, list):
        raise ValueError("Ожидался список записей")

    entries = []
    for item in data:
        if not isinstance(item, dict):
            continue
        date_s = item.get("date")
        if not date_s:
            continue
        # парсинг даты (попытка dateutil, иначе несколько форматов)
        dt = None
        if dparse:
            try:
                dt = dparse(date_s)
            except Exception:
                dt = None
        if dt is None:
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
                try:
                    dt = datetime.strptime(date_s, fmt)
                    break
                except Exception:
                    pass
        if dt is None:
            continue
        entries.append((dt, item))

    if not entries:
        raise ValueError("Нет корректных записей с датой")

    # сортировка по дате
    entries.sort(key=lambda x: x[0])
    xs = [e[0] for e in entries]

    fields = [
        ("reverberation_50", "reverberation_50"),
        ("reverberation_75", "reverberation_75"),
        ("reverberation_75_1_minutes", "reverberation_75_1_minutes"),
        ("reverberation_75_5_minutes", "reverberation_75_5_minutes"),
    ]
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red"]

    ys_list = []
    for key, _ in fields:
        ys = []
        for _, item in entries:
            try:
                v = item.get(key)
                if v is None:
                    ys.append(np.nan)
                else:
                    ys.append(float(v))
            except Exception:
                ys.append(np.nan)
        ys_list.append(ys)

    # Построение
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for (key, label), color, ys in zip(fields, [*colors], ys_list):
        ax.plot(xs, ys, label=label, color=color, marker=None)

    ax.set_ylim(0, 1)
    ax.set_xlabel("date")
    ax.set_ylabel("value")
    ax.set_title("Reverberation metrics over time")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")

    # Форматирование оси X
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    fig.autofmt_xdate(rotation=0)
    plt.tight_layout()
    plt.show()