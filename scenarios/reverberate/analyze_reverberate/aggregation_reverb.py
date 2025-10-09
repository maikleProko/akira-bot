import json
import ast
from typing import List, Dict, Any
from statistics import mean
import math
from pathlib import Path

def load_list_of_dicts(path: str) -> List[Dict[str, Any]]:
    s = Path(path).read_text(encoding='utf-8')
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)

def is_number(x):
    return isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x))

def average_numbers(values):
    vals = [v for v in values if is_number(v)]
    return mean(vals) if vals else None

def merge_pairwise(
    a: List[Dict[str, Any]],
    b: List[Dict[str, Any]],
    keep_tail: bool = True,
    prefer: str = 'a'  # how to pick non-numeric / date when different: 'a' or 'b'
) -> List[Dict[str, Any]]:
    n = min(len(a), len(b))
    out = []

    for i in range(n):
        da = a[i] or {}
        db = b[i] or {}
        merged = {}

        # handle date specially: prefer a's date, but warn if differ (simple print)
        date_a = da.get('date')
        date_b = db.get('date')
        if date_a is not None and date_b is not None and date_a != date_b:
            # предупреждение — можно заменить логированием
            print(f'Warning: differing dates at index {i}: {date_a!r} vs {date_b!r}; preferring {prefer}')
        merged['date'] = date_a if prefer == 'a' else date_b
        if merged['date'] is None:
            merged['date'] = date_a if date_a is not None else date_b

        # all keys except date
        keys = (set(da.keys()) | set(db.keys())) - {'date'}
        for k in keys:
            va = da.get(k)
            vb = db.get(k)

            # numeric averaging if either side numeric
            if is_number(va) or is_number(vb):
                avg = average_numbers([va, vb])
                merged[k] = avg
            else:
                # non-numeric: choose by preference if available, otherwise whichever exists
                if prefer == 'a':
                    merged[k] = va if va is not None else vb
                else:
                    merged[k] = vb if vb is not None else va

        out.append(merged)

    # опционально добавить хвост более длинного списка без изменений
    if keep_tail:
        if len(a) > n:
            out.extend(a[n:])
        elif len(b) > n:
            out.extend(b[n:])

    return out

# Пример использования:
def aggregation_reverb():
    p1 = 'files/reverberation_data_binance1.json'
    p2 = 'files/reverberation_data_binance2.json'
    a = load_list_of_dicts(p1)
    b = load_list_of_dicts(p2)

    merged = merge_pairwise(a, b, keep_tail=True, prefer='a')

    # сохранить в файл
    out_path = 'files/reverberation_data_merged.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, ensure_ascii=False, indent=2)

    print(f'Merged {min(len(a), len(b))} pairs; result saved to {out_path}')