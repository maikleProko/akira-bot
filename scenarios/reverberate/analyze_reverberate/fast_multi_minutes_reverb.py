import json


def save_to_file(array):
    # сохранить в файл
    out_path = f'files/fast_multi_minutes_reverb.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(array, f, ensure_ascii=False, indent=2)

    print(f'fast reverberation result saved to {out_path}')



def fast_multi_minutes_reverb(path="files/reverb_full_08.json"):
    import json, ast
    with open(path, "r", encoding="utf-8") as f:
        s = f.read()
    try:
        data = json.loads(s)
    except Exception:
        data = ast.literal_eval(s)
    result = []
    for item in data:
        try:
            v1 = float(item.get("reverberation_75_1_minutes"))
            v2 = float(item.get("reverberation_75_2_minutes"))
            v3 = float(item.get("reverberation_75_3_minutes"))
            v4 = float(item.get("reverberation_75_4_minutes"))
            v5 = float(item.get("reverberation_75_5_minutes"))
        except Exception:
            continue
        if v1 >= 0.5 and v2 >= 0.5 and v3 >= 0.5 and v4 >= 0.5 and v5 >= 0.5:
            result.append(item)
    print(result)
    save_to_file(result)
    return result