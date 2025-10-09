def fast_reverb(path="files/reverb_full_08.json"):
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
            v5 = float(item.get("reverberation_75_5_minutes"))
        except Exception:
            continue
        if v1 > 0.7 and v5 > 0.5:
            result.append(item)
    print(result)
    return result