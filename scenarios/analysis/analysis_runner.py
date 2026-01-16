import json

from scenarios.analysis.property.abstracts.property import Property
from scenarios.analysis.property.instances.atr_property import AtrProperty
from scenarios.analysis.property.instances.kama_property import KamaProperty
from scenarios.analysis.property.instances.smc_bos_choch_property import SMCBosChochProperty
from scenarios.analysis.property.instances.smc_orderblock_property import SMCOrderblokProperty

properties: list[Property] = [
    KamaProperty(120),
    KamaProperty(60),
    KamaProperty(15),
    AtrProperty(5),
    AtrProperty(1),
    SMCBosChochProperty(1),
    SMCOrderblokProperty(1)
]


def filter_by_main_combination():
    data_list = []
    with open(f'files/decisions/deals_check.txt', 'r', encoding='utf-8') as f:
        data_list = json.load(f)  # сразу парсит и возвращает список/словарь
    filtered = []
    count_tp = 0
    count_sl = 0
    for item in data_list:
        if item.get('a5_r') > 0.8 and item.get('a1_r') > 0.6 and abs(item.get('k15_sl')) < 0.79:
            filtered.append(item)
            if item['result'] == 'TP':
                count_tp += 1

            if item['result'] == 'SL':
                count_sl += 1

    print(filtered)
    print(f"TP: {count_tp}")
    print(f"SL: {count_sl}")
    return filtered


import json


def convert_custom_bools(obj):
    if isinstance(obj, dict):
        return {k: convert_custom_bools(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return type(obj)(convert_custom_bools(i) for i in obj)
    elif hasattr(obj, '__class__') and obj.__class__.__name__ == 'bool':
        try:
            return bool(obj)  # Attempt to convert to built-in bool
        except Exception:
            return str(obj)  # Fallback if conversion fails (e.g., log or inspect these cases)
    else:
        return obj

class AnalysisRunner:

    def get_data(self):
        with open(f'files/decisions/deals_test.txt', 'r', encoding='utf-8') as f:
            self.data = json.load(f)  # сразу парсит и возвращает список/словарь

    def _process_deal(self, deal):
        for property in properties:
            deal = property.build_market_processes(deal)
        self.deals.append(deal)

    def _process_deals(self):
        counter = 0
        for deal in self.data:
            self._process_deal(deal)
            counter += 1
            print(f"[AnalysisRunner] Analyzed: [{counter}/{len(self.data)}]")

    def save(self):
        # Pre-process to fix custom 'bool' objects
        with open('files/decisions/deals_modify.txt', 'w', encoding='utf-8') as file:
            json.dump(self.deals, file, ensure_ascii=False, indent=4)

    def __init__(self):
        self.data = []
        self.get_data()
        self.deals = []
        self._process_deals()
        self.save()


AnalysisRunner()
#filter_by_main_combination()

