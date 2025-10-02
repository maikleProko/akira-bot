import json

from core.selenium.SeleniumProcessor import SeleniumProcessor


class LeaderboardParser(SeleniumProcessor):
    def __init__(self):
        if not self.file_path:
            self.file_path = '/files/common_traders.json'
        self.traders = []
        self.trades = []
        super().__init__()


    def _load_traders(self) -> None:
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            self.traders = []; return
        if isinstance(data, dict) and 'traders' in data:
            self.root_is_dict = True; self.traders = data['traders']
        elif isinstance(data, list):
            self.traders = data
        else:
            self.traders = []
        if self.traders and isinstance(self.traders[0], dict):
            for k in ('trades', 'traders'):
                if k in self.traders[0]:
                    self.per_trader_list_key = k; break

    def _save_traders(self) -> None:
        payload = {'traders': self.traders} if self.root_is_dict else self.traders
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=4, ensure_ascii=False, default=str)