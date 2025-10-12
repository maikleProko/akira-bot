import json

from core.selenium.SeleniumProcessor import SeleniumProcessor, logger

import json
import logging
from time import sleep
from typing import List, Dict, Any

from selenium.common import ElementClickInterceptedException, WebDriverException
from selenium.webdriver import ActionChains
from selenium.webdriver.common.by import By
from datetime import datetime
import re
from decimal import Decimal


class LeaderboardParser(SeleniumProcessor):
    def __init__(self):
        if not self.file_path:
            self.file_path = '/files/common_traders.json'
        self.traders = []
        self.trades = []
        self.traders: List[Dict[str, Any]] = []
        self.root_is_dict = False
        self.per_trader_list_key = 'trades'   # will be autodetected on load
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



    def _element(self, by: By, selector: str):
        return self.get_element(by, selector)

    def _click(self, el) -> None:
        try:
            self.driver.execute_script("arguments[0].scrollIntoView({block:'center'});", el)
            sleep(0.2); ActionChains(self.driver).move_to_element(el).click().perform(); return
        except (ElementClickInterceptedException, WebDriverException):
            pass
        try:
            self.driver.execute_script(
                "var f = document.getElementById('trade-status-footer-v2'); if (f) { f.style.display='none'; }"
            )
            sleep(0.1); self.driver.execute_script("arguments[0].click();", el); return
        except Exception:
            pass
        try:
            self.driver.execute_script("arguments[0].click();", el); return
        except Exception as e:
            logger.exception("Click failed"); raise ElementClickInterceptedException(f"Не удалось кликнуть: {e}")

    def run(self) -> None:
        self.go_leaderboard()
        self.get_traders_all()
        self.parse_all_trades()

    def go_leaderboard(self):
        pass

    def get_traders_all(self):
        pass

    def parse_all_trades(self):
        pass