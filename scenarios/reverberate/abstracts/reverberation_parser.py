import json
import re
from time import sleep
from datetime import datetime


import requests
from bs4 import BeautifulSoup

from core.selenium.SeleniumProcessor import SeleniumProcessor


def clean_string(s):
    return re.sub(r'[^0-9\.]', '', s)

class ReverberationParser(SeleniumProcessor):
    def __init__(self):
        if not self.url:
            self.url = ''

        if not self.market:
            self.market = ''
        self.particles = []
        self.file_path = 'files/reverberation_data_' + self.market + '.json'
        super().__init__()

    def go_page(self):
        self.go_no_check(self.url)

    def get_particles(self):
        return {
            'buy_value': 50,
            'sell_value': 50
        }

    def is_have_particles(self):
        return False


    def get_reverberation(self, array, percent):
        count = sum(1 for item in array if item['buy_value'] > percent)
        return count / len(array)


    def push_particles(self):
        self.particles = []
        for i in range(0, 60):
            self.particles.append(self.get_particles())
            print('check: ' + str(i))
            sleep(1)
        print(self.particles)


    def generate_particles(self):
        if self.is_have_particles():
            self.push_particles()

            with open(self.file_path, "r", encoding="utf-8") as file:
                existing_data = json.load(file)

            existing_data.append({
                'date': datetime.now(),
                'reverberation_50': self.get_reverberation(self.particles, 50),
                'reverberation_75': self.get_reverberation(self.particles, 75)
            })

            with open(self.file_path, "w", encoding="utf-8") as file:
                json.dump(existing_data, file, indent=4, default=str)

