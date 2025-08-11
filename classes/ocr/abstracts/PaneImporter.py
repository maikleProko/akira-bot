import math
from copy import deepcopy

import cv2
import numpy as np
from PIL import Image

from classes.ocr.constants.ChartParserConstants import *


class PaneImporter:
    def __init__(self):
        self.pane_data = []
        pass

    def is_corrected(self, rgb, color, delta):
        return all(abs(int(a) - int(b)) <= delta for a, b in zip(rgb, color))

    def import_pane_data(self, image_path):
        img = Image.open(image_path)
        target_colors = {
            (255, 82, 82): 'red',  # ff5252 в RGB
            (119, 123, 134): 'gray',  # 777b86 в RGB
            (0, 230, 118): 'green'  # 00e676 в RGB
        }
        w, h = img.size
        pixels = img.load()
        delta = 2

        # Нормализуем ключи цветов на tuple
        color_map = {tuple(k): v for k, v in target_colors.items()}

        rectangles = []
        active = {}

        for y in range(h):
            # Собираем все runs в текущей строке
            runs = []
            x = 0
            while x < w:
                rgb = pixels[x, y]
                matched_color_key = None
                matched_name = None
                for color_key, cname in color_map.items():
                    if self.is_corrected(rgb, color_key, delta):
                        matched_color_key = color_key
                        matched_name = cname
                        break

                if matched_color_key is not None:
                    x1 = x
                    x += 1
                    while x < w and self.is_corrected(pixels[x, y], matched_color_key, delta):
                        x += 1
                    x2 = x
                    runs.append((x1, x2, matched_name))
                else:
                    x += 1

            run_keys = {(r[0], r[1], r[2]): r for r in runs}

            # Проверяем продолжаются ли активные прямоугольники
            new_active = {}
            for key, rect in active.items():
                if key in run_keys:
                    # продолжается на этой строке
                    rect['y_last'] = y
                    new_active[key] = rect
                    del run_keys[key]
                else:
                    # не продолжается -> закрываем (y2 = rect['y_last']+1)
                    rectangles.append({
                        'x1': rect['x1'],
                        'y1': rect['y1'],
                        'x2': rect['x2'],
                        'y2': rect['y_last'] + 1,
                        'width': rect['y_last'] + 1 - rect['y1'],
                        'color': key[2]
                    })

            # Оставшиеся run_keys — новые прямоугольники, начинающиеся на этой строке
            for key in run_keys.keys():
                x1, x2, cname = key
                new_active[key] = {'x1': x1, 'y1': y, 'x2': x2, 'y_last': y}

            active = new_active

        # Закрываем все оставшиеся активные прямоугольники по окончании изображения
        for key, rect in active.items():
            rectangles.append({
                'x1': rect['x1'],
                'y1': rect['y1'],
                'x2': rect['x2'],
                'y2': rect['y_last'] + 1,
                'width': rect['y_last'] + 1 - rect['y1'],
                'color': key[2]
            })

        self.pane_data = rectangles

    def fill_gaps(self):
        if not self.pane_data:
            return []

        rects_sorted = sorted(deepcopy(self.pane_data), key=lambda r: (r['y1'], r['x1']))
        result = []

        for i in range(len(rects_sorted) - 1):
            curr = rects_sorted[i]
            nxt = rects_sorted[i + 1]
            result.append(curr)

            # Если есть разрыв (gap) по Y
            if nxt['y1'] > curr['y2']:
                # вычисляем начало заполнителя по формуле: ceil(x1 + x2/2)
                start = math.ceil(curr['x1'] + curr['x2'] / 2.0)
                filler = {
                    'x1': int(start),
                    'x2': int(start) + 1,
                    'y1': int(curr['y2']),
                    'y2': int(nxt['y1']),
                    'color': curr['color']
                }
                result.append(filler)

        # Добавляем последний исходный прямоугольник
        result.append(rects_sorted[-1])
        self.pane_data = result

    def restore_pane_data(self, top_border, bottom_border, round_digits=None, inplace=False):
        if not isinstance(top_border, dict) or not isinstance(bottom_border, dict):
            raise ValueError("top_border и bottom_border должны быть dict с ключами 'pixel_y' и 'real_y'")
        for k in ('pixel_y', 'real_y'):
            if k not in top_border or k not in bottom_border:
                raise ValueError("top_border и bottom_border должны содержать ключи 'pixel_y' и 'real_y'")

        p_top = float(top_border['pixel_y'])
        r_top = float(top_border['real_y'])
        p_bot = float(bottom_border['pixel_y'])
        r_bot = float(bottom_border['real_y'])

        if p_bot == p_top:
            raise ValueError("pixel_y верхней и нижней границ совпадают — невозможна линейная аппроксимация")

        a = (r_bot - r_top) / (p_bot - p_top)
        b = r_top - a * p_top

        def pixel_to_real(py):
            val = a * float(py) + b
            if round_digits is not None:
                return round(val, round_digits)
            return val

        output = self.pane_data if inplace else deepcopy(self.pane_data)

        for rect in output:
            if 'y1' not in rect or 'y2' not in rect:
                continue
            y1r = pixel_to_real(rect['y1'])
            y2r = pixel_to_real(rect['y2'])
            # height в реальных единицах
            hreal = y2r - y1r
            rect['y1_real'] = y1r
            rect['y2_real'] = y2r
            rect['height_real'] = abs(hreal)
            if inplace:
                rect['y1'] = y1r
                rect['y2'] = y2r

        self.pane_data = output

    def get_pane_data(self):
        self.import_pane_data(INPUT_PANE_FILE)
        self.fill_gaps()
