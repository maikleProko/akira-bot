import cv2
import pytesseract
from classes.generate_order_book_data.ocr.constants.OrderBookParserConstants import *

pytesseract.pytesseract.tesseract_cmd = r'D:\Utilities\Tesseract-OCR\tesseract.exe'


class AxisImporter:
    def __init__(self):
        self.dimension = 0
        self.axis_data = []
        self.axis_y_begin = {}
        self.axis_y_end = {}
        self.image_height = 0

    def import_axis_data(self, image_path: str):
        axis_data = []
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise FileNotFoundError(f"Изображение не найдено по пути: {image_path}")

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            axis_data = pytesseract.image_to_data(gray, output_type=pytesseract.
                                                  Output.DICT, lang='eng', config='--psm 6')

            self.image_height = img.shape[0]
        except Exception as e:
            print('Ошибка: ' + str(e))
        self.axis_data = axis_data

    def parse_to_axis_data(self):
        required_keys = ['left', 'top', 'text', 'height']
        if not all(key in self.axis_data for key in required_keys):
            raise ValueError("Отсутствует один или несколько необходимых ключей")

        lengths = [len(self.axis_data[key]) for key in required_keys]
        if len(set(lengths)) != 1:
            raise ValueError("Все списки должны иметь одинаковую длину")

        result = []
        length = len(self.axis_data[required_keys[0]])  # длина любого списка

        for i in range(length):
            small_dict = {
                'left': self.axis_data['left'][i],
                'top': self.axis_data['top'][i],
                'text': self.axis_data['text'][i],
                'height': self.axis_data['height'][i],
            }
            result.append(small_dict)
        self.axis_data = result

    def fix_comma(self):
        def find_common_pattern(numbers):
            positions = []
            for num in numbers:
                if ',' in num:
                    pos = len(num.split(',')[0])
                    positions.append(pos)

            if positions:
                return max(set(positions), key=positions.count)
            return None

        def add_comma_at_position(num_str, position):
            num_str = num_str.replace(',', '')
            return f"{num_str[:position]}.{num_str[position:]}"

        numbers = [item['text'] for item in self.axis_data]

        common_position = find_common_pattern(numbers)

        def process_item(item):
            num_str = item['text']
            if common_position is not None:
                formatted_num = add_comma_at_position(num_str, common_position)
            else:
                formatted_num = num_str
            return {**item, 'text': formatted_num}

        self.axis_data = [process_item(item) for item in self.axis_data]

    def clean_data(self):
        self.axis_data = [item for item in self.axis_data if item['text'] != '' and item['text'] != 'owen']

    def get_axis_data(self):
        self.import_axis_data(INPUT_AXIS_FILE)
        self.parse_to_axis_data()
        self.clean_data()
        self.fix_comma()
