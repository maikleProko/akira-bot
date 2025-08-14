from utils.decorators import doing_periodical_per_1_minute
from classes.generate_order_book_data.generate_order_book_data import generate_order_book_data
from classes.analyze_chart_data.analyze_chart_data import *


@doing_periodical_per_1_minute
def run():
    generate_order_book_data()


class MainProcessor:
    def __init__(self):
        run()