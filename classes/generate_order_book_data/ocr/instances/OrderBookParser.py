from classes.generate_order_book_data.ocr.abstracts.AxisImporter import AxisImporter
from classes.generate_order_book_data.ocr.abstracts.PaneImporter import PaneImporter



class OrderBookParser(AxisImporter, PaneImporter):
    def __init__(self):
        AxisImporter.__init__(self)
        PaneImporter.__init__(self)
        self.run()

    def get_dimension_of_order_book(self):
        first_cutoff_float = float(self.axis_data[1]['text'])
        second_cutoff_float = float(self.axis_data[2]['text'])

        first_cutoff_top = float(self.axis_data[1]['top'])
        second_cutoff_top = float(self.axis_data[2]['top'])

        self.dimension = abs(first_cutoff_float - second_cutoff_float) / abs(first_cutoff_top - second_cutoff_top)

    def get_axis_begins_ends(self):
        last_number = len(self.axis_data) - 1
        self.axis_y_begin = { 'pixel_y': float(self.axis_data[0]['top']) + float(self.axis_data[0]['height'])/2, 'real_y': float(self.axis_data[0]['text']) }
        self.axis_y_end = { 'pixel_y': float(self.axis_data[last_number]['top'])+float(self.axis_data[last_number]['height'])/2, 'real_y': float(self.axis_data[last_number]['text']) }

    def run(self):
        try:
            self.get_axis_data()
            self.get_axis_begins_ends()
            self.get_order_book_data()
            self.restore_order_book_data(self.axis_y_begin, self.axis_y_end)
        except Exception as e:
            print('returned none pane data')
            self.order_book_data = []
