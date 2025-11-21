
class CyclePriceFetcher:
    def __init__(self, exchange_client, logger):
        self.exchange_client = exchange_client
        self.logger = logger

    def get_current_price(self, sym, direction, current_prices, price_map):
        current_ticker = current_prices.get(sym)
        if current_ticker is None:
            self.logger.log_message(f"Ошибка: Недопустимый ответ API для {sym}, использую price_map")
            return price_map[sym]['bid'] if direction == 'sell' else price_map[sym]['ask']
        return current_ticker['sell'] if direction == 'sell' else current_ticker['buy']