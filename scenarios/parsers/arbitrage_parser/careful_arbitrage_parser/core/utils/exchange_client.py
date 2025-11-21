from abc import abstractmethod, ABC


class ExchangeClient(ABC):
    @abstractmethod
    def init_clients(self, api_key, api_secret, api_passphrase):
        pass

    @abstractmethod
    def test_clients(self):
        pass

    @abstractmethod
    def test_ticker(self):
        pass

    @abstractmethod
    def fetch_available_coins(self):
        pass

    @abstractmethod
    def fetch_symbols(self):
        pass

    @abstractmethod
    def fetch_tickers(self):
        pass

    @abstractmethod
    def check_balance(self, asset):
        pass

    @abstractmethod
    def fetch_ticker_price(self, symbol):
        pass

    @abstractmethod
    def create_order_params(self, symbol, direction, ordType, adjusted_amount):
        pass

    @abstractmethod
    def place_order(self, order_params):
        pass

    @abstractmethod
    def cancel_order(self, symbol, order_id):
        pass

    @abstractmethod
    def get_order_details(self, symbol, order_id):
        pass

    @abstractmethod
    def fetch_current_prices(self):
        pass

    @abstractmethod
    def check_pair_available(self, sym, direction, price_map):
        pass