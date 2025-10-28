from scenarios.parsers.arbitrage_parser.instances.careful_kucoin_arbitrage_parser import CarefulKuCoinArbitrageParser
from scenarios.parsers.arbitrage_parser.instances.careful_okx_arbitrage_parser import CarefulOKXArbitrageParser
from scenarios.parsers.arbitrage_parser.instances.kucoin_arbitrage_parser import KuCoinArbitrageParser
from scenarios.parsers.history_market_parser.instances.history_binance_parser import HistoryBinanceParser


#SYMBOLS
symbol1 = 'BTC'
symbol2 = 'USDT'


#FOR HISTORICAL TRADING
realtime = True
start_time_string='2025/10/21 06:30'
end_time_string='2025/10/22 15:55'

#PROCESSES (STRATEGIES)

arbitrage_parser = CarefulOKXArbitrageParser(
    production=False,
    deposit=1500,
    api_key='',
    api_secret='',
    api_passphrase='kxmb263ru'
)

#MARKET PROCESSES
MARKET_PROCESSES = [
    arbitrage_parser
]