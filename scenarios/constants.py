from scenarios.parsers.arbitrage_parser.instances.bybit_careful_arbitrage_parser import BybitCarefulArbitrageParser
from scenarios.parsers.arbitrage_parser.instances.okx_careful_arbitrage_parser import OkxCarefulArbitrageParser

#SYMBOLS
symbol1 = 'BTC'
symbol2 = 'USDT'


#FOR HISTORICAL TRADING
realtime = True
start_time_string='2025/10/21 06:30'
end_time_string='2025/10/22 15:55'

#PROCESSES (STRATEGIES)

arbitrage_parser = BybitCarefulArbitrageParser(
    production=False,
    deposit=0.00061172,
    #api_key='',
    #api_secret='',
    api_passphrase='Kxmb263ru-',
    strict_coin='BTC',
    strict = True,
    min_profit=0.005,
    max_profit=10.009,
    fee_rate=0.001,
    #ignore = ['USDC', 'EUR', 'AED', 'AUD', 'USD']
)

#MARKET PROCESSES
MARKET_PROCESSES = [
    arbitrage_parser
]