from scenarios.parsers.arbitrage_parser.instances.mexc_careful_arbitrage_parser import MexcCarefulArbitrageParser
from scenarios.parsers.arbitrage_parser.instances.okx_careful_arbitrage_parser import OkxCarefulArbitrageParser

#SYMBOLS
symbol1 = 'BTC'
symbol2 = 'USDT'


#FOR HISTORICAL TRADING
realtime = True
start_time_string='2025/10/21 06:30'
end_time_string='2025/10/22 15:55'

#PROCESSES (STRATEGIES)

okx_arbitrage_parser = OkxCarefulArbitrageParser(
    production=True,
    deposit=0.00061172,
    api_key='',
    api_secret='',
    api_passphrase='Kxmb263ru-',
    strict_coin='BTC',
    strict = True
)

mexc_arbitrage_parser = MexcCarefulArbitrageParser(
    production=False,
    deposit=0.00061172,
    api_key='',
    api_secret='',
    api_passphrase='Kxmb263ru-',
    strict_coin='BTC',
    strict = True
)

#MARKET PROCESSES
MARKET_PROCESSES = [
    okx_arbitrage_parser
]