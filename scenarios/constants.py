from scenarios.parsers.arbitrage_parser.instances.bybit_careful_arbitrage_parser import BybitCarefulArbitrageParser

#SYMBOLS
symbol1 = 'BTC'
symbol2 = 'USDT'


#FOR HISTORICAL TRADING
realtime = True
start_time_string='2025/10/21 06:30'
end_time_string='2025/10/22 15:55'

#PROCESSES (STRATEGIES)


arbitrage_parser = BybitCarefulArbitrageParser(
    production=True,
    deposit=0.00011,
    api_key='',
    api_secret='',
    api_passphrase='-1',
    strict_coin='BTC',
    strict=True,
    min_profit=0.006,
    max_profit=10.009,
    fee_rate=0.0012,
    ignore=['RLUSD'],
    only_once=True,
    abusing_only_once=True
)

'''
arbitrage_parser = BybitCarefulArbitrageParser(
    production=True,
    deposit=0.00011,
    api_key='3jClHWcEvB2fBLp9Z0',
    api_secret='DxSDnMvnKJzDXkvNKOKkF5DU5DtXFkq0CG6y',
    api_passphrase='-1',
    strict_coin='BTC',
    strict=True,
    min_profit=0.006,
    max_profit=10.009,
    fee_rate=0.0012,
    ignore=['RLUSD'],
    only_once=True,
    abusing_only_once=True
)
'''


#MARKET PROCESSES
MARKET_PROCESSES = [
    arbitrage_parser
]