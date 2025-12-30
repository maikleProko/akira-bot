from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.masters.instances.choch_master import CHoCHMaster


#SYMBOLS
symbol1 = 'BTC'
symbol2 = 'USDT'


#FOR HISTORICAL TRADING
realtime = False
start_time_string='2025/09/01 00:00'
end_time_string='2025/12/30 00:00'
is_printed_ticks = False
balance_usdt = BalanceUSDT(7712)



#MARKET PROCESSES
MARKET_PROCESSES = [
    CHoCHMaster('BTC', 'USDT', balance_usdt),
    CHoCHMaster('ATOM', 'USDT', balance_usdt),
    CHoCHMaster('XTZ', 'USDT', balance_usdt),
    CHoCHMaster('WLD', 'USDT', balance_usdt),
]