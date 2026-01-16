from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.masters.instances.choch_nofee_master import CHoCHNoFeeMaster

realtime = False
start_time_string='2025/06/01 00:00'
end_time_string='2025/06/15 00:00'
is_printed_ticks = False
balance_usdt = BalanceUSDT(7712)



#MARKET PROCESSES
MARKET_PROCESSES = [
    CHoCHNoFeeMaster('BTC', 'USDT', balance_usdt),
]