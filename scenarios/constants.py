from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.masters.instances.choch_master import CHoCHMaster


#FOR HISTORICAL TRADING
realtime = False
start_time_string='2025/12/10 14:15'
end_time_string='2026/01/04 18:00'
is_printed_ticks = False
balance_usdt = BalanceUSDT(7712)



#MARKET PROCESSES
MARKET_PROCESSES = [
    CHoCHMaster('BTC', 'USDT', balance_usdt),    # Основная крипта
]