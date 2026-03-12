from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.masters.instances.choch_kimmi_master import CHoCHKimmiMaster

realtime = True
start_time_string='2025/01/01 00:00'
end_time_string='2025/02/01 00:00'
is_printed_ticks = False
balance_usdt = BalanceUSDT(62)



#MARKET PROCESSES
MARKET_PROCESSES = [
    CHoCHKimmiMaster('BTC', 'USDC', balance_usdt),
]





