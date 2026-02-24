from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.masters.instances.choch_nofee_master import CHoCHNoFeeMaster
from scenarios.masters.instances.kama_yung_master import KamaYungMaster
from scenarios.masters.instances.nofee_master import NoFeeMaster

realtime = True
start_time_string='2025/06/01 00:00'
end_time_string='2025/07/30 00:00'
is_printed_ticks = False
balance_usdt = BalanceUSDT(47)



#MARKET PROCESSES
MARKET_PROCESSES = [
    CHoCHNoFeeMaster('BTC', 'USDC', balance_usdt),
]
