from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.masters.instances.choch_nofee_master import CHoCHNoFeeMaster
from scenarios.masters.instances.kama_simple import KamaSimple
from scenarios.masters.instances.obflow_master import ObFlowMaster

realtime = True
start_time_string='2025/12/01 00:00'
end_time_string='2026/01/20 00:00'
is_printed_ticks = False
balance_usdt = BalanceUSDT(7712)



#MARKET PROCESSES
MARKET_PROCESSES = [
    CHoCHNoFeeMaster('BTC', 'USDT', balance_usdt),
    CHoCHNoFeeMaster('ETH', 'USDT', balance_usdt),
    CHoCHNoFeeMaster('SOL', 'USDT', balance_usdt)
]