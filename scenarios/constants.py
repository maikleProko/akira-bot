from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.masters.instances.choch_nofee_master import CHoCHNoFeeMaster
from scenarios.masters.instances.kama_simple import KamaSimple
from scenarios.masters.instances.obflow_master import ObFlowMaster

realtime = True
start_time_string='2025/06/01 00:00'
end_time_string='2025/06/30 00:00'
is_printed_ticks = False
balance_usdt = BalanceUSDT(57)



#MARKET PROCESSES
MARKET_PROCESSES = [
    CHoCHNoFeeMaster('BTC', 'USDC', balance_usdt)
]