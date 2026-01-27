from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.masters.instances.choch_nofee_master import CHoCHNoFeeMaster
from scenarios.masters.instances.kama_simple import KamaSimple
from scenarios.masters.instances.obflow_master import ObFlowMaster

realtime = False
start_time_string='2025/09/01 00:00'
end_time_string='2025/10/30 00:00'
is_printed_ticks = False
balance_usdt = BalanceUSDT(10282)



#MARKET PROCESSES
MARKET_PROCESSES = [
    CHoCHNoFeeMaster('BTC', 'USDC', balance_usdt),
    CHoCHNoFeeMaster('ETH', 'USDC', balance_usdt),
    CHoCHNoFeeMaster('SOL', 'USDC', balance_usdt),
    CHoCHNoFeeMaster('AVAX', 'USDC', balance_usdt),
    CHoCHNoFeeMaster('LINK', 'USDC', balance_usdt),
    CHoCHNoFeeMaster('XRP', 'USDC', balance_usdt)
]
