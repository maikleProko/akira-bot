from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.masters.instances.choch_nofee_master import CHoCHNoFeeMaster
from scenarios.masters.instances.kama_yung_master import KamaYungMaster

realtime = False
start_time_string='2025/06/01 00:00'
end_time_string='2025/07/30 00:00'
is_printed_ticks = False
balance_usdt = BalanceUSDT(7712)



#MARKET PROCESSES
MARKET_PROCESSES = [
    CHoCHNoFeeMaster('BTC', 'USDC', balance_usdt),
    CHoCHNoFeeMaster('ETH', 'USDC', balance_usdt),
    CHoCHNoFeeMaster('SOL', 'USDC', balance_usdt),
    CHoCHNoFeeMaster('AVAX', 'USDC', balance_usdt),
    CHoCHNoFeeMaster('LINK', 'USDC', balance_usdt),
    CHoCHNoFeeMaster('XRP', 'USDC', balance_usdt)

]
