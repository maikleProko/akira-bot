from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.masters.instances.choch_nofee_master import CHoCHNoFeeMaster
from scenarios.masters.instances.choch_nofeeroyal_master import CHoCHNoFeeRoyalMaster
from scenarios.masters.instances.kama_yung_master import KamaYungMaster
from scenarios.masters.instances.nofee_master import NoFeeMaster

realtime = True
start_time_string='2025/06/01 00:00'
end_time_string='2025/07/30 00:00'
is_printed_ticks = False
balance_usdt = BalanceUSDT(61)



#MARKET PROCESSES
MARKET_PROCESSES = [
    CHoCHNoFeeRoyalMaster('BTC', 'USDC', balance_usdt),
    CHoCHNoFeeRoyalMaster('ETH', 'USDC', balance_usdt),
    CHoCHNoFeeRoyalMaster('SOL', 'USDC', balance_usdt),
    CHoCHNoFeeRoyalMaster('LINK', 'USDC', balance_usdt),
    CHoCHNoFeeRoyalMaster('TRX', 'USDC', balance_usdt),
]
