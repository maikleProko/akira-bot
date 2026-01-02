from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.masters.instances.choch_master import CHoCHMaster


#FOR HISTORICAL TRADING
realtime = False
start_time_string='2025/12/03 00:00'
end_time_string='2025/12/10 07:00'
is_printed_ticks = False
balance_usdt = BalanceUSDT(7712)



#MARKET PROCESSES
MARKET_PROCESSES = [
    CHoCHMaster('BTC', 'USDT', balance_usdt),    # Основная крипта
    CHoCHMaster('ETH', 'USDT', balance_usdt),    # Крупная альта, но коррелирует с BTC ~0.8–0.9
    CHoCHMaster('SOL', 'USDT', balance_usdt),    # Solana — отдельная экосистема, корреляция ниже
    CHoCHMaster('XRP', 'USDT', balance_usdt),    # Ripple — часто движется независимо (регуляторные новости)
    CHoCHMaster('DOT', 'USDT', balance_usdt),    # Polkadot — парачейн-экосистема
    CHoCHMaster('LINK', 'USDT', balance_usdt),   # Chainlink — оракулы, реагирует на DeFi-активность
]