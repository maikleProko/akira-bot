from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.masters.instances.choch_master import CHoCHMaster


#FOR HISTORICAL TRADING
realtime = False
start_time_string='2025/12/01 00:00'
end_time_string='2025/12/10 07:00'
is_printed_ticks = False
balance_usdt = BalanceUSDT(7712)



#MARKET PROCESSES
MARKET_PROCESSES = [
    CHoCHMaster('BTC', 'USDT', balance_usdt, 'loading'),    # Основная крипта
    CHoCHMaster('ETH', 'USDT', balance_usdt, 'loading'),    # Крупная альта, но коррелирует с BTC ~0.8–0.9
    CHoCHMaster('SOL', 'USDT', balance_usdt, 'loading'),    # Solana — отдельная экосистема, корреляция ниже
    CHoCHMaster('XRP', 'USDT', balance_usdt, 'loading'),    # Ripple — часто движется независимо (регуляторные новости)
    CHoCHMaster('ADA', 'USDT', balance_usdt, 'loading'),    # Cardano — свой цикл
    CHoCHMaster('AVAX', 'USDT', balance_usdt, 'loading'),   # Avalanche — L1 с собственной динамикой
    CHoCHMaster('DOT', 'USDT', balance_usdt, 'loading'),    # Polkadot — парачейн-экосистема
    CHoCHMaster('LINK', 'USDT', balance_usdt, 'loading'),   # Chainlink — оракулы, реагирует на DeFi-активность
    CHoCHMaster('MATIC', 'USDT', balance_usdt, 'loading'),  # Polygon — L2, корреляция с ETH, но слабее
    CHoCHMaster('NEAR', 'USDT', balance_usdt, 'loading'),   # NEAR Protocol — AI-направление + свой хайп
]