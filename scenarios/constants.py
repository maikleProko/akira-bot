from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.masters.instances.choch_master import CHoCHMaster


#FOR HISTORICAL TRADING
from scenarios.masters.instances.obflow_master import ObFlowMaster

realtime = False
start_time_string='2025/12/02 00:00'
end_time_string='2026/01/05 10:00'
is_printed_ticks = False
balance_usdt = BalanceUSDT(27712)



#MARKET PROCESSES
MARKET_PROCESSES = [
    ObFlowMaster('BTC', 'USDT', balance_usdt),
    ObFlowMaster('ETH', 'USDT', balance_usdt),  # Крупная альта, но коррелирует с BTC ~0.8–0.9
    ObFlowMaster('SOL', 'USDT', balance_usdt),  # Solana — отдельная экосистема, корреляция ниже
    ObFlowMaster('XRP', 'USDT', balance_usdt),  # Ripple — часто движется независимо (регуляторные новости)
    ObFlowMaster('DOT', 'USDT', balance_usdt),  # Polkadot — парачейн-экосистема
    ObFlowMaster('LINK', 'USDT', balance_usdt),  # Chainlink — оракулы, реагирует на DeFi-активность

]