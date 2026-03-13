from scenarios.market.buyers.balance_usdt import BalanceUSDT
from scenarios.masters.instances.choch_kimmi_master import CHoCHKimmiMaster
from scenarios.masters.instances.choch_chippi_master import CHoCHChippiMaster

market_modes = [
    {'fee': 0, 'risk_usdt': 30, 'min_profit_usdt': 50, 'is_used_broker': False, 'realtime': False, 'balance_usdt': BalanceUSDT(7712)},
    {'fee': 0,'risk_usdt': 0.12,'min_profit_usdt': 0.2,'is_used_broker': True,'realtime': True, 'balance_usdt': BalanceUSDT(62)},
]

selected_mode_index = 1

_mode = market_modes[selected_mode_index]
realtime = _mode['realtime']

start_time_string = '2025/01/01 00:00'
end_time_string = '2025/02/01 00:00'
is_printed_ticks = False


# MARKET PROCESSES
MARKET_PROCESSES = [
    CHoCHChippiMaster('BTC', 'USDC', balance_usdt=_mode['balance_usdt'], fee=_mode['fee'], risk_usdt=_mode['risk_usdt'], min_profit_usdt=_mode['min_profit_usdt'], is_used_broker=_mode['is_used_broker']),
]
