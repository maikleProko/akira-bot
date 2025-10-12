from scenarios.reverberate.analyze_reverberate.aggregation_reverb import aggregation_reverb
from scenarios.reverberate.analyze_reverberate.fast_reverb import fast_reverb
from scenarios.reverberate.analyze_reverberate.plot_reverb import plot_reverb

from scenarios.reverberate.history_reverberate.history_reverberate import history_reverberate
from scenarios.reverberate.selenium_reverberate.selenium_reverberate import selenium_reverberate



def reverberate():
    history_reverberate('binance', '2025-10-01 10:00:00', '2025-10-10 18:00:00', 'ETHUSDT')
