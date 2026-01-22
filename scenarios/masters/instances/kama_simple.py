from scenarios.market.buyers.buyer_kama import BuyerKAMA
from scenarios.market.buyers.buyer_tpsl import BuyerTPSL
from scenarios.market.regulators.regulator_nweatr import RegulatorNWEATR
from scenarios.masters.abstracts.market_master import MarketMaster
from scenarios.parsers.history_market_parser.instances.history_binance_parser import HistoryBinanceParser
from scenarios.parsers.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.parsers.indicators.instances.choch_indicator import CHoCHIndicator
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.strategies.instances.choch_percent_strategy import CHoCHPercentStrategy
from scenarios.strategies.instances.kama_strategy import KamaStrategy


class KamaSimple(MarketMaster):
    def __init__(self, symbol1, symbol2, balance_usdt, mode='generating'):
        super().__init__()
        history_market_parser_60m = HistoryBinanceParser(symbol1, symbol2, 60, 1000, mode)
        history_market_parser_60m_SOL = HistoryBinanceParser('SOL', symbol2, 60, 1000, mode)
        history_market_parser_60m_ETH = HistoryBinanceParser('ETH', symbol2, 60, 1000, mode)
        kama_indicator_60m = KamaIndicator(history_market_parser_60m, 7, 2, 30)
        kama_indicator_60m_SOL = KamaIndicator(history_market_parser_60m_SOL, 7, 2, 30)
        kama_indicator_60m_ETH = KamaIndicator(history_market_parser_60m_SOL, 7, 2, 30)
        buyer = BuyerKAMA(
            symbol1=symbol1,
            symbol2=symbol2,
            history_market_parser=history_market_parser_60m,
            balance_usdt=balance_usdt,
            kama_indicator=kama_indicator_60m,
            kama_indicator_other=kama_indicator_60m_SOL,
            kama_indicator_other2=kama_indicator_60m_ETH,
            fee=0
        )

        self.market_processes = [
            history_market_parser_60m,
            history_market_parser_60m_SOL,
            history_market_parser_60m_ETH,
            kama_indicator_60m,
            kama_indicator_60m_SOL,
            kama_indicator_60m_ETH,
            buyer
        ]