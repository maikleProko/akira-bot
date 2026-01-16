from scenarios.market.buyers.buyer_tpsl import BuyerTPSL
from scenarios.market.regulators.regulator_nweatr import RegulatorNWEATR
from scenarios.masters.abstracts.market_master import MarketMaster
from scenarios.parsers.history_market_parser.instances.history_binance_parser import HistoryBinanceParser
from scenarios.parsers.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.parsers.indicators.instances.choch_indicator import CHoCHIndicator
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.strategies.instances.choch_percent_strategy import CHoCHPercentStrategy


class CHoCHPermitMaster(MarketMaster):
    def __init__(self, symbol1, symbol2, balance_usdt, mode='generating'):
        super().__init__()

        # PROCESSES (PARSERS)
        history_market_parser_1m = HistoryBinanceParser(symbol1, symbol2, 1, 1000, mode)
        history_market_parser_15m = HistoryBinanceParser(symbol1, symbol2, 15, 1000, mode)
        history_market_parser_60m = HistoryBinanceParser(symbol1, symbol2, 60, 1000, mode)
        nwe_bounds_indicator = NweBoundsIndicator(history_market_parser_1m)
        atr_bounds_indicator = AtrBoundsIndicator(history_market_parser_1m)
        kama_indicator_60m = KamaIndicator(history_market_parser_60m, 7, 2, 30)
        kama_indicator_15m = KamaIndicator(history_market_parser_15m, 7, 2, 30)
        choch_indicator = CHoCHIndicator(history_market_parser_15m)

        # PROCESSES (STRATEGIES)
        strategy = CHoCHPercentStrategy(
            history_market_parser_1m=history_market_parser_1m,
            history_market_parser_15m=history_market_parser_15m,
            kama_indicator_60m=kama_indicator_60m,
            kama_indicator_15m=kama_indicator_15m,
            choch_indicator=choch_indicator
        )

        # PROCESSES (MARKET)
        regulator = RegulatorNWEATR(
            history_market_parser=history_market_parser_1m,
            nwe_bounds_indicator=nwe_bounds_indicator,
            atr_bounds_indicator=atr_bounds_indicator,
            strategy=strategy,
            fee_rate=0,
            risk_usdt=10,
            min_profit_usdt=15
        )

        buyer = BuyerTPSL(
            symbol1=symbol1,
            symbol2=symbol2,
            history_market_parser=history_market_parser_1m,
            regulator_tpsl=regulator,
            balance_usdt=balance_usdt,
            fee=0
        )

        self.market_processes = [
            history_market_parser_1m,
            history_market_parser_15m,
            history_market_parser_60m,
            kama_indicator_60m,
            kama_indicator_15m,
            choch_indicator,
            atr_bounds_indicator,
            nwe_bounds_indicator,
            strategy,
            regulator,
            buyer
        ]