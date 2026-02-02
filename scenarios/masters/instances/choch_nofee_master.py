from scenarios.market.brokers.mexc_broker import MEXCBroker
from scenarios.market.buyers.buyer_tpsl import BuyerTPSL
from scenarios.market.regulators.regulator_nweatr import RegulatorNWEATR
from scenarios.masters.abstracts.market_master import MarketMaster
from scenarios.parsers.history_market_parser.instances.history_binance_parser import HistoryBinanceParser
from scenarios.parsers.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.parsers.indicators.instances.choch_indicator import CHoCHIndicator
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.strategies.instances.choch_nofee_strategy import CHoCHNoFeeStrategy
import mexc_api


class CHoCHNoFeeMaster(MarketMaster):
    def __init__(self, symbol1, symbol2, balance_usdt, mode='generating'):
        super().__init__()

        # PROCESSES (PARSERS)
        history_market_parser_1m = HistoryBinanceParser(symbol1, symbol2, 1, 1000, mode)
        history_market_parser_15m = HistoryBinanceParser(symbol1, symbol2, 15, 1000, mode)
        history_market_parser_context = HistoryBinanceParser(symbol1, symbol2, 120, 1000, mode)
        nwe_bounds_indicator = NweBoundsIndicator(history_market_parser_1m)
        atr_bounds_indicator = AtrBoundsIndicator(history_market_parser_1m)
        kama_indicator_context = KamaIndicator(history_market_parser_context, 7, 2, 30)
        choch_indicator = CHoCHIndicator(history_market_parser_15m)

        # PROCESSES (STRATEGIES)
        strategy = CHoCHNoFeeStrategy(
            history_market_parser_1m=history_market_parser_1m,
            kama_indicator=kama_indicator_context,
            choch_indicator=choch_indicator,
            nwe_bounds_indicator=nwe_bounds_indicator,
        )

        # PROCESSES (MARKET)
        regulator = RegulatorNWEATR(
            history_market_parser=history_market_parser_1m,
            nwe_bounds_indicator=nwe_bounds_indicator,
            atr_bounds_indicator=atr_bounds_indicator,
            strategy=strategy,
            fee_rate=0,
            risk_usdt=0.21,
            min_profit_usdt=0.35
        )

        buyer = BuyerTPSL(
            symbol1=symbol1,
            symbol2=symbol2,
            history_market_parser=history_market_parser_1m,
            regulator_tpsl=regulator,
            balance_usdt=balance_usdt,
            fee=0,
            is_take_profit_for_close=False
        )

        broker = MEXCBroker(buyer, api_key=mexc_api.API_KEY, api_secret=mexc_api.API_SECRET)

        self.market_processes = [
            history_market_parser_1m,
            history_market_parser_15m,
            history_market_parser_context,
            kama_indicator_context,
            choch_indicator,
            atr_bounds_indicator,
            nwe_bounds_indicator,
            strategy,
            regulator,
            buyer,
            broker
        ]
