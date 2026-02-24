from scenarios.market.brokers.mexc_broker_maker import MEXCBrokerMaker
from scenarios.market.brokers.mexc_broker_taker import MEXCBrokerTaker
from scenarios.market.buyers.buyer_tpsl import BuyerTPSL
from scenarios.market.regulators.regulator_nweatr import RegulatorNWEATR
from scenarios.masters.abstracts.market_master import MarketMaster
from scenarios.parsers.history_market_parser.instances.history_binance_parser import HistoryBinanceParser
from scenarios.parsers.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.strategies.abstracts.strategy import Strategy
import mexc_api


class NoFeeMaster(MarketMaster):
    def __init__(self, symbol1, symbol2, balance_usdt, mode='generating'):
        super().__init__()

        # PROCESSES (PARSERS)
        history_market_parser_1m = HistoryBinanceParser(symbol1, symbol2, 1, 1000, mode)
        nwe_bounds_indicator = NweBoundsIndicator(history_market_parser_1m)
        atr_bounds_indicator = AtrBoundsIndicator(history_market_parser_1m)

        # PROCESSES (STRATEGIES)
        strategy = Strategy()

        # PROCESSES (MARKET)
        regulator = RegulatorNWEATR(
            history_market_parser=history_market_parser_1m,
            nwe_bounds_indicator=nwe_bounds_indicator,
            atr_bounds_indicator=atr_bounds_indicator,
            strategy=strategy,
            fee_rate=0,
            risk_usdt=0.1,
            min_profit_usdt=0.2
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

        broker = MEXCBrokerTaker(buyer, api_key=mexc_api.API_KEY, api_secret=mexc_api.API_SECRET)

        self.market_processes = [
            history_market_parser_1m,
            atr_bounds_indicator,
            nwe_bounds_indicator,
            strategy,
            regulator,
            buyer,
            broker
        ]
