from scenarios.market.brokers.mexc_broker_taker import MEXCBrokerTaker
from scenarios.market.buyers.buyer_tpsl_kama_exit import BuyerTPSLKamaExit
from scenarios.market.regulators.regulator_nweatr import RegulatorNWEATR
from scenarios.masters.abstracts.market_master import MarketMaster
from scenarios.parsers.history_market_parser.instances.history_binance_parser import HistoryBinanceParser
from scenarios.parsers.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.parsers.indicators.instances.bos_indicator import BosIndicator
from scenarios.parsers.indicators.instances.choch_indicator import CHoCHIndicator
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
from scenarios.strategies.instances.choch_kimmi_strategy import CHoCHKimmiStrategy
import mexc_api


class CHoCHKimmiMaster(MarketMaster):
    def __init__(self, symbol1, symbol2, balance_usdt, mode='generating',
                 fee=0, risk_usdt=30, min_profit_usdt=50, is_used_broker=False):
        super().__init__()

        # PROCESSES (PARSERS)
        history_market_parser_1m = HistoryBinanceParser(symbol1, symbol2, 1, 1000, mode)
        history_market_parser_15m = HistoryBinanceParser(symbol1, symbol2, 15, 1000, mode)
        history_market_parser_120m = HistoryBinanceParser(symbol1, symbol2, 120, 1000, mode)

        # PROCESSES (INDICATORS)
        kama_indicator_15m = KamaIndicator(history_market_parser_15m, 7, 2, 30)
        kama_indicator_120m = KamaIndicator(history_market_parser_120m, 7, 2, 30)
        choch_indicator_1m = CHoCHIndicator(history_market_parser_1m)
        bos_indicator_1m = BosIndicator(history_market_parser_1m)
        nwe_bounds_indicator = NweBoundsIndicator(history_market_parser_1m)
        atr_bounds_indicator = AtrBoundsIndicator(history_market_parser_1m)

        # PROCESSES (STRATEGIES)
        strategy = CHoCHKimmiStrategy(
            history_market_parser_1m=history_market_parser_1m,
            kama_indicator_15m=kama_indicator_15m,
            kama_indicator_120m=kama_indicator_120m,
            choch_indicator_1m=choch_indicator_1m,
            bos_indicator_1m=bos_indicator_1m,
        )

        # PROCESSES (MARKET)
        regulator = RegulatorNWEATR(
            history_market_parser=history_market_parser_1m,
            nwe_bounds_indicator=nwe_bounds_indicator,
            atr_bounds_indicator=atr_bounds_indicator,
            strategy=strategy,
            fee_rate=fee,
            risk_usdt=risk_usdt,
            min_profit_usdt=min_profit_usdt
        )

        buyer = BuyerTPSLKamaExit(
            symbol1=symbol1,
            symbol2=symbol2,
            history_market_parser=history_market_parser_1m,
            regulator_tpsl=regulator,
            kama_indicator_exit=kama_indicator_15m,
            balance_usdt=balance_usdt,
            fee=fee,
            is_take_profit_for_close=False
        )

        broker = MEXCBrokerTaker(buyer, api_key=mexc_api.API_KEY, api_secret=mexc_api.API_SECRET)

        self.market_processes = [
            history_market_parser_1m,
            history_market_parser_15m,
            history_market_parser_120m,
            kama_indicator_15m,
            kama_indicator_120m,
            choch_indicator_1m,
            bos_indicator_1m,
            atr_bounds_indicator,
            nwe_bounds_indicator,
            strategy,
            regulator,
            buyer,
            *([ broker ] if is_used_broker else [])
        ]
