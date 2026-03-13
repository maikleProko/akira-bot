from scenarios.market.brokers.mexc_broker_taker import MEXCBrokerTaker
from scenarios.market.buyers.buyer_tpsl import BuyerTPSL
from scenarios.market.regulators.regulator_nweatr import RegulatorNWEATR
from scenarios.masters.abstracts.market_master import MarketMaster
from scenarios.parsers.history_market_parser.instances.history_binance_parser import HistoryBinanceParser
from scenarios.parsers.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.parsers.indicators.instances.bos_indicator import BosIndicator
from scenarios.parsers.indicators.instances.choch_indicator import CHoCHIndicator
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
from scenarios.parsers.indicators.instances.nwe_bounds_indicator import NweBoundsIndicator
import mexc_api
from scenarios.strategies.instances.choch_nofeeroyal_strategy import CHoCHNoFeeRoyalStrategy


class CHoCHNoFeeRoyalMaster(MarketMaster):
    def __init__(self, symbol1, symbol2, balance_usdt, mode='generating',
                 fee=0, risk_usdt=30, min_profit_usdt=50, is_used_broker=False):
        super().__init__()

        # PROCESSES (PARSERS)
        history_market_parser_1m = HistoryBinanceParser(symbol1, symbol2, 1, 1000, mode)
        history_market_parser_15m = HistoryBinanceParser(symbol1, symbol2, 15, 1000, mode)
        history_market_parser_30m = HistoryBinanceParser(symbol1, symbol2, 30, 1000, mode)
        history_market_parser_240m = HistoryBinanceParser(symbol1, symbol2, 120, 1000, mode)
        nwe_bounds_indicator = NweBoundsIndicator(history_market_parser_1m)
        atr_bounds_indicator = AtrBoundsIndicator(history_market_parser_1m)
        kama_indicator_30m = KamaIndicator(history_market_parser_30m, 7, 2, 30)
        kama_indicator_240m = KamaIndicator(history_market_parser_240m, 7, 2, 30)
        choch_indicator_15m = CHoCHIndicator(history_market_parser_15m)
        choch_indicator_2h = CHoCHIndicator(history_market_parser_240m)
        bos_indicator_2h = BosIndicator(history_market_parser_240m)

        # PROCESSES (STRATEGIES)
        strategy = CHoCHNoFeeRoyalStrategy(
            history_market_parser_1m=history_market_parser_1m,
            kama_indicator_30m=kama_indicator_30m,
            kama_indicator_240m=kama_indicator_240m,
            choch_indicator_15m=choch_indicator_15m,
            choch_indicator_2h=choch_indicator_2h,
            bos_indicator_2h=bos_indicator_2h,
            nwe_bounds_indicator=nwe_bounds_indicator,
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

        buyer = BuyerTPSL(
            symbol1=symbol1,
            symbol2=symbol2,
            history_market_parser=history_market_parser_1m,
            regulator_tpsl=regulator,
            balance_usdt=balance_usdt,
            fee=fee,
            is_take_profit_for_close=False
        )

        broker = MEXCBrokerTaker(buyer, api_key=mexc_api.API_KEY, api_secret=mexc_api.API_SECRET)

        self.market_processes = [
            history_market_parser_1m,
            history_market_parser_15m,
            history_market_parser_30m,
            history_market_parser_240m,
            kama_indicator_30m,
            kama_indicator_240m,
            choch_indicator_15m,
            choch_indicator_2h,
            bos_indicator_2h,
            atr_bounds_indicator,
            nwe_bounds_indicator,
            strategy,
            regulator,
            buyer,
            *([ broker ] if is_used_broker else [])
        ]
