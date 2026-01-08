from scenarios.market.buyers.buyer_tpsl import BuyerTPSL
from scenarios.market.regulators.regulator_take_profit import RegulatorTakeProfit
from scenarios.masters.abstracts.market_master import MarketMaster
from scenarios.parsers.history_market_parser.instances.history_binance_parser import HistoryBinanceParser
from scenarios.parsers.indicators.instances.orderblock_indicator import OrderblockIndicator
from scenarios.parsers.indicators.instances.atr_bounds_indicator import AtrBoundsIndicator
from scenarios.parsers.indicators.instances.kama_indicator import KamaIndicator
from scenarios.strategies.instances.obflow_strategy import ObFlowStrategy


class ObFlowMaster(MarketMaster):
    def __init__(self, symbol1, symbol2, balance_usdt, mode='generating'):
        super().__init__()

        # PROCESSES (PARSERS)
        history_market_parser_1m = HistoryBinanceParser(symbol1, symbol2, 1, 1000, mode)
        history_market_parser_240m = HistoryBinanceParser(symbol1, symbol2, 240, 1000, mode)
        kama_indicator_240m = KamaIndicator(history_market_parser_240m, 7, 2, 30)
        atr_bounds_indicator = AtrBoundsIndicator(history_market_parser_1m)
        orderblock_indicator = OrderblockIndicator(history_market_parser_1m)

        # PROCESSES (STRATEGIES)
        strategy = ObFlowStrategy(
            history_market_parser_1m=history_market_parser_1m,
            kama_indicator_240m=kama_indicator_240m,
            orderblock_indicator=orderblock_indicator,
            atr_bounds_indicator=atr_bounds_indicator
        )

        # PROCESSES (MARKET)
        regulator = RegulatorTakeProfit(
            history_market_parser=history_market_parser_1m,
            take_profit_indicator=strategy,
            strategy=strategy,
        )

        buyer = BuyerTPSL(
            symbol1=symbol1,
            symbol2=symbol2,
            history_market_parser=history_market_parser_1m,
            regulator_tpsl=regulator,
            balance_usdt=balance_usdt
        )

        self.market_processes = [
            history_market_parser_1m,
            history_market_parser_240m,
            orderblock_indicator,
            kama_indicator_240m,
            atr_bounds_indicator,
            strategy,
            regulator,
            buyer
        ]