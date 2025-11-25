
from scenarios.parsers.arbitrage_parser.rolling_arbitrage_parser.core.abstract_arbitrage_parser import \
    AbstractArbitrageParser


class RollingArbitrageParser(AbstractArbitrageParser):

    def run_realtime(self):
        out_edges, symbol_map, price_map = self.build_graph_and_prices()
        print(list(out_edges))
        ops = self.find_arbitrage_cycles(out_edges, symbol_map, price_map, self.fee_rate, self.min_profit, self.start_amount, self.max_cycles, self.max_cycle_len, self.max_profit, [])
        if ops > 0:
            print(ops[0])
        #best_op = self.calculate_best_op(ops, self.strict)
        #if len(self.consecutive_same) > 0  and self.possible:
            #self.start_trade(best_op, out_edges, symbol_map, price_map)



