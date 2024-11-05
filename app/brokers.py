from abc import abstractmethod

from ibapi.client import *
from ibapi.wrapper import *

from app.interactive_brokers_client import IBClient
from app.predictors import TradeAction

logger = logging.getLogger(__name__)


class Broker():

    @abstractmethod
    def place_order(self, trade_action: TradeAction, take_profit_price: Decimal, stop_loss_price: Decimal):
        pass


class FakeBroker(Broker):

    def place_order(self, trade_action: TradeAction, take_profit_price: Decimal, stop_loss_price: Decimal):
        logger.info(f"Placing order: {trade_action} - TP: {take_profit_price} - SL: {stop_loss_price}")


class EurUsd1mBroker(Broker):

    def __init__(self, platform: IBClient):
        self.platform = platform
        self.contract = self._create_contract()
        self.order_quantity = 10_000

    def place_order(self, trade_action: TradeAction, take_profit_price: Decimal, stop_loss_price: Decimal):
        logger.info(f"Placing order: {trade_action} - TP: {take_profit_price} - SL: {stop_loss_price}")
        bracket_order = self.platform.create_bracket_order(self.platform.nextId(), trade_action, self.order_quantity,
                                                           take_profit_price, stop_loss_price)
        for o in bracket_order:
            self.platform.placeOrder(o.orderId, self.contract, o)

    def _create_contract(self):
        contract = Contract()
        contract.symbol = "EUR"
        contract.secType = "CFD"
        contract.exchange = "SMART"
        contract.currency = "USD"
        return contract
