import threading

from ibapi.client import *
from ibapi.wrapper import *

from app.predictors import TradeAction


class IBClient(EClient, EWrapper):

    def __init__(self):
        EClient.__init__(self, self)
        self.requested_bar_data = []
        self.data_ready_event = threading.Event()

    def nextValidId(self, orderId: OrderId):
        self.orderId = orderId

    def nextId(self):
        self.orderId += 1
        return self.orderId

    def error(self, reqId, errorCode, errorString, advancedOrderReject):
        print(f"reqId: {reqId}, errorCode: {errorCode}, errorString: {errorString}, orderReject: {advancedOrderReject}")

    def historicalData(self, reqId, bar):
        self.requested_bar_data.append(bar)

    def historicalDataEnd(self, reqId, start, end):
        self.cancelHistoricalData(reqId)
        self.data_ready_event.set()

    def openOrder(self, orderId: OrderId, contract: Contract, order: Order, orderState: OrderState):
        print(
            f"openOrder: {orderId}, contract: {contract}, order: {order}, Maintenance Margin: {orderState.maintMarginChange}"
        )

    def orderStatus(self, orderId: OrderId, status: str, filled: float, remaining: float, avgFillPrice: float,
                    permId: int, parentId: int, lastFillPrice: float, clientId: int, whyHeld: str, mktCapPrice: float):
        print(
            f"orderStatus. orderId: {orderId}, status:  {status}, filled: {filled}, remaining: {remaining}, avgFillPrice: {avgFillPrice}, permId: {permId}, parentId: {parentId}, lastFillPrice: {lastFillPrice}, clientId: {clientId}, whyHeld: {whyHeld}, mktCapPrice: {mktCapPrice}"
        )

    def execDetails(self, reqId: int, contract: Contract, execution: Execution):
        print(f"execDetails. reqId: {reqId}, contract: {contract}, execution:  {execution}")

    def create_bracket_order(self, parentOrderId: int, action: TradeAction, quantity: Decimal,
                             takeProfitLimitPrice: float, stopLossPrice: float):
        #This will be our main or "parent" order
        parent = Order()
        parent.orderId = parentOrderId
        parent.action = action.value
        parent.orderType = "MKT"
        parent.totalQuantity = quantity
        # parent.lmtPrice = limitPrice
        #The parent and children orders will need this attribute set to False to prevent accidental executions.
        #The LAST CHILD will have it set to True,
        parent.transmit = False
        parent.tif = 'GTC'
        parent.outsideRth = True

        takeProfit = Order()
        takeProfit.orderId = self.nextId()
        takeProfit.action = "SELL" if action == TradeAction.BUY else "BUY"
        takeProfit.orderType = "LMT"
        takeProfit.totalQuantity = quantity
        takeProfit.lmtPrice = takeProfitLimitPrice
        takeProfit.parentId = parentOrderId
        takeProfit.transmit = False
        takeProfit.tif = 'GTC'
        takeProfit.outsideRth = True

        stopLoss = Order()
        stopLoss.orderId = self.nextId()
        stopLoss.action = "SELL" if action == TradeAction.BUY else "BUY"
        stopLoss.orderType = "STP"
        #Stop trigger price
        stopLoss.auxPrice = stopLossPrice
        stopLoss.totalQuantity = quantity
        stopLoss.parentId = parentOrderId
        #In this case, the low side order will be the last child being sent. Therefore, it needs to set this attribute to True
        #to activate all its predecessors
        stopLoss.transmit = True
        stopLoss.tif = 'GTC'
        stopLoss.outsideRth = True

        bracketOrder = [parent, takeProfit, stopLoss]
        return bracketOrder

    def get_requested_data(self):
        res = self.requested_bar_data
        self.requested_bar_data = []
        return res

    def wait_for_requested_data(self):
        self.data_ready_event.wait()
        self.data_ready_event.clear()
