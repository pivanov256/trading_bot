import logging
import time
from venv import logger

from app.brokers import Broker
from app.data_fetchers import DataFetcher
from app.predictors import Predictor, TradeAction

logger = logging.getLogger(__name__)


class Trader():

    def __init__(self, data_fetcher: DataFetcher, predictor: Predictor, broker: Broker):
        self.run = True
        self.data_fetcher = data_fetcher
        self.predictor = predictor
        self.broker = broker

    def trade(self):
        while self.run:
            current_data = self.data_fetcher.get_data()
            trade_action = self.predictor.predict(current_data)
            logger.info(f"Trade action: {trade_action}")
            if trade_action != TradeAction.NO_ACTION:
                self.broker.place_order(trade_action, self.predictor.calculate_take_profit(current_data, trade_action),
                                        self.predictor.calculate_stop_loss(current_data, trade_action))

            time.sleep(60)
