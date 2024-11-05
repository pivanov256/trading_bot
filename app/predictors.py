import logging
from abc import abstractmethod
from decimal import ROUND_HALF_UP, Decimal
from enum import Enum

import joblib
import numpy as np
import torch
from pandas import DataFrame

from app.models import ForexCNNLSTM, ForexLSTM

logger = logging.getLogger(__name__)


class TradeAction(Enum):
    SELL = "SELL"
    BUY = "BUY"
    NO_ACTION = "NO_ACTION"


class Predictor():

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @abstractmethod
    def predict(self, data: DataFrame):
        pass

    @abstractmethod
    def calculate_stop_loss(self, data: DataFrame, trade_action: TradeAction):
        pass

    @abstractmethod
    def calculate_take_profit(self, data: DataFrame, trade_action: TradeAction):
        pass


class EurUsd1MinPredictor(Predictor):

    def __init__(self):
        super().__init__()
        self.buy_scaler = joblib.load('./app/utils/data/eurusd_1m_scaler_500k.pkl')
        self.sell_scaler = joblib.load('./app/utils/data/eurusd_1m_scaler_500k.pkl')
        self.buy_model = self._load_buy_model()
        self.sell_model = self._load_sell_model()
        self.window_size = 10
        self.profit_loss_ration = 4

    def predict(self, data: DataFrame):
        logger.info("Predicting trade action")
        data = self._preprocess_data(data)
        logger.info(f"Data: {data}")
        data = data[['open', 'high', 'low', 'close']]
        window_normalized_buy = self.buy_scaler.transform(data.to_numpy())
        window_tensor_buy = torch.tensor(window_normalized_buy, dtype=torch.float32).unsqueeze(0)
        window_normalized_sell = self.sell_scaler.transform(data.to_numpy())
        window_tensor_sell = torch.tensor(window_normalized_sell, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            buy_prediction = self.buy_model(window_tensor_buy)
            _, buy_predicted_class = torch.max(buy_prediction, 1)

        with torch.no_grad():
            sell_prediction = self.sell_model(window_tensor_sell)
            _, sell_predicted_class = torch.max(sell_prediction, 1)

        logger.info(f"Buy prediction: {buy_predicted_class.item()}")
        logger.info(f"Sell prediction: {sell_predicted_class.item()}")

        return self._decide_trade_action(sell_predicted_class, buy_predicted_class)

    def calculate_stop_loss(self, data: DataFrame, trade_action: TradeAction):
        data = self._preprocess_data(data)
        if trade_action == TradeAction.BUY:
            return self._round(data['low'].min())
        elif trade_action == TradeAction.SELL:
            return self._round(data['high'].max())

    def calculate_take_profit(self, data: DataFrame, trade_action: TradeAction):
        data = self._preprocess_data(data)
        if trade_action == TradeAction.BUY:
            return self._round(data.iloc[-1]['open'] +
                               (data.iloc[-1]['open'] - data['low'].min()) * self.profit_loss_ration)
        elif trade_action == TradeAction.SELL:
            return self._round(data.iloc[-1]['open'] -
                               (data['high'].max() - data.iloc[-1]['open']) * self.profit_loss_ration)

    def _decide_trade_action(self, sell_predicted_class, buy_predicted_class):
        if (sell_predicted_class.item() == 1 and buy_predicted_class.item() == 1) or \
        (sell_predicted_class.item() == 0 and buy_predicted_class.item() == 0):
            return TradeAction.NO_ACTION
        elif sell_predicted_class.item() == 1 and buy_predicted_class.item() == 0:
            return TradeAction.SELL
        elif sell_predicted_class.item() == 0 and buy_predicted_class.item() == 1:
            return TradeAction.BUY

    def _round(self, value) -> Decimal:
        return Decimal(value).quantize(Decimal('1.00000'), rounding=ROUND_HALF_UP)

    def _preprocess_data(self, data: DataFrame) -> DataFrame:
        if len(data) < self.window_size:
            data = DataFrame(np.zeros((10, 4)), columns=['open', 'high', 'low', 'close'])
        return data.iloc[-self.window_size - 1:-1]

    def _load_buy_model(self):
        logger.info("Loading buy model")
        input_size = 4
        hidden_size = 64
        num_layers = 2
        num_classes = 2
        model = ForexLSTM(input_size, hidden_size, num_layers, num_classes)
        model.load_state_dict(
            torch.load('./app/utils/data/eurusd_1m_lstm_500k_6epoch_buy.pth', map_location=self.device))
        model.eval()

        return model

    def _load_sell_model(self):
        logger.info("Loading sell model")
        input_size = 4
        hidden_size = 64
        num_layers = 2
        num_classes = 2
        model = ForexLSTM(input_size, hidden_size, num_layers, num_classes)
        model.load_state_dict(
            torch.load('./app/utils/data/eurusd_1m_lstm_500k_6epoch_sell.pth', map_location=self.device))
        model.eval()

        return model
