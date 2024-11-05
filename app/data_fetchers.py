from abc import abstractmethod

import pandas as pd
from ibapi.client import *
from ibapi.wrapper import *
from pandas import DataFrame

from app.interactive_brokers_client import IBClient

logger = logging.getLogger(__name__)


class DataFetcher():

    def __init__(self, source: IBClient):
        self.source = source

    @abstractmethod
    def get_data() -> DataFrame:
        pass


class EurUsd1MinDataFetcher(DataFetcher):

    def __init__(self, source: IBClient):
        super().__init__(source)
        self.contract = self._create_contract()

    def get_data(self) -> DataFrame:
        logger.info("Requesting data")
        self.source.reqHistoricalData(self.source.nextId(), self.contract, "", "1 D", "1 min", "MIDPOINT", 1, 1, False,
                                      [])
        self.source.wait_for_requested_data()
        data = self.source.get_requested_data()
        df_data = []
        [
            df_data.append({
                'date': row.date,
                'open': row.open,
                'high': row.high,
                'low': row.low,
                'close': row.close,
                'volume': row.volume,
                'barCount': row.barCount}) for row in data]
        logger.info("Data received")

        return DataFrame(df_data)

    def _create_contract(self):
        contract = Contract()
        contract.symbol = "EUR"
        contract.secType = "CASH"
        contract.exchange = "IDEALPRO"
        contract.currency = "USD"
        return contract


class LocalDataFetcher(DataFetcher):

    def __init__(self, source: IBClient):
        super().__init__(source)
        self._data = pd.read_parquet('./playground/EURUSD_M1_tws_sell_signals.parquet')
        self.window_size = 10
        self.start_index = 0
        self._data = self._data[-100_000:]

    def get_data(self) -> DataFrame:
        logger.info("Requesting data")
        if self.start_index + self.window_size > len(self._data):
            raise IndexError("Sliding window has moved beyond the available data.")

        # Get the current window
        window_data = self._data.iloc[self.start_index:self.start_index + self.window_size]

        # Move the window one step forward
        self.start_index += 1

        return window_data
