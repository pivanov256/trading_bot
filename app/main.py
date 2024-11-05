import logging
import threading
import time
from logging.handlers import TimedRotatingFileHandler

from app.brokers import EurUsd1mBroker
from app.data_fetchers import EurUsd1MinDataFetcher
from app.interactive_brokers_client import IBClient
from app.predictors import EurUsd1MinPredictor
from app.trader import Trader
from app.utils.constants import LOCALHOST, TWS_DEMO_PORT

logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S', handlers=[
        TimedRotatingFileHandler('./app/logs/app.log', when="W0", interval=1, backupCount=4),
        logging.StreamHandler()])

logger = logging.getLogger(__name__)


def main():
    logger.info("Starting the trader")
    logger.info("Connecting to the Interactive Brokers API")
    ib_client = IBClient()
    ib_client.connect(LOCALHOST, TWS_DEMO_PORT, 0)
    threading.Thread(target=lambda: (ib_client.run(), time.sleep(1))).start()
    logger.info("Connected to the Interactive Brokers API")
    try:
        trader = Trader(data_fetcher=EurUsd1MinDataFetcher(ib_client), predictor=EurUsd1MinPredictor(),
                        broker=EurUsd1mBroker(ib_client))
        trader.trade()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    finally:
        ib_client.disconnect()
        logger.info("Disconnected from the Interactive Brokers API")


if __name__ == "__main__":
    main()
