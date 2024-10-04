from vald_dynamo import Vald
from dotenv import load_dotenv
import os
import time
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

def main():

    logging.info('Starting...')
    vald = Vald()
    logging.info(str(vald.base_directory))
    logging.info("Vald Dynamo initialized")
    vald.populate_folders()
    logging.info("Vald Dynamo first populate")
    while True:
        interval = 300
        countdown = interval
        while countdown > 0:
            minutes, seconds = divmod(countdown, 60)
            logging.info(f"{minutes}m{seconds}s remaining Dynamo")
            time.sleep(30)
            countdown -= 30
        vald.populate_folders()

if __name__ == "__main__":
    main()
