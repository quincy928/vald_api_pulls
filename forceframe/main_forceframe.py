from vald_forceframe import Vald
from dotenv import load_dotenv
import os
import time
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

test = os.getenv("VALD_MASTER_FILE_PATH")

def main():

    logging.info('Starting...')
    
    vald = Vald()
    logging.info("Vald ForceFrame initialized")
    logging.info(test)
    vald.populate_folders()
    logging.info("Vald ForceFrame first populate")

    while True:
        interval = 300
        countdown = interval
        while countdown > 0:
            minutes, seconds = divmod(countdown, 60)
            logging.info(f"{minutes}m{seconds}s remaining ForceFrame")
            time.sleep(30)
            countdown -= 30
        vald.populate_folders()

if __name__ == "__main__":
    main()
