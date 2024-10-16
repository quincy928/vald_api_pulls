from vald_forcedecks import Vald
from dotenv import load_dotenv
import os
import time
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

def main():

    logging.info('Starting...')
    
    vald = Vald()
    vald.get_weight_table()
    if os.path.exists(vald.vald_master_file_path) == False:
            print("Setting up intial")
            vald.retrieve_tests_until_today('2023-08-01T00:00:00Z')
    while True:
        interval = 300
        countdown = interval
        while countdown > 0:
            minutes, seconds = divmod(countdown, 60)
            logging.info(f"{minutes}m{seconds}s remaining ForceDecks")
            time.sleep(30)
            countdown -= 30
        vald.update_forcedecks()
        vald.get_weight_table()

if __name__ == "__main__":
    main()
