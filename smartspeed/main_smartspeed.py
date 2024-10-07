from vald_smartspeed import Vald
from dotenv import load_dotenv
import os
import time
import logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

def main():

    logging.info('Starting...')
    vald = Vald()
    #Edit this as you see fit. The program will retrieve all smart speed records from after this date    
    start_date = '2023-08-01T00:00:00Z'
    if os.path.exists(vald.vald_master_file_path) == False:
            print("Setting up intial")
            vald.get_data_until_today(start_date)
    logging.info(str(vald.base_directory))
    while True:
        interval = 300
        countdown = interval
        while countdown > 0:
            minutes, seconds = divmod(countdown, 60)
            logging.info(f"{minutes}m{seconds}s remaining")
            time.sleep(30)
            countdown -= 30
        vald.update_smartspeed

if __name__ == "__main__":
    main()
