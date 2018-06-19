import logging
import time

import matplotlib.pyplot as plt
# import numpy as np
import pandas as pd
import os

if __name__ == '__main__':
    start_time = time.time()

    formatter = logging.Formatter('%(asctime)s : %(name)s :: %(levelname)s : %(message)s')
    logger = logging.getLogger('main')
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    console_handler.setLevel(logging.DEBUG)
    logger.debug('started')

    input_folder = '../data/'
    output_folder = '../output/'

    input_folder_exists = os.path.isdir(input_folder)
    if not input_folder_exists:
        logger.warning('input folder %s does not exist. Quitting.' % input_folder)
        quit()
    output_folder_exists = os.path.isdir(output_folder)
    if not output_folder_exists:
        logger.warning('output folder %s does not exist. Quitting.' % output_folder)
        quit()

    train_file = input_folder + 'Train.csv'
    train_df = pd.read_csv(train_file)
    logger.debug(train_df.shape)

    test_file = input_folder + 'Test.csv'
    test_df = pd.read_csv(test_file)
    logger.debug(test_df.shape)

    # df.Timestamp = pd.to_datetime(df.Datetime, format='%d-%m-%Y %H:%M')
    # df.index = df.Timestamp
    # df = df.resample('D').mean()
    train_df['Timestamp'] = pd.to_datetime(train_df.Datetime, format='%d-%m-%Y %H:%M')
    train_df.index = train_df.Timestamp
    train_df = train_df.resample('D').mean()
    test_df['Timestamp'] = pd.to_datetime(test_df.Datetime, format='%d-%m-%Y %H:%M')
    test_df.index = test_df.Timestamp
    test_df = test_df.resample('D').mean()
    train_df.Count.plot()
    plt.show()

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
