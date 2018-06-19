import logging
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

from sklearn.metrics import mean_squared_error
from math import sqrt

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
    df = pd.read_csv(train_file)
    logger.debug('the original training dataset has shape %d x %d' % df.shape)

    df['Timestamp'] = pd.to_datetime(df.Datetime, format='%d-%m-%Y %H:%M')
    df.index = df.Timestamp
    df = df.resample('D').mean()
    logger.debug('the daily-sampled training dataset has shape %d x %d ' % df.shape)

    split_point = 3 * df.shape[0] // 4
    logger.debug('split point is %d' % split_point)

    train_df = df[:split_point]
    test_df = df[split_point:]

    train_df.Count.plot()
    test_df.Count.plot()
    plt.savefig(output_folder + 'train-test-plot.png')
    plt.close()

    naive_data = np.asarray(train_df.Count)
    y_hat = test_df.copy()
    y_hat['naive'] = naive_data[len(naive_data) - 1]
    plt.figure()
    plt.plot(train_df.index, train_df['Count'], label='Train')
    plt.plot(test_df.index, test_df['Count'], label='Test')
    plt.plot(y_hat.index, y_hat['naive'], label='Naive Forecast')
    plt.legend(loc='best')
    plt.title("Naive Forecast")
    plt.savefig(output_folder + 'naive_model.png')
    plt.close()

    rms = sqrt(mean_squared_error(test_df.Count, y_hat.naive))
    logger.debug('naive model root-mean-squared error: %.3f' % rms)

    y_hat_avg = test_df.copy()
    y_hat_avg['avg_forecast'] = train_df['Count'].mean()
    plt.figure()
    plt.plot(train_df['Count'], label='Train')
    plt.plot(test_df['Count'], label='Test')
    plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')
    plt.legend(loc='best')
    plt.show()

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
