import logging
import os
import time
from math import sqrt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.api import ExponentialSmoothing
from statsmodels.tsa.api import Holt
from statsmodels.tsa.api import SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

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

    train_fraction = 0.75
    split_point = int(train_fraction * df.shape[0])
    logger.debug('we are training on %.3f of the data, or %d rows.' % (train_fraction, split_point))

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

    rms_naive = sqrt(mean_squared_error(test_df.Count, y_hat.naive))
    logger.debug('naive model root-mean-squared error: %.3f' % rms_naive)

    y_hat_avg = test_df.copy()
    y_hat_avg['avg_forecast'] = train_df['Count'].mean()
    plt.figure()
    plt.plot(train_df['Count'], label='Train')
    plt.plot(test_df['Count'], label='Test')
    plt.plot(y_hat_avg['avg_forecast'], label='Average Forecast')
    plt.legend(loc='best')
    plt.savefig(output_folder + 'average_model.png')
    plt.close()

    rms_average = sqrt(mean_squared_error(test_df.Count, y_hat_avg.avg_forecast))
    logger.debug('average model root-mean-squared error: %.3f' % rms_average)

    y_hat_avg = test_df.copy()
    y_hat_avg['moving_avg_forecast'] = train_df['Count'].rolling(60).mean().iloc[-1]
    plt.figure()
    plt.plot(train_df['Count'], label='Train')
    plt.plot(test_df['Count'], label='Test')
    plt.plot(y_hat_avg['moving_avg_forecast'], label='Moving Average Forecast')
    plt.legend(loc='best')
    plt.savefig(output_folder + 'moving_average.png')
    plt.close()

    rms_moving_average = sqrt(mean_squared_error(test_df.Count, y_hat_avg.moving_avg_forecast))
    logger.debug('moving average model root-mean-squared error: %.3f' % rms_moving_average)

    y_hat_avg = test_df.copy()
    fit2 = SimpleExpSmoothing(np.asarray(train_df['Count'])).fit(smoothing_level=0.6, optimized=False)
    y_hat_avg['SES'] = fit2.forecast(len(test_df))
    plt.figure()
    plt.plot(train_df['Count'], label='Train')
    plt.plot(test_df['Count'], label='Test')
    plt.plot(y_hat_avg['SES'], label='SES')
    plt.legend(loc='best')
    plt.savefig(output_folder + 'simple_exponential_smoothing.png')
    plt.close()

    rms_ses = sqrt(mean_squared_error(test_df.Count, y_hat_avg.SES))
    logger.debug('simple exponential smoothing model root-mean-squared error: %.3f' % rms_moving_average)

    # todo figure out what's wrong here
    y_hat_avg = test_df.copy()
    fit1 = Holt(np.asarray(train_df['Count'])).fit(smoothing_level=0.3, smoothing_slope=0.1)
    y_hat_avg['Holt_linear'] = fit1.forecast(len(test_df))
    plt.figure()
    plt.plot(train_df['Count'], label='Train')
    plt.plot(test_df['Count'], label='Test')
    plt.plot(y_hat_avg['Holt_linear'], label='Holt_linear')
    plt.legend(loc='best')
    plt.savefig(output_folder + 'holt-model.png')
    plt.close()

    rms_holt = sqrt(mean_squared_error(test_df.Count, y_hat_avg.Holt_linear))
    logger.debug('holt model root-mean-squared error: %.3f' % rms_holt)

    y_hat_avg = test_df.copy()
    seasonal_periods = 7
    trend = 'add'
    seasonal = 'add'
    fit1 = ExponentialSmoothing(np.asarray(train_df['Count']), seasonal_periods=seasonal_periods, trend=trend,
                                seasonal=seasonal).fit()
    y_hat_avg['Holt_Winter'] = fit1.forecast(len(test_df))
    plt.figure()
    plt.plot(train_df['Count'], label='Train')
    plt.plot(test_df['Count'], label='Test')
    plt.plot(y_hat_avg['Holt_Winter'], label='Holt_Winter')
    plt.legend(loc='best')
    plt.savefig(output_folder + 'holt_winter.png')
    plt.close()

    rms_holt_winter = sqrt(mean_squared_error(test_df.Count, y_hat_avg.Holt_Winter))
    logger.debug('holt-winter model root-mean-squared error: %.3f' % rms_holt_winter)

    y_hat_avg = test_df.copy()
    fit1 = SARIMAX(train_df.Count, order=(2, 1, 4), seasonal_order=(0, 1, 1, 7)).fit()

    y_hat_avg['SARIMA'] = fit1.predict(start=test_df.index[0], end=test_df.index[-1], dynamic=True)
    plt.figure(figsize=(12, 8))
    plt.plot(train_df['Count'], label='Train')
    plt.plot(test_df['Count'], label='Test')
    plt.plot(y_hat_avg['SARIMA'], label='SARIMA')
    plt.legend(loc='best')
    plt.savefig(output_folder + 'sarimax.png')
    plt.close()

    rms_sarimax = sqrt(mean_squared_error(test_df.Count, y_hat_avg.SARIMA))
    logger.debug('SARIMAX root-mean-squared error: %.3f' % rms_sarimax)

    logger.debug('done')
    finish_time = time.time()
    elapsed_hours, elapsed_remainder = divmod(finish_time - start_time, 3600)
    elapsed_minutes, elapsed_seconds = divmod(elapsed_remainder, 60)
    logger.info("Time: {:0>2}:{:0>2}:{:05.2f}".format(int(elapsed_hours), int(elapsed_minutes), elapsed_seconds))
    console_handler.close()
    logger.removeHandler(console_handler)
