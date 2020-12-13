# LSTM-Autoencoder based Anomaly Detection (LAAD)

######################################################
# Author: Gireesh K. Bogu                            #
# Email: gbogu17@stanford.edu                        #
# Location: Dept.of Genetics, Stanford University    #
# Date: Nov 20 2020                                   #
######################################################

#python laad.py  --heart_rate COVID-19-Wearables/ASFODQR_hr.csv --steps COVID-19-Wearables/ASFODQR_steps.csv --myphd_id ASFODQR --symptom_date 2024-08-14


import warnings
warnings.filterwarnings('ignore')
import sys 
import argparse
import copy
import numpy as np
import pandas as pd
import seaborn as sns
import itertools
from itertools import cycle
from tqdm import tqdm
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from scipy import interp
from arff2pandas import a2p
from datetime import date, datetime, timedelta
from statsmodels.tsa.seasonal import seasonal_decompose

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model, save_model

from pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
palette = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(palette))
rcParams['figure.figsize'] = 12, 8


# as command prompts -----------------------

parser = argparse.ArgumentParser(description='Find anomalies in wearables time-series data')
parser.add_argument('--heart_rate', metavar='', help ='raw heart rate count with a header = heartrate')
parser.add_argument('--steps',metavar='', help ='raw steps count with a header = steps')
parser.add_argument('--myphd_id',metavar='', default = 'myphd_id', help ='user myphd_id')
#parser.add_argument('--symptom_date', metavar='', default = 'NaN', help = 'symptom date with y-m-d format')
parser.add_argument('--random_seed', metavar='', type=int, default=42, help='random seed')
args = parser.parse_args()

# as arguments -----------------------

fitbit_oldProtocol_hr = args.heart_rate
fitbit_oldProtocol_steps = args.steps
myphd_id = args.myphd_id
#symptom_date = args.symptom_date
RANDOM_SEED = args.random_seed

np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
random.seed(RANDOM_SEED)

 
# Hyper-parameters --------------------

TIME_STEPS = 8
EPOCHS = 1200 
BATCH_SIZE = 64
VALIDATION_SPLIT = 0.05
LEARNING_RATE = 0.0001

BASE_LINE_DAYS = 10


########################################################################

class LAAD:

    # infer resting heart rate ------------------------------------------------------

    def resting_heart_rate(self, heartrate, steps):
        """
        This function uses heart rate and steps data to infer resting heart rate.
        It filters the heart rate with steps that are zero and also 12 minutes ahead.
        """

        # heart rate data
        df_hr = pd.read_csv(fitbit_oldProtocol_hr)
        df_hr = df_hr.set_index('datetime')
        df_hr.index.name = None
        df_hr.index = pd.to_datetime(df_hr.index)

        # steps data
        df_steps = pd.read_csv(fitbit_oldProtocol_steps)
        df_steps = df_steps.set_index('datetime')
        df_steps.index.name = None
        df_steps.index = pd.to_datetime(df_steps.index)

        # merge dataframes
        #df_hr = df_hr.resample('1min').mean()
        #df_steps = df_steps.resample('1min').mean()

        df1 = pd.merge(df_hr, df_steps, left_index=True, right_index=True)
        df1 = df1.resample('1min').mean()
        df1 = df1.dropna()
        
        # define RHR as the HR measurements recorded when there were zero steps taken during a rolling time window of the preceding 12 minutes (including the current minute).
        df1['steps_window_12'] = df1['steps'].rolling(12).sum()
        df1 = df1.loc[(df1['steps_window_12'] == 0)]
        return df1


    # pre-processing ------------------------------------------------------

    def pre_processing(self, resting_heart_rate):
        """
        It takes resting heart rate data and applies moving averages to smooth the data and 
        aggregates to one hour by taking the avegare values
        """

        # smooth data
        df_nonas = df1.dropna()
        df1_rom = df_nonas.rolling(400).mean()
        # resample
        df1_resmp = df1_rom.resample('1H').mean()
        df2 = df1_resmp.drop(['steps'], axis=1)
        df2 = df2.drop(['steps_window_12'], axis=1)
        #df2 = df2.resample('24H').mean()
        df2 = df2.dropna()
        df2 = df2.rename(columns={"heartrate": "RHR"})
        return df2


   # data splitting ------------------------------------------------------

    def data_splitting(self, processed_data):
        """
        It splits data into training data by taking first 20 days and the rest as testing data.
        It also creates windows of pre- and post-symptomatic COVID-periods.
        """

        train = processed_data[:BASE_LINE_DAYS]
        processed_data = processed_data.reset_index()
        processed_data['date'] = [d.date() for d in processed_data['index']]
        processed_data['time'] = [d.time() for d in processed_data['index']]
        processed_data = processed_data.set_index('date')
        processed_data.index.name = None
        processed_data.index = pd.to_datetime(processed_data.index)

        # split data into train
        start1 = processed_data.index[0] + timedelta(days=BASE_LINE_DAYS)
        train = processed_data[(processed_data.index.get_level_values(0) < start)]
        train = train.set_index('index')
        train = train.drop(['time'], axis=1)

        processed_data = processed_data.reset_index()
        processed_data['date'] = [d.date() for d in processed_data['index']]
        processed_data['time'] = [d.time() for d in processed_data['index']]
        processed_data = processed_data.set_index('date')
        processed_data.index.name = None
        processed_data.index = pd.to_datetime(processed_data.index)
        start = processed_data.index[0] + timedelta(days=BASE_LINE_DAYS)
        test = processed_data[(processed_data.index.get_level_values(0) >= start)]
        test = test.set_index('index')
        test = test.drop(['time'], axis=1)
        end = processed_data.index[-1]

        # create a random symptom date using test data
        r_start = processed_data.index[0] + timedelta(days=23)
        r_test = processed_data[(processed_data.index.get_level_values(0) >= r_start)]
        r_test = r_test.set_index('index')
        r_test = r_test.drop(['time'], axis=1)

        sdate = r_test.sample(n=1, random_state=RANDOM_SEED)
        symptom_date = sdate.index.format()
        symptom_date = ''.join(symptom_date)
        symptom_date1 = pd.to_datetime(symptom_date)        
        symptom_date_before_7 = pd.to_datetime(symptom_date1) + timedelta(days=-7)
        symptom_date_after_21 = pd.to_datetime(symptom_date1) + timedelta(days=21)
        symptom_date_before_20 = pd.to_datetime(symptom_date1) + timedelta(days=-20)
        symptom_date_before_10 = pd.to_datetime(symptom_date1) + timedelta(days=-10)

        # calculate mean RHR for train data for delta RHR calculations
        train_reset = train.reset_index()
        train_baseline_RHR = train_reset['RHR'].mean()

        # calculate mean RHR for train data for delta RHR calculations
        symptom_date1 = pd.to_datetime(symptom_date)
        symptom_date_before_7 = pd.to_datetime(symptom_date1) + timedelta(days=-7)
        symptom_date_after_21 = pd.to_datetime(symptom_date1) + timedelta(days=21)
        test_anomaly_RHR = test[symptom_date_before_7:symptom_date_after_21]
        test_anomaly_delta_RHR = test_anomaly_RHR['RHR'] - train_baseline_RHR

        with open(myphd_id+'_data_split_dates.csv', 'w') as f:
            print("id","start_date ","symptom_date_before_20 ","symptom_date_before_7 ", "symptom_date_before_10 ", "symptom_date_after_21 ","end_date ","\n",
                myphd_id, start1, symptom_date_before_20, symptom_date_before_7, symptom_date_before_10, symptom_date_after_21, end, file=f)

        return symptom_date1, symptom_date_before_20, symptom_date_before_7, symptom_date_before_10, symptom_date_after_21, train, test, test_anomaly_delta_RHR


    # standardization ------------------------------------------------------

    def standardization(self, train_data, test_data, symptom_date_before_20, symptom_date_before_7, symptom_date_before_10, symptom_date_after_21):
        """
        It standardizes the data with zero mean and unit variance (Z-score). 
        We should normalize the test data using the feature summary statistics computed from the training data. 

        It also  splits the test data into test-normal and tes-anomaly for metrics calcualtions later
        It calculates delta RHR for test-anomlay data using baseline/training data
        """

        # standardize train data 
        scaler = StandardScaler().fit(train_data)
        train_data[['RHR']] = scaler.fit_transform(train_data[['RHR']])
        print(train_data)

        # standardize test data 
        test_data[['RHR']] = scaler.transform(test_data[['RHR']])
        test_data = test_data.drop(['level_0'], axis=1)
        print(test_data)

        # split data for test_normal and test_anomaly
        test_anomaly = test_data[symptom_date_before_7:symptom_date_after_21]
        test_normal = test_data[symptom_date_before_20:symptom_date_before_10]

        all_merged = pd.concat([train_data, test_data])
        print(all_merged)

        with open(myphd_id+'_data_size.csv', 'w') as f:
            print("id","train ","test ", "test_normal ", "test_anomaly ","\n",
                myphd_id, train_data.shape, test_data.shape, test_normal.shape, test_anomaly.shape, file=f)

        return train_data, test_data, test_normal, test_anomaly, all_merged


    # creating LSTM input ------------------------------------------------------
    """
    Apply lag method to create subsequences by keepng the temporal order of the data constant 
    """

    def create_dataset(self, dataset, time_steps=1):
        Xs = []
        for i in range(len(dataset) - time_steps):
            v = dataset.iloc[i:(i + time_steps)].values
            Xs.append(v)
        return np.array(Xs)


    # Data Augmentation ------------------------------------------------------
    """
    Applies a combination of different distortions to the data including 
    scaling, rotating, permutating, magnitude warping, time-warping, window slicing, window warping
    """

    def augmentation(self, dataset):

        def scaling(dataset, sigma=0.1):
            factor = np.random.normal(loc=1., scale=sigma, size=(dataset.shape[0],dataset.shape[2]))
            data_scaled = np.multiply(dataset, factor[:,np.newaxis,:])
            return data_scaled

        def rotation(dataset):
            flip = np.random.choice([-1, 1], size=(dataset.shape[0],dataset.shape[2]))
            rotate_axis = np.arange(dataset.shape[2])
            np.random.shuffle(rotate_axis) 
            data_rotation = flip[:,np.newaxis,:] * dataset[:,:,rotate_axis]   
            return data_rotation

        def permutation(dataset, max_segments=5, seg_mode="equal"):
            orig_steps = np.arange(dataset.shape[1])
            num_segs = np.random.randint(1, max_segments, size=(dataset.shape[0]))
            data_permute = np.zeros_like(dataset)
            for i, pat in enumerate(dataset):
                if num_segs[i] > 1:
                    if seg_mode == "random":
                        split_points = np.random.choice(dataset.shape[1]-2, num_segs[i]-1, replace=False)
                        split_points.sort()
                        splits = np.split(orig_steps, split_points)
                    else:
                        splits = np.array_split(orig_steps, num_segs[i])
                    warp = np.concatenate(np.random.permutation(splits)).ravel()
                    data_permute[i] = pat[warp]
                else:
                    data_permute[i] = pat
            return data_permute

        def magnitude_warp(dataset, sigma=0.2, knot=4):
            from scipy.interpolate import CubicSpline
            orig_steps = np.arange(dataset.shape[1])
            random_warps = np.random.normal(loc=1.0, scale=sigma, size=(dataset.shape[0], knot+2, dataset.shape[2]))
            warp_steps = (np.ones((dataset.shape[2],1))*(np.linspace(0, dataset.shape[1]-1., num=knot+2))).T
            data_m_Warp = np.zeros_like(dataset)
            for i, pat in enumerate(dataset):
                warper = np.array([CubicSpline(warp_steps[:,dim], random_warps[i,:,dim])(orig_steps) for dim in range(dataset.shape[2])]).T
                data_m_Warp[i] = pat * warper
            return data_m_Warp

        def time_warp(dataset, sigma=0.2, knot=4):
            from scipy.interpolate import CubicSpline
            orig_steps = np.arange(dataset.shape[1])
            random_warps = np.random.normal(loc=1.0, scale=sigma, size=(dataset.shape[0], knot+2, dataset.shape[2]))
            warp_steps = (np.ones((dataset.shape[2],1))*(np.linspace(0, dataset.shape[1]-1., num=knot+2))).T
            data_t_Warp = np.zeros_like(dataset)
            for i, pat in enumerate(dataset):
                for dim in range(dataset.shape[2]):
                    time_warp = CubicSpline(warp_steps[:,dim], warp_steps[:,dim] * random_warps[i,:,dim])(orig_steps)
                    scale = (dataset.shape[1]-1)/time_warp[-1]
                    data_t_Warp[i,:,dim] = np.interp(orig_steps, np.clip(scale*time_warp, 0, dataset.shape[1]-1), pat[:,dim]).T
            return data_t_Warp

        def window_slice(dataset, reduce_ratio=0.9):
            target_len = np.ceil(reduce_ratio*dataset.shape[1]).astype(int)
            if target_len >= dataset.shape[1]:
                return dataset
            starts = np.random.randint(low=0, high=dataset.shape[1]-target_len, size=(dataset.shape[0])).astype(int)
            ends = (target_len + starts).astype(int)
            data_w_Slice = np.zeros_like(dataset)
            for i, pat in enumerate(dataset):
                for dim in range(dataset.shape[2]):
                    data_w_Slice[i,:,dim] = np.interp(np.linspace(0, target_len, num=dataset.shape[1]), np.arange(target_len), pat[starts[i]:ends[i],dim]).T
            return data_w_Slice

        def window_warp(dataset, window_ratio=0.1, scales=[0.5, 2.]):
            warp_scales = np.random.choice(scales, dataset.shape[0])
            warp_size = np.ceil(window_ratio*dataset.shape[1]).astype(int)
            window_steps = np.arange(warp_size)
            window_starts = np.random.randint(low=1, high=dataset.shape[1]-warp_size-1, size=(dataset.shape[0])).astype(int)
            window_ends = (window_starts + warp_size).astype(int)
            data_w_Warp = np.zeros_like(dataset)
            for i, pat in enumerate(dataset):
                for dim in range(dataset.shape[2]):
                    start_seg = pat[:window_starts[i],dim]
                    window_seg = np.interp(np.linspace(0, warp_size-1, num=int(warp_size*warp_scales[i])), window_steps, pat[window_starts[i]:window_ends[i],dim])
                    end_seg = pat[window_ends[i]:,dim]
                    warped = np.concatenate((start_seg, window_seg, end_seg))                
                    data_w_Warp[i,:,dim] = np.interp(np.arange(dataset.shape[1]), np.linspace(0, dataset.shape[1]-1., num=warped.size), warped).T
            return data_w_Warp

        data_scaled = scaling(dataset)
        data_rotation = rotation(dataset)
        data_permute = permutation(dataset)
        data_m_Warp = magnitude_warp(dataset)
        data_t_Warp = time_warp(dataset)
        data_w_Slice = window_slice(dataset)
        data_w_Warp = window_warp(dataset)

        augment_dataset = np.concatenate([dataset, data_scaled, data_rotation, data_permute, data_m_Warp, data_t_Warp, data_w_Slice, data_w_Warp])

        return augment_dataset


    # LSTM Autoencoder model ------------------------------------------------------
    """
    Builds ENCODER and DECODER architecture with LSTM layers
    """

    def LA(self, train, valid):
        model = keras.Sequential()
        # shape [batch, time, features] => [batch, time, lstm_units]
        model.add(keras.layers.LSTM(units=128, 
            input_shape=(train_dataset.shape[1], train_dataset.shape[2]), 
            return_sequences=True))
        #model.add(keras.layers.Dropout(rate=0.2))
        model.add(keras.layers.LSTM(units=64, return_sequences=False))
        model.add(keras.layers.RepeatVector(n=train_dataset.shape[1]))
        model.add(keras.layers.LSTM(units=64, return_sequences=True))
        model.add(keras.layers.LSTM(units=128, return_sequences=True))
        #model.add(keras.layers.Dropout(rate=0.2))
        # shape => [batch, time, features]
        model.add(keras.layers.TimeDistributed(
            keras.layers.Dense(units=train_dataset.shape[2])))
        model.compile(loss=tf.losses.MeanSquaredError(),
                optimizer=tf.optimizers.Adam(lr = LEARNING_RATE),
                metrics=[tf.metrics.MeanSquaredError()])
        history = model.fit(train, valid, 
            batch_size=BATCH_SIZE,
            epochs=EPOCHS,
            validation_split=VALIDATION_SPLIT, 
            shuffle=False,
            callbacks=[early_stopping_callback, checkpoint_callback])
        return history, model

   # visualization ------------------------------------------------------

    def visualize_loss(self, history):
        history = pd.DataFrame(history.history)
        fig, ax = plt.subplots(1, figsize=(8,6))
        #ax = plt.figure(figsize=(8,5)).gca()
        ax.plot(history['loss'], lw=1, c='blue')
        ax.plot(history['val_loss'], lw=1, c='magenta')
        plt.ylabel('Loss\n')
        plt.xlabel('\nEpoch')
        plt.legend(['train', 'validation'])
        plt.title(myphd_id)
        plt.tight_layout()
        figure = fig.savefig(myphd_id+"_loss.pdf")
        return figure


    # save model  ------------------------------------------------------

    def save_model(self, model):
        MODEL_PATH = myphd_id+'.pth'
        torch.save(model, MODEL_PATH)
        return MODEL_PATH



    # define automatic threshold  ------------------------------------------------------
    """
    take the maximum MAE - Mean Absolute Error (loss) value of the train data  as a threshold to detect anomalies in test data
    """

    def predictions_loss_train(self, losses, train_dataset):
        plt.figure(figsize=(5,3))
        figure = sns.distplot(losses, bins=50, kde=True).set_title(myphd_id)
        plt.savefig(myphd_id+"_predictions_loss_train.pdf")
        return figure

    def anomaly_threshold(self, losses):
        stats = pd.DataFrame(losses).describe()
        #print(stats)
        mean = stats.filter(like='mean', axis=0)
        mean = float(mean[0]) 
        std = stats.filter(like='std', axis=0)
        std = float(std[0]) 
        max = stats.filter(like='max', axis=0)
        max = float(max[0])
        
        #THRESHOLD = max
        # We can calculate the mean and standard deviation of training data loss 
        # then calculate the cut-off as more than 2 standard deviations from the mean.
        # We can then identify anomalies as those examples that fall outside of the defined upper limit.
        cut_off = std * 3
        THRESHOLD =  mean + cut_off
        return THRESHOLD


    # visualization ------------------------------------------------------
    """
    visualise losses
    """

    def predictions_loss_test_normal(self, losses, train_normal_dataset):
        plt.figure(figsize=(5,3))
        figure = sns.distplot(losses, bins=50, kde=True).set_title(myphd_id)
        plt.savefig(myphd_id + "_predictions_loss_test_normal.pdf")
        return figure

    def predictions_loss_test_anomaly(self, losses, test_anomaly_dataset):
        plt.figure(figsize=(5,3))
        figure = sns.distplot(losses, bins=50, kde=True).set_title(myphd_id)
        plt.savefig(myphd_id + "_predictions_loss_test_anomaly.pdf")
        return figure
    
    def predictions_loss_test(self, losses, test_dataset):
        plt.figure(figsize=(5,3))
        figure = sns.distplot(losses, bins=50, kde=True).set_title(myphd_id)
        plt.savefig(myphd_id + "_predictions_loss_test.pdf")
        return figure


    # save anomalies  and delta RHR ------------------------------------------------------
    """
    Save anomalies predicted in test data.
    Calculate delta RHR of test anomaly reltive to trainn/baseline.
    """

    def save_anomalies(self, test, test_anomaly_delta_RHR):
        test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
        test_score_df['loss'] = losses
        test_score_df['threshold'] = THRESHOLD
        test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
        test_score_df['RHR'] = test[TIME_STEPS:].RHR
        anomalies = test_score_df[test_score_df.anomaly == True]

        print("..................................................................\n" + myphd_id +": Anomalies:")
        print("..................................................................\n")
        print(anomalies)

        #save delta RHR of test anomaly data
        delta_RHR = pd.merge(anomalies, test_anomaly_delta_RHR, left_index=True, right_index=True)
        delta_RHR = delta_RHR.rename(columns={'RHR_y':'delta_RHR'})
        delta_RHR = delta_RHR['delta_RHR']
        #print(delta_RHR)
        delta_RHR.to_csv(myphd_id + '_delta_RHR.csv')
        anomalies.to_csv(myphd_id + '_anomalies.csv')
        return anomalies, delta_RHR

    # evaluate complete dataset  ------------------------------------------------------
    """
    For figures evaluate complete dataset annd plot loss of the all  the values as anomaly score later
    """

    def evaluate_complete_dataset(self, all_merged, THRESHOLD):
        plt.figure(figsize=(5,3))
        sns.distplot(losses, bins=50, kde=True).set_title(myphd_id)
        plt.savefig(myphd_id + "_predictions_loss_all.pdf")
        anomalies = sum(l < THRESHOLD for l in losses)

        all_score_df = pd.DataFrame(index=all_merged[TIME_STEPS:].index)
        all_score_df['loss'] = losses
        all_score_df['threshold'] = THRESHOLD
        all_score_df['anomaly'] = all_score_df.loss > all_score_df.threshold
        all_score_df['RHR'] = all_merged[TIME_STEPS:].RHR
        all_anomalies = all_score_df
        
        all_anomalies.index = all_anomalies.index.rename('datetime')
        all_anomalies = all_anomalies.sort_index()

        #print(all_anomalies)

        all_anomalies.to_csv(myphd_id + '_anomalies_all.csv')
        return all_anomalies


    # evaluate metrics  ------------------------------------------------------
    """
    True positives (TP) are the number of anomalous days that are correctly identified as anomalous,
    False negatives (FN) are the no.of anomalous days that are incorrectly identified as normal.
    -7+21 window (True preds are TPs and False are TNs)
    True negative (TN) are the number of normal days that are correctly identified as normal
    False positives (FP) are the no.of normal days that are incorrectly identified as anomalous. 
    -7: window (False=1)
    """


    def metrics_1(self, all_anomalies, test_normal_data, symptom_date_before_7, symptom_date_after_21):

        def listToStringWithoutBrackets(list1):
            return str(list1).replace('[','').replace(']','').replace('\'','').replace('(','').replace(')','').replace(': , ',':').replace(':, ',':')


        all_score_df1 = all_anomalies[['anomaly', 'RHR']]
        all_score_df1 = all_score_df1.resample('1H').mean()
        all_score_df1 =  all_score_df1.fillna(0)
        all_score_df1 = all_score_df1[all_score_df1['RHR']!=0]

        test_anomaly_df1 = all_score_df1[symptom_date_before_7:symptom_date_after_21]
        test_anomaly_df1 = test_anomaly_df1.groupby(['anomaly']).count()
        test_anomaly_df1 = test_anomaly_df1.reset_index()

        test_anomaly_df2 = test_anomaly_df1[test_anomaly_df1['anomaly'] == True]
        TP = int(test_anomaly_df2['RHR'].values) if len(test_anomaly_df2)>0 else 0
        print("..................................................................\n" + myphd_id +": Metrics:")
        print("..................................................................\n")

        # True negative (TN) are the number of normal days that are correctly identified as normal
        # False positives (FP) are the no.of normal days that are incorrectly identified as anomalous. 
        #-20:-10 window (False=1)

        test_anomaly_df3 = test_anomaly_df1[test_anomaly_df1['anomaly'] == False]
        FN = int(test_anomaly_df3['RHR'].values) if len(test_anomaly_df3)>0 else 0

        test_normal_df1 = pd.merge(test_normal_data, all_anomalies,  how='outer',  left_index=True, right_index=True)
        #print(test_normal_df1)
        test_normal_df1 = test_normal_df1.loc[test_normal_df1['RHR_x'].notnull()]
        test_normal_df1 = test_normal_df1.drop({'RHR_x'},  axis=1)
        test_normal_df1 = test_normal_df1.rename({'RHR_y':'RHR'}, axis=1)

        test_normal_df1 = test_normal_df1.groupby(['anomaly']).count()
        test_normal_df1 = test_normal_df1.reset_index()

        test_normal_df2 = test_normal_df1[test_normal_df1['anomaly'] == False]
        TN = int(test_normal_df2['RHR'].values) if len(test_normal_df2)>0 else 0

        test_normal_df3 = test_normal_df1[test_normal_df1['anomaly'] == True]
        FP = int(test_normal_df3['RHR'].values) if len(test_normal_df3)>0 else 0

        print("TP:",TP,"FP:",FP,"TN:",TN,"FN:",FN)

        formatted_list_2 = [TP, FP, TN, FN]
        formatted_list_2_df = pd.DataFrame([formatted_list_2])
        formatted_list_2_df.columns =['TP', 'FP', 'TN', 'FN']
        formatted_list_2_df.rename({0: myphd_id}, axis='index')
        formatted_list_2_df.index = [myphd_id]
        formatted_list_2_df.to_csv(myphd_id + '_all_basic_metrics.csv', header=True)

        formatted_list_2  = ('TP: ', TP,'FP: ', FP,'TN: ',TN,'FN:',FN)
        formatted_list_2 = listToStringWithoutBrackets(formatted_list_2)


        return TP, FP, TN, FN, formatted_list_2

    # visualization ------------------------------------------------------

    def visualize_complete_dataset1(self, all_anomalies, symptom_date1, symptom_date_before_7, symptom_date_after_21, formatted_list_2):
        
        # original sequence
        all_score_df = all_anomalies
        ax1 = all_score_df[['RHR']].plot(figsize=(24,4.5), color="black", rot=90)
        ax1.set_xlim(all_score_df.index[0], all_score_df.index[-1]) 
        ax1.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y-%b-%d'))
        ax1.set_ylabel('Orig Seq\n', fontsize = 20) # Y label
        ax1.set_xlabel('', fontsize = 0) # X label
        ax1.tick_params(axis='both', which='major', labelsize=22)
        ax1.set_xlabel('', fontsize = 0) # X label
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.set_title(myphd_id,fontweight="bold", size=30) # Title
        plt.xticks(fontsize=0, rotation=90)
        plt.tick_params(axis='both',which='both',bottom=True, top=False, labelbottom=True)
        plt.tight_layout()
        plt.savefig(myphd_id + '_all_original_seq.pdf', bbox_inches='tight')  
        #plt.show()


        # plot anomaly scores
        all_score_df1 = all_anomalies.reset_index()
        ax3 = all_score_df1.plot.scatter(x='datetime',y='loss', figsize=(24,4),  rot=90, marker='o', lw=2,
            c=['red' if i== True else 'mediumblue'  for i in all_score_df1['anomaly']])
        ax3.set_xlim(all_score_df1['datetime'].iloc[0], all_score_df1['datetime'].iloc[-1]) 
        ax3.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax3.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%y-%b-%d'))
        ax3.set_ylabel('Anomaly Score\n', fontsize = 20) # Y label
        ax3.set_xlabel('', fontsize = 0) # X label
        ax3.axvline(pd.to_datetime(symptom_date_before_7), color='orange', zorder=1, linestyle='--',  lw=6, alpha=0.5) # Symptom date 
        ax3.axvline(pd.to_datetime(symptom_date1), color='red', zorder=1, linestyle='--',  lw=6, alpha=0.5) # Symptom date 
        ax3.axvline(pd.to_datetime(symptom_date_after_21), color='purple', zorder=1, linestyle='--', lw=6, alpha=0.5) # Symptom date 
        ax3.tick_params(axis='both', which='major', labelsize=18)
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax3.set_title(myphd_id+ '\n', fontweight="bold", size=30) # Title
        plt.axhline(y=THRESHOLD, color='grey', linestyle='--', lw=3, alpha=0.3)
        plt.tick_params(axis='both',which='both',bottom=True, top=False, labelbottom=True) 
        #plt.title(myphd_id + '\n\n', fontweight="bold", size=30) # Sub title
        plt.suptitle(formatted_list_2+ '\n', fontweight="bold", size=20) # Sub title
        #plt.tight_layout()
        plt.savefig(myphd_id + '_all_anomaly_scores.pdf', bbox_inches='tight')  
        #plt.show()


    def metrics_2(self, tp, fp, tn, fn):
        Sensitivity = tp / (tp+fn),
        Specificity = tn / (tn+fp),
        PPV = tp / (tp+fp),
        NPV = tn / (tn+fn),
        Precision = tp / (tp+fp),
        Recall = tp / (tp+fn),
        # F1 = 2 * ( (Precision * Recall) / (Precision + Recall) )
        F1 =  2 *( ((tp / (tp+fp)) * (tp / (tp+fn))) / ((tp / (tp+fp)) + (tp / (tp+fn))))
        # Fbeta = ((1 + beta^2) * Precision * Recall) / (beta^2 * Precision + Recall)
        Fbeta =  ((1+0.1**2) * ((tp / (tp+fp)) * (tp / (tp+fn)))) / ((0.1**2) * (tp / (tp+fp)) + (tp / (tp+fn)))
        return Sensitivity, Specificity, PPV, NPV, Precision, Recall, F1, Fbeta


    # save metrics  ------------------------------------------------------
    """
    Calculate Sensitivity, Specificity, PPV, NPV, Precision, Recall, F1
    """

    def save_metrics(self, TP, FP, TN, FN, Sensitivity, Specificity, PPV, NPV, Precision, Recall, F1, Fbeta):
        
        def listToStringWithoutBrackets(list1):
            return str(list1).replace('[','').replace(']','').replace('\'','').replace('(','').replace(')','').replace(': , ',':').replace(':, ',':')

        Sensitivity = [ '%.3f' % elem for elem in Sensitivity ]
        Specificity = [ '%.3f' % elem for elem in Specificity ]
        PPV = [ '%.3f' % elem for elem in PPV ]
        NPV = [ '%.3f' % elem for elem in NPV ]
        Precision = [ '%.3f' % elem for elem in Precision ]
        Recall = [ '%.3f' % elem for elem in Recall ]
        F1 = round(F1,3)
        Fbeta = round(Fbeta,3)


        formatted_list  = ('TP: ', TP,'FP: ', FP,'TN: ',TN,'FN:',FN, 
            'Sensitivity:',Sensitivity,'Specificity:',Specificity,
            'PPV:',PPV, 'NPV:', NPV,
            'Precision:',Precision, 'Recall:',Recall, 'F1:',F1, 'Fbeta:', Fbeta)

        formatted_list_1  = ('TP: ', TP,'FP: ', FP,'TN: ',TN,'FN:',FN, 
            'Precision:',Precision, 'Recall:',Recall, 'F1:',F1,'Fbeta:', Fbeta)

        formatted_list = listToStringWithoutBrackets(formatted_list)
        formatted_list_1 = listToStringWithoutBrackets(formatted_list_1)

        #print(formatted_list_1)

        metrics_list = [TP, FP, TN, FN, Sensitivity, Specificity, PPV, NPV, Precision, Recall, F1, Fbeta]
        #metrics_list = listToStringWithoutBrackets(metrics_list)

        metrics_df = pd.DataFrame([metrics_list])
        metrics_df.columns =['TP', 'FP', 'TN', 'FN', 'Sensitivity','Specificity','PPV', 'NPV', 'Precision', 'Recall', 'F1', 'Fbeta']
        metrics_df.rename({0: myphd_id}, axis='index')
        metrics_df.index = [myphd_id]
        metrics_df.to_csv(myphd_id + '_metrics.csv', header=True)

        #print(metrics_df)

        return formatted_list, formatted_list_1


    # visualization ------------------------------------------------------

    def visualize_complete_dataset2(self, all_anomalies, symptom_date1, symptom_date_before_7, symptom_date_after_21, formatted_list_1):
        # plot original data
        all_score_df = all_anomalies
        ax1 = all_score_df[['RHR']].plot(figsize=(24,4.5), color="black", rot=90)
        ax1.set_xlim(all_score_df.index[0], all_score_df.index[-1]) 
        ax1.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%y-%b-%d'))
        ax1.set_ylabel('Orig Seq\n', fontsize = 20) # Y label
        ax1.set_xlabel('', fontsize = 0) # X label
        ax1.tick_params(axis='both', which='major', labelsize=22)
        ax1.set_xlabel('', fontsize = 0) # X label
        ax1.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax1.set_title(myphd_id,fontweight="bold", size=30) # Title
        plt.xticks(fontsize=0, rotation=90)
        plt.tick_params(axis='both',which='both',bottom=True, top=False, labelbottom=True)
        plt.tight_layout()
        plt.savefig(myphd_id + '_all_original_seq.pdf', bbox_inches='tight')  
        #plt.show()

        # plot anomaly scores
        all_score_df1 = all_anomalies.reset_index()
        ax3 = all_score_df1.plot.scatter(x='datetime',y='loss', figsize=(24,4),  rot=90, marker='o', lw=2,
            c=['red' if i== True else 'mediumblue'  for i in all_score_df1['anomaly']])
        ax3.set_xlim(all_score_df1['datetime'].iloc[0], all_score_df1['datetime'].iloc[-1]) 
        ax3.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax3.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%y-%b-%d'))
        ax3.set_ylabel('Anomaly Score\n', fontsize = 20) # Y label
        ax3.set_xlabel('', fontsize = 0) # X label
        ax3.axvline(pd.to_datetime(symptom_date_before_7), color='orange', zorder=1, linestyle='--',  lw=6, alpha=0.5) # Symptom date 
        ax3.axvline(pd.to_datetime(symptom_date1), color='red', zorder=1, linestyle='--',  lw=6, alpha=0.5) # Symptom date 
        ax3.axvline(pd.to_datetime(symptom_date_after_21), color='purple', zorder=1, linestyle='--', lw=6, alpha=0.5) # Symptom date 
        ax3.tick_params(axis='both', which='major', labelsize=18)
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax3.set_title(myphd_id+ '\n', fontweight="bold", size=30) # Title
        plt.axhline(y=THRESHOLD, color='grey', linestyle='--', lw=3, alpha=0.3)
        plt.tick_params(axis='both',which='both',bottom=True, top=False, labelbottom=True) 
        #plt.title(myphd_id + '\n\n', fontweight="bold", size=30) # Sub title
        plt.suptitle(formatted_list_1+ '\n', fontweight="bold", size=20) # Sub title
        #plt.tight_layout()
        plt.savefig(myphd_id + '_all_anomaly_scores.pdf', bbox_inches='tight')  
        #plt.show()


#################################################################################################


LAAD = LAAD()

# pre-process data
df1 = LAAD.resting_heart_rate(fitbit_oldProtocol_hr, fitbit_oldProtocol_steps)
processed_data = LAAD.pre_processing(df1)

# split dates and data using assumptions listed in the paper
symptom_date1, symptom_date_before_20, symptom_date_before_7, symptom_date_before_10, symptom_date_after_21, train, test, test_anomaly_delta_RHR = LAAD.data_splitting(processed_data)

# standardization
train_data, test_data, test_normal_data, test_anomaly_data, all_merged = LAAD.standardization(train, test, symptom_date_before_20, symptom_date_before_7, symptom_date_before_10, symptom_date_after_21)


#  Create subsequences in tensor format from a dataframe
train_dataset= LAAD.create_dataset(train_data[['RHR']],TIME_STEPS)
test_dataset= LAAD.create_dataset(test_data[['RHR']],TIME_STEPS)
#test_normal_dataset= LAAD.create_dataset(test_normal_data[['RHR']],TIME_STEPS)
#test_anomaly_dataset= LAAD.create_dataset(test_anomaly_data[['RHR']],TIME_STEPS)
all_merged_dataset= LAAD.create_dataset(all_merged[['RHR']],TIME_STEPS)


# data augmentation of trainign dataset
train_aug_dataset = LAAD.augmentation(train_dataset)

# Use train model as both input and target  since this is recosntruction model
# save the best model with lowest loss
early_stopping_callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode="min")
checkpoint_callback = keras.callbacks.ModelCheckpoint(myphd_id+'_model.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')

history, LA = LAAD.LA(train_aug_dataset, train_aug_dataset)
LAAD.visualize_loss(history)

# Save the model
#filepath = './'+myphd_id+'_model.h5'
#save_model(LA, filepath, save_format='h5')

# evaluate train dataset to calculate MAE loss and set a threshold
predictions = LA.predict(train_dataset)
losses = np.mean(np.abs(predictions - train_dataset), axis=1)
LAAD.predictions_loss_train(losses, train_dataset)
THRESHOLD = LAAD.anomaly_threshold(losses)

# evaluate test normal and anomaly datasets
#predictions = LA.predict(test_normal_dataset)
#losses = np.mean(np.abs(predictions - test_normal_dataset), axis=1)
#LAAD.predictions_loss_test_normal(losses, test_normal_dataset)

#predictions = LA.predict(test_anomaly_dataset)
#losses = np.mean(np.abs(predictions - test_anomaly_dataset), axis=1)
#LAAD.predictions_loss_test_anomaly(losses, test_anomaly_dataset)

# evaluate test dataset
predictions = LA.predict(test_dataset)
losses = np.mean(np.abs(predictions - test_dataset), axis=1)
LAAD.predictions_loss_test(losses, test_dataset)

# save anomalies
anomalies, delta_RHR = LAAD.save_anomalies(test, test_anomaly_delta_RHR)

# evaluate complete dataset
predictions = LA.predict(all_merged_dataset)
losses = np.mean(np.abs(predictions - all_merged_dataset), axis=1)
all_anomalies = LAAD.evaluate_complete_dataset(all_merged, THRESHOLD)
  
# metrics
TP, FP, TN, FN, formatted_list_2 = LAAD.metrics_1(all_anomalies, test_normal_data, symptom_date_before_7, symptom_date_after_21)
LAAD.visualize_complete_dataset1(all_anomalies, symptom_date1, symptom_date_before_7, symptom_date_after_21, formatted_list_2)
LAAD.visualize_complete_dataset1(all_anomalies, symptom_date1, symptom_date_before_7, symptom_date_after_21, formatted_list_2)
Sensitivity, Specificity, PPV, NPV, Precision, Recall, F1, Fbeta = LAAD.metrics_2(TP, FP, TN, FN)

# visualization
formatted_list,formatted_list_1 = LAAD.save_metrics(TP, FP, TN, FN, Sensitivity, Specificity, PPV, NPV, Precision, Recall, F1, Fbeta)
LAAD.visualize_complete_dataset2(all_anomalies, symptom_date1, symptom_date_before_7, symptom_date_after_21, formatted_list_1)

print("\nCompleted!\n")

