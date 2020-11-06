# LSTM-Autoencoder based Anomaly Detection (LAAD)

######################################################
# Author: Gireesh K. Bogu                            #
# Email: gbogu17@stanford.edu                        #
# Location: Dept.of Genetics, Stanford University    #
# Date: Nov 2 2020                                   #
######################################################

#python laad_RHR.py  --heart_rate COVID-19-Wearables/ASFODQR_hr.csv --steps COVID-19-Wearables/ASFODQR_steps.csv --myphd_id ASFODQR --symptom_date 2024-08-14

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
from pylab import rcParams
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FormatStrFormatter
from matplotlib import rc
from datetime import date, datetime, timedelta
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm, datasets
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score
import torch
from torch import nn, optim
import torch.nn.functional as F
from arff2pandas import a2p
from statsmodels.tsa.seasonal import seasonal_decompose
#%matplotlib inline
#%config InlineBackend.figure_format='retina'
sns.set(style='whitegrid', palette='muted', font_scale=1.2)
palette = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
sns.set_palette(sns.color_palette(palette))
rcParams['figure.figsize'] = 12, 8


####################################

parser = argparse.ArgumentParser(description='Find anomalies in wearables time-series data')
parser.add_argument('--heart_rate', metavar='', help ='raw heart rate count with a header = heartrate')
parser.add_argument('--steps',metavar='', help ='raw steps count with a header = steps')
parser.add_argument('--myphd_id',metavar='', default = 'myphd_id', help ='user myphd_id')
parser.add_argument('--symptom_date', metavar='', default = 'NaN', help = 'symptom date with y-m-d format')
parser.add_argument('--random_seed', metavar='', type=int, default=42, help='random seed')
args = parser.parse_args()

# as arguments
fitbit_oldProtocol_hr = args.heart_rate
fitbit_oldProtocol_steps = args.steps
myphd_id = args.myphd_id
symptom_date = args.symptom_date
RANDOM_SEED = args.random_seed

np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
 


# LSTM ENCODER-DECODER model ########################################################################


class Encoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(Encoder, self).__init__()

    self.seq_len, self.n_features = seq_len, n_features
    self.embedding_dim, self.hidden_dim = embedding_dim, 2 * embedding_dim

    self.rnn1 = nn.LSTM(
      input_size=n_features,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )
    
    self.rnn2 = nn.LSTM(
      input_size=self.hidden_dim,
      hidden_size=embedding_dim,
      num_layers=1,
      batch_first=True
    )

  def forward(self, x):
    x = x.reshape((1, self.seq_len, self.n_features))

    x, (_, _) = self.rnn1(x)
    x, (hidden_n, _) = self.rnn2(x)

    return hidden_n.reshape((self.n_features, self.embedding_dim))



class Decoder(nn.Module):

  def __init__(self, seq_len, input_dim=64, n_features=1):
    super(Decoder, self).__init__()

    self.seq_len, self.input_dim = seq_len, input_dim
    self.hidden_dim, self.n_features = 2 * input_dim, n_features

    self.rnn1 = nn.LSTM(
      input_size=input_dim,
      hidden_size=input_dim,
      num_layers=1,
      batch_first=True
    )

    self.rnn2 = nn.LSTM(
      input_size=input_dim,
      hidden_size=self.hidden_dim,
      num_layers=1,
      batch_first=True
    )

    self.output_layer = nn.Linear(self.hidden_dim, n_features)

  def forward(self, x):
    x = x.repeat(self.seq_len, self.n_features)
    x = x.reshape((self.n_features, self.seq_len, self.input_dim))

    x, (hidden_n, cell_n) = self.rnn1(x)
    x, (hidden_n, cell_n) = self.rnn2(x)
    x = x.reshape((self.seq_len, self.hidden_dim))

    return self.output_layer(x)


class RecurrentAutoencoder(nn.Module):

  def __init__(self, seq_len, n_features, embedding_dim=64):
    super(RecurrentAutoencoder, self).__init__()

    self.encoder = Encoder(seq_len, n_features, embedding_dim).to(device)
    self.decoder = Decoder(seq_len, embedding_dim, n_features).to(device)

  def forward(self, x):
    x = self.encoder(x)
    x = self.decoder(x)

    return x



########################################################################

class RHRAD:

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
        #df_hr = df_hr.resample('1min').median()
        #df_steps = df_steps.resample('1min').median()

        df1 = pd.merge(df_hr, df_steps, left_index=True, right_index=True)
        df1 = df1.resample('1min').mean()
        df1 = df1.dropna()
        
        # define RHR as the HR measurements recorded when there were zero steps taken during a rolling time window of the preceding 12 minutes (including the current minute)
        df1['steps_window_12'] = df1['steps'].rolling(12).sum()
        df1 = df1.loc[(df1['steps_window_12'] == 0)]
        return df1


    # pre-processing ------------------------------------------------------

    def pre_processing(self, resting_heart_rate):
        """
        This function takes resting heart rate data and applies moving averages to smooth the data and 
        downsamples to one hour by taking the avegare values
        """
        # smooth data
        df_nonas = df1.dropna()
        df1_rom = df_nonas.rolling(400).mean()
        # resample
        df1_resmp = df1_rom.resample('1H').mean()
        df2 = df1_resmp.drop(['steps'], axis=1)
        df2 = df2.drop(['steps_window_12'], axis=1)
        df2 = df2.resample('24H').mean()
        df2 = df2.dropna()
        df2 = df2.rename(columns={"heartrate": "RHR"})
        return df2


   # data splitting ------------------------------------------------------

    def data_splitting(self, processed_data):
        train = processed_data[:20]
        df4 = train.sample(frac=0.1, replace=False, random_state=1)
        df5 = pd.merge(train, df4, how='outer', left_index=True, right_index=True, indicator=True)
        df5_train = df5.query('_merge != "both"')
        df5_train = df5_train.drop(['RHR_y', '_merge'], axis=1)
        df5_train = df5_train.rename(columns={"RHR_x": "RHR"})
        df5_valid = df5.query('_merge == "both"')
        df5_valid = df5_valid.drop(['RHR_y', '_merge'], axis=1)
        df5_valid = df5_valid.rename(columns={"RHR_x": "RHR"})
        symptom_date1 = pd.to_datetime(symptom_date)
        symptom_date_before_7 = pd.to_datetime(symptom_date1) + timedelta(days=-7)
        symptom_date_after_21 = pd.to_datetime(symptom_date1) + timedelta(days=21)
        return df5_train, df5_valid, symptom_date1, symptom_date_before_7, symptom_date_after_21

    # standardization ------------------------------------------------------

    def standardization(self, train_data, valid_data, processed_data):
        """
        Standardize the data with zero meann and unit variance (Z-score).
        One common mistake is: we normalize the entire data and then split into train-test. 
        This is not correct. Test data should be completely unseen to anything during the modeling. 
        We should normalize the test data using the feature summary statistics computed from the training data. 
        For normalization, these statistics are the mean and variance for each feature.
        The same logic should be used for the validation set. This makes the model more stable for a test data.
        """
        scaler = StandardScaler().fit(train_data)
        train_data[['RHR']] = scaler.fit_transform(train_data[['RHR']])
        valid_data[['RHR']] = scaler.transform(valid_data[['RHR']])
        test = processed_data[20:]
        test_data = test
        test_data[['RHR']] = scaler.transform(test[['RHR']])
        symptom_date1 = pd.to_datetime(symptom_date)
        symptom_date_before_7 = pd.to_datetime(symptom_date1) + timedelta(days=-7)
        symptom_date_after_21 = pd.to_datetime(symptom_date1) + timedelta(days=21)
        test_anomaly = test_data[symptom_date_before_7:symptom_date_after_21]
        test_normal = pd.merge(test_data,test_anomaly,  how='outer',  left_index=True, right_index=True)
        test_normal = test_normal.loc[test_normal['RHR_y'].isnull()]
        test_normal = test_normal.drop('RHR_y', axis=1)
        test_normal = test_normal.rename({'RHR_x':'RHR'}, axis=1)
        all_merged = pd.concat([train_data, valid_data, test])
        return train_data, valid_data, test, test_data, test_normal, test_anomaly, all_merged

   # creating LSTM input ------------------------------------------------------

    def create_dataset(self, df):
        sequences = df.astype(np.float32).to_numpy().tolist()
        dataset = [torch.tensor(s).unsqueeze(1).float() for s in sequences]
        n_seq, seq_len, n_features = torch.stack(dataset).shape
        return dataset, seq_len, n_features

   # training model ------------------------------------------------------

    def train_model(self, model, train_dataset, val_dataset, n_epochs):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = nn.L1Loss(reduction='sum').to(device) 
        history = dict(train=[], val=[])
        best_model_wts = copy.deepcopy(model.state_dict())
        best_loss = 10000.0

        for epoch in range(1, n_epochs + 1):
            model = model.train()
            train_losses = []

            for seq_true in train_dataset:
                optimizer.zero_grad()
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            val_losses = []
            model = model.eval()

            with torch.no_grad():
             for seq_true in val_dataset:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                val_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            val_loss = np.mean(val_losses)
            history['train'].append(train_loss)
            history['val'].append(val_loss)

            if val_loss < best_loss:
                best_loss = val_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            print(f'Epoch {epoch}: train loss {train_loss} val loss {val_loss}')

        model.load_state_dict(best_model_wts)
        return model.eval(), history

   # visualization ------------------------------------------------------

    def visualize_loss(self, history):
        fig, ax = plt.subplots(1, figsize=(8,6))
        #ax = plt.figure(figsize=(8,5)).gca()
        ax.plot(history['train'], lw=1, c='blue')
        ax.plot(history['val'], lw=1, c='magenta')
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

    def predict(self, model, dataset):
        predictions, losses = [], []
        criterion = nn.L1Loss(reduction='sum').to(device)
        with torch.no_grad():
            model = model.eval()
            for seq_true in dataset:
                seq_true = seq_true.to(device)
                seq_pred = model(seq_true)
                loss = criterion(seq_pred, seq_true)
                predictions.append(seq_pred.cpu().numpy().flatten())
                losses.append(loss.item())
        return predictions, losses

    def predictions_loss_train(self, losses, train_dataset):
        plt.figure(figsize=(5,3))
        figure = sns.distplot(losses, bins=50, kde=True).set_title(myphd_id)
        plt.savefig(myphd_id+"_predictions_loss_train.pdf")
        return figure

    def anomaly_threshold(self, losses):
        stats = pd.DataFrame(losses).describe()
        mean = stats.filter(like='mean', axis=0)
        mean = float(mean[0]) 
        std = stats.filter(like='std', axis=0)
        std = float(std[0]) 
        THRESHOLD = mean + std + 0.5
        return THRESHOLD


   # visualization ------------------------------------------------------

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


    # save anomalies  ------------------------------------------------------

    def save_anomalies(self, test):
        df3_test = test.reset_index()
        df3_test = df3_test.rename(columns={'index':'datetime'})
        test_score_df = df3_test
        test_score_df['loss'] = pred_losses
        test_score_df['threshold'] = THRESHOLD
        test_score_df['anomaly'] = test_score_df.loss > test_score_df.threshold
        test_score_df = test_score_df.set_index("datetime")
        anomalies = test_score_df[test_score_df.anomaly == True]
        anomalies.to_csv(myphd_id + '_anomalies.csv')
        return anomalies

    # evaluate complete dataset  ------------------------------------------------------

    def evaluate_complete_dataset(self, all_merged, THRESHOLD):
        plt.figure(figsize=(5,3))
        sns.distplot(pred_losses, bins=50, kde=True).set_title(myphd_id)
        plt.savefig(myphd_id + "_predictions_loss_all.pdf")
        anomalies = sum(l < THRESHOLD for l in pred_losses)
        df3_all = all_merged.reset_index()
        df3_all = df3_all.rename(columns={'index':'datetime'})
        all_score_df = df3_all
        all_score_df['predictions'] = predictions
        all_score_df['loss'] = pred_losses
        all_score_df['threshold'] = THRESHOLD
        all_score_df['anomaly'] = all_score_df.loss > all_score_df.threshold
        all_score_df['predictions'] = all_score_df['predictions'].str[0]
        all_anomalies = all_score_df.set_index("datetime")
        #print(all_anomalies)
        return all_anomalies

    # visualization ------------------------------------------------------

    def visualize_complete_dataset(self, all_anomalies):
        # plot original data
        all_score_df = all_anomalies.reset_index()
        all_score_df = all_score_df.set_index('datetime')
        all_score_df.index.name = None
        all_score_df.index = pd.to_datetime(all_score_df.index)
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

        # plot reconstructed data
        ax2 = all_score_df[['predictions']].plot(figsize=(24,4.5), color="green", rot=90)
        ax2.set_xlim(all_score_df.index[0], all_score_df.index[-1])  
        ax2.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%y-%b-%d'))
        ax2.set_ylabel('Recon Seq\n', fontsize = 20) # Y label
        ax2.set_xlabel('', fontsize = 0) # X label
        ax2.tick_params(axis='both', which='major', labelsize=22)
        ax2.set_xlabel('', fontsize = 0) # X label
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax2.set_title(myphd_id,fontweight="bold", size=30) # Title
        plt.xticks(fontsize=0, rotation=90)
        plt.tick_params(axis='both',which='both',bottom=True, top=False, labelbottom=True)
        plt.tight_layout()
        plt.savefig(myphd_id + '_all_reconstructed_seq.pdf', bbox_inches='tight')  
        #plt.show()

        # plot anomaly scores
        all_score_df1 = all_anomalies.reset_index()
        ax3 = all_score_df1.plot.scatter(x='datetime',y='loss', figsize=(24,6),  rot=90, marker='o', lw=5,
            c=['red' if i== True else 'blue'  for i in all_score_df1['anomaly']])
        ax3.set_xlim(all_score_df1['datetime'].iloc[0], all_score_df1['datetime'].iloc[-1]) 
        ax3.xaxis.set_minor_locator(mdates.DayLocator(interval=1))
        ax3.xaxis.set_major_locator(mdates.DayLocator(interval=2))
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%y-%b-%d'))
        ax3.set_ylabel('Anomaly Score\n', fontsize = 20) # Y label
        ax3.set_xlabel('', fontsize = 0) # X label
        ax3.tick_params(axis='both', which='major', labelsize=22)
        ax3.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax3.set_title(myphd_id,fontweight="bold", size=30) # Title
        plt.axhline(y=THRESHOLD, color='r', linestyle='--', lw=4)
        plt.tick_params(axis='both',which='both',bottom=True, top=False, labelbottom=True) 
        plt.tight_layout()
        plt.savefig(myphd_id + '_all_anomaly_scores.pdf', bbox_inches='tight')  
        #plt.show()


    # evaluate metrics  ------------------------------------------------------


    def metrics_1(self, all_anomalies, test_normal_data, symptom_date_before_7, symptom_date_after_21):
        # True positives (TP) are the number of anomalous days that are correctly identified as anomalous,
        # False negatives (FN) are the no.of anomalous days that are incorrectly identified as normal.
        #7-21 window (True preds are TPs and False are TNs)

        all_score_df1 = all_anomalies[['anomaly', 'predictions']]
        test_anomaly_df1 = all_score_df1[symptom_date_before_7:symptom_date_after_21]
        test_anomaly_df1 = test_anomaly_df1.groupby(['anomaly']).count()
        test_anomaly_df1 = test_anomaly_df1.reset_index()

        test_anomaly_df2 = test_anomaly_df1[test_anomaly_df1['anomaly'] == True]
        TP = int(test_anomaly_df2['predictions'].values) if len(test_anomaly_df2)>0 else 0
        print("..................................................................\nMetrics:\n")

        test_anomaly_df3 = test_anomaly_df1[test_anomaly_df1['anomaly'] == False]
        FN = int(test_anomaly_df3['predictions'].values) if len(test_anomaly_df3)>0 else 0

        # True negative (TN) are the number of normal days that are correctly identified as normal
        # False positives (FP) are the no.of normal days that are incorrectly identified as anomalous. 
        #7: window (False=1)

        test_normal_df1 = pd.merge(test_normal_data, all_anomalies,  how='outer',  left_index=True, right_index=True)
        test_normal_df1 = test_normal_df1.loc[test_normal_df1['RHR_x'].notnull()]
        test_normal_df1 = test_normal_df1.drop({'RHR_x'},  axis=1)
        test_normal_df1 = test_normal_df1.rename({'RHR_y':'RHR'}, axis=1)

        test_normal_df1 = test_normal_df1.groupby(['anomaly']).count()
        test_normal_df1 = test_normal_df1.reset_index()

        test_normal_df2 = test_normal_df1[test_normal_df1['anomaly'] == False]
        TN = int(test_normal_df2['predictions'].values) if len(test_normal_df2)>0 else 0

        test_normal_df3 = test_normal_df1[test_normal_df1['anomaly'] == True]
        FP = int(test_normal_df3['predictions'].values) if len(test_normal_df3)>0 else 0
        return TP, FP, TN, FN


    def metrics_2(self, tp, fp, tn, fn):
        sensitivity = tp / (tp+fn),
        specificity = tn / (tn+fp),
        precision = tp / (tp+fp),
        recall = tn / (tn+fn),
        return sensitivity, specificity, precision, recall


    # ROC plot ------------------------------------------------------

    def roc_input(self, all_anomalies, test_normal_data, symptom_date_before_7, symptom_date_after_21):

        def test_normal_types(anomaly):
            if anomaly['anomaly'] == True:
                return 1
            elif anomaly['anomaly'] == False:
                return 0
            else:
                return 'OTHER'


        def test_anomaly_types(anomaly):
            if anomaly['anomaly'] == False:
                return 1
            elif anomaly['anomaly'] == True:
                return 0
            else:
                return 'OTHER'

        all_score_df1 = all_anomalies[['anomaly', 'predictions']]
        test_anomaly_df1 = all_score_df1[symptom_date_before_7:symptom_date_after_21]
        test_normal_df1 = pd.merge(test_normal_data, all_anomalies,  how='outer',  left_index=True, right_index=True)
        test_normal_df1 = test_normal_df1.loc[test_normal_df1['RHR_x'].notnull()]
        test_normal_df1 = test_normal_df1.drop({'RHR_x'},  axis=1)
        test_normal_df1 = test_normal_df1.rename({'RHR_y':'RHR'}, axis=1)
        test_anomaly_df1['y_score'] = test_anomaly_df1.apply(test_anomaly_types, axis=1)
        test_anomaly_df1['y_test'] = '1'
        test_normal_df1['y_score'] = test_normal_df1.apply(test_normal_types, axis=1)
        test_normal_df1['y_test'] = '0'
        preds = pd.concat([test_anomaly_df1, test_normal_df1], axis=0)
        preds1 = preds.drop(['anomaly', 'predictions', 'loss', 'threshold', 'RHR'], axis=1)
        roc_input = preds1.reset_index()
        roc_input = roc_input.rename({'index':'datetime'}, axis=1)
        return roc_input


    def roc_plot(self, roc_input):
        n_classes = 2
        y_test = np.array(roc_input['y_test'], dtype=int)
        y_score = np.array(roc_input['y_score'], dtype=int)
        plt.figure(figsize=(5,5))
        fpr, tpr, thresholds = roc_curve(y_test, y_score)
        auc1 = auc(fpr,tpr)

        plt.plot(fpr, tpr,label='AUC = %0.2f)' % auc1, color='red', linewidth=2)

        plt.plot([0, 1], [0, 1], 'k--', lw=1)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')  
        plt.ylabel('True Positive Rate') 
        plt.title(myphd_id) 
        plt.grid(True)
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(myphd_id + '_roc_plot.pdf', bbox_inches='tight') 
        return auc1

    # save metrics  ------------------------------------------------------

    def save_metrics(self, TP, FP, TN, FN, sensitivity, specificity, precision, recall, AUC):
        metrics_list = [TP, FP, TN, FN, sensitivity, specificity, precision, recall, AUC]
        metrics_df = pd.DataFrame([metrics_list])
        metrics_df.columns =['TP', 'FP', 'TN', 'FN', 'Sensitivity','Specificity','Precision', 'Recall', 'AUC']
        metrics_df.rename({0: myphd_id}, axis='index')
        metrics_df.index = [myphd_id]
        print(metrics_df)
        metrics_df.to_csv(myphd_id + '_metrics.csv', header=True)
        return metrics_df


#################################################################################################

rhrad = RHRAD()

# pre-process data
df1 = rhrad.resting_heart_rate(fitbit_oldProtocol_hr, fitbit_oldProtocol_steps)
processed_data = rhrad.pre_processing(df1)

# split dates and data using assumptions listed in the paper
train, valid, symptom_date1, symptom_date_before_7, symptom_date_after_21 = rhrad.data_splitting(processed_data)

# standardization
train_data, valid_data, test, test_data, test_normal_data, test_anomaly_data, all_merged = rhrad.standardization(train, valid, processed_data)

#  Create tensor datasets from a dataframe
train_dataset, seq_len, n_features = rhrad.create_dataset(train_data)
val_dataset, _, _ = rhrad.create_dataset(valid_data)
test_dataset, _, _ = rhrad.create_dataset(test_data)
test_normal_dataset, _, _ = rhrad.create_dataset(test_normal_data)
test_anomaly_dataset, _, _ = rhrad.create_dataset(test_anomaly_data)
all_merged_dataset, _, _ = rhrad.create_dataset(all_merged)

# set CPU or GPU
device = torch.device("cpu")

# run LAAD on train and validation datasets
model = RecurrentAutoencoder(seq_len, n_features, embedding_dim=128)
model = model.to(device)
model, history = rhrad.train_model(model, train_dataset, val_dataset, n_epochs=60)
rhrad.visualize_loss(history)
rhrad.save_model(model)

# evaluate train dataset
_, losses = rhrad.predict(model, train_dataset)
rhrad.predictions_loss_train(losses, train_dataset)
THRESHOLD = rhrad.anomaly_threshold(losses)

# evaluate test normal and anomaly datasets
_, losses = rhrad.predict(model, test_normal_dataset)
rhrad.predictions_loss_test_normal(losses, test_normal_dataset)
_, losses = rhrad.predict(model, test_anomaly_dataset)
rhrad.predictions_loss_test_anomaly(losses, test_anomaly_dataset)

# evaluate test dataset
predictions, pred_losses = rhrad.predict(model, test_dataset)
rhrad.predictions_loss_test_anomaly(losses, test_dataset)


# save anomalies
anomalies = rhrad.save_anomalies(test)

# evaluate complete dataset
predictions, pred_losses = rhrad.predict(model, all_merged_dataset)
all_anomalies = rhrad.evaluate_complete_dataset(all_merged, THRESHOLD)

# metrics
rhrad.visualize_complete_dataset(all_anomalies)
TP, FP, TN, FN = rhrad.metrics_1(all_anomalies, test_normal_data, symptom_date_before_7, symptom_date_after_21)
sensitivity, specificity, precision, recall = rhrad.metrics_2(TP, FP, TN, FN)

# ROC plot
roc_input = rhrad.roc_input(all_anomalies, test_normal_data, symptom_date_before_7, symptom_date_after_21)
auc = rhrad.roc_plot(roc_input)
rhrad.save_metrics(TP, FP, TN, FN, sensitivity, specificity, precision, recall, auc)

############ ********************* ---------------- END -----------------------**************** ###############
