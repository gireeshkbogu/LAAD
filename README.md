# LAAD
LSTM-based Autoencoder Anomaly Detection. 

LAAD is primarily developed to detect abnormal resting heart rate (RHR) during the COVID-19 period. 

Data splitting:

Data is split into train and test
 - Days prior COVID-19 symptoms = Train
 - Days proximal COVID-19 symptoms = Test
     - Days proximal to COVID-19 symptoms excluding pre-symptomatic and recovery days = Test_normal
     - Days proximal to COVID-19 symptoms with only pre-symptomatic and symptomatic days = Test_anomaly

If on average the dataset is 2 months long:
   - the train data will first 20 days and the later would be tested assuming the symptom date is in the later month. 

   - Test data is further split into anomaly and normal
        - 7-0-21 = test_anomaly (14 days, 7-0 = pre-symptomatic, 0-21 = symptomatic)
        - < 7 | > 21 = test_normal ( ~26 days, < 7 = downstream of pre-symptomatic, > 21 = upstream of symtpomatic)

Metrics:

    If anomaly sequence prediction falls in 7-0-21 = TP
    True positives (TP) are the number of anomalous days that are correctly identified as anomalous.
    
    If anomaly sequence prediction falls out of 7-0-21 = FN
    False negatives (FN) are the no.of anomalous days that are incorrectly identified as normal.
    
    If normal sequence prediction falls in 7-0-21 = TN
    True negatives (TN) are the number of normal days that are correctly identified as normal.
    
    If normal sequence prediction falls out of 7-0-21 = FP
    False positives (FP) are the no.of normal days that are incorrectly identified as anomalous. 


References:

https://github.com/shobrook/sequitur

https://github.com/curiousily/Getting-Things-Done-with-Pytorch

https://github.com/BLarzalere/LSTM-Autoencoder-for-Anomaly-Detection

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

http://karpathy.github.io/2015/05/21/rnn-effectiveness/

https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L14_intro-rnn-part2_slides.pdf

https://arxiv.org/pdf/1607.00148.pdf (LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection)
