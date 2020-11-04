# LAAD
LSTM-based Autoencoder Anomaly Detection (LAAD) is primarily developed to detect abnormal resting heart rate (RHR) during the COVID-19 period. 


### Usage

```
python laad.py  --heart_rate data/ASFODQR_hr.csv --steps data/ASFODQR_steps.csv --myphd_id ASFODQR --symptom_date 2024-08-14
```


### Results
<p align="center">
<img width="1150" alt="results" src="https://user-images.githubusercontent.com/3885659/98135666-f3c8e100-1e74-11eb-9fb9-39321c375442.png">
</p>

<p align="center">
<img width="175" alt="metrics" src="https://user-images.githubusercontent.com/3885659/98135949-45716b80-1e75-11eb-9b9c-7fbaeb02fedd.png"> 
</p>

<p align="center">
<img width="323" alt="sn vs sp" src="https://user-images.githubusercontent.com/3885659/98135967-4904f280-1e75-11eb-8b39-4dcfe1de9956.png"> 
</p>

<p align="center">
<img width="331" alt="roc" src="https://user-images.githubusercontent.com/3885659/98135989-4d311000-1e75-11eb-978e-555aeca7f749.png">
</p>

### Timeline graphic 
COVID-19 virus exposure, symptom onset and testing (1)


![timeline](https://user-images.githubusercontent.com/3885659/98132147-e9a4e380-1e70-11eb-9185-16d4406422a3.jpeg)


### Data splitting: 
We split the data using the no.of days during virus exposure and symptom onset as shown above

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

### Metrics:

    If anomaly sequence prediction falls in 7-0-21 = TP
    True positives (TP) are the number of anomalous days that are correctly identified as anomalous.
    
    If anomaly sequence prediction falls out of 7-0-21 = FN
    False negatives (FN) are the no.of anomalous days that are incorrectly identified as normal.
    
    If normal sequence prediction falls in 7-0-21 = TN
    True negatives (TN) are the number of normal days that are correctly identified as normal.
    
    If normal sequence prediction falls out of 7-0-21 = FP
    False positives (FP) are the no.of normal days that are incorrectly identified as anomalous. 


### LAAD architecutre (2)

<p align="center">
<img width="840" alt="pipeline" src="https://user-images.githubusercontent.com/3885659/98139807-781d6300-1e79-11eb-8eff-4313b0cef37b.png">
</p>

<p align="center">
<img width="464" alt="lstm" src="https://user-images.githubusercontent.com/3885659/98139812-794e9000-1e79-11eb-8790-a5e3657afc34.png">
</p>



### Comparing single-sensor data versus multi-sensor

#### Single-sensor (RHR)

<p align="center">
<img width="893" alt="single-sensor" src="https://user-images.githubusercontent.com/3885659/98157243-eec45b80-1e8d-11eb-93f5-303e313f97a5.png">
</p>


#### Multi-sensor (RHR, Steps, Sleep)

<p align="center">
<img width="907" alt="multi-sensor" src="https://user-images.githubusercontent.com/3885659/98157247-eff58880-1e8d-11eb-8176-62afe9ddc9a4.png">
</p>


### References:

https://github.com/shobrook/sequitur

https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html

https://github.com/curiousily/Getting-Things-Done-with-Pytorch

https://github.com/BLarzalere/LSTM-Autoencoder-for-Anomaly-Detection

http://colah.github.io/posts/2015-08-Understanding-LSTMs/

http://karpathy.github.io/2015/05/21/rnn-effectiveness/

https://sebastianraschka.com/pdf/lecture-notes/stat479ss19/L14_intro-rnn-part2_slides.pdf

https://arxiv.org/pdf/1607.00148.pdf (LSTM-based Encoder-Decoder for Multi-sensor Anomaly Detection)



(1) What defines an efficacious COVID-19 vaccine? A review of the challenges assessing the clinical efficacy of vaccines against SARS-CoV-2 Susanne H Hodgson, DPhil, The Lancet Infectious Diseases

(2) Detecting Mobile Traffic Anomalies through Physical Control Channel Fingerprinting: a Deep Semi-supervised Approach, HOANG DUY TRINH1,IEEE

