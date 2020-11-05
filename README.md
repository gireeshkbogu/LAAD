# LAAD
LSTM-based Autoencoder Anomaly Detection (LAAD) is primarily developed to detect abnormal resting heart rate (RHR) during the COVID-19 period. 


### Usage

```
python laad.py  --heart_rate data/ASFODQR_hr.csv --steps data/ASFODQR_steps.csv --myphd_id ASFODQR --symptom_date 2024-08-14
```

### COVID-19 timeline graphic 
COVID-19 virus exposure, symptom onset and testing (1). Viral shedding begins 5-6 days before symptom onset  in COVID-19 cases and continues for 21 days;  and peaks at 2 days before and 1 day after symptom onset (4). Asymptomatic patients viral load was similar to that in symptomatic patients (5). There are exceptions like 70 days of virial shredding in asymptomatic and immunnocompromised 71 old women (3). We don't have asymptomatic cohort (6).


![timeline](https://user-images.githubusercontent.com/3885659/98132147-e9a4e380-1e70-11eb-9185-16d4406422a3.jpeg)


### Data splitting: 
We split the data using the no.of days during general trend of the COVID-19 virus exposure and symptom onset as shown in Hodgson et.al., study (1).

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



### Comparing single-sensor data versus multi-sensor results

#### Single-sensor (RHR)

<p align="center">
<img width="893" alt="single-sensor" src="https://user-images.githubusercontent.com/3885659/98157243-eec45b80-1e8d-11eb-93f5-303e313f97a5.png">
<img width="902" alt="single-sensor" src="https://user-images.githubusercontent.com/3885659/98171758-1a9f0b80-1ea5-11eb-900f-e06214be8762.png">
<img width="685" alt="metrics_single_sensor" src="https://user-images.githubusercontent.com/3885659/98172764-c4cb6300-1ea6-11eb-9472-e699c9da6fd3.png">
<img width="329" alt="roc_single_sennsor" src="https://user-images.githubusercontent.com/3885659/98172776-c8f78080-1ea6-11eb-8cbd-31666502ba81.png">
</p>


#### Multi-sensor (RHR, Steps, Sleep)

<p align="center">
<img width="907" alt="multi-sensor" src="https://user-images.githubusercontent.com/3885659/98157247-eff58880-1e8d-11eb-8176-62afe9ddc9a4.png">
<img width="901" alt="mulit-sensor" src="https://user-images.githubusercontent.com/3885659/98171762-1bd03880-1ea5-11eb-9735-9c09b405e40f.png">
<img width="682" alt="metrics_multi-sensor" src="https://user-images.githubusercontent.com/3885659/98172767-c5fc9000-1ea6-11eb-9581-b4264a415a9b.png">
<img width="327" alt="roc_muliti-sensor" src="https://user-images.githubusercontent.com/3885659/98172783-cc8b0780-1ea6-11eb-950a-b991a12553de.png">
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

(3) Case Study: Prolonged infectious SARS-CoV-2 shedding from an asymptomatic immunocompromised cancer patient, Avanzato et.al., Cell

(4) Temporal dynamics in viral shedding and transmissibility of COVID-19, Xi He et.al., Nat Medicine
(5) Clinical Course and Molecular Viral Shedding Among Asymptomatic and Symptomatic Patients With SARS-CoV-2 Infection in a Community Treatment Center in the Republic of Korea Seungjae Lee

(6) Asymptomatic Transmission, the Achilles’ Heel of Current Strategies to Control Covid-19 List of authors. Monica Gandhi et.al., NEJM
