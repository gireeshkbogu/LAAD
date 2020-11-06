# LAAD
LSTM-based Autoencoder Anomaly Detection (LAAD) is primarily developed to detect abnormal resting heart rate (RHR) during the COVID-19 period. 


### Usage

```
python laad_RHR.py  --heart_rate data/ASFODQR_hr.csv --steps data/ASFODQR_steps.csv --myphd_id ASFODQR --symptom_date 2024-08-14
python laad_multi_sensor.py  --heart_rate data/ASFODQR_hr.csv --steps data/ASFODQR_steps.csv --sleep data/ASFODQR_sleep.csv --myphd_id ASFODQR --symptom_date 2024-08-14
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



### Comparing RHR data versus multi-sensor results

#### RHR

<p align="middle">
<img width="377" alt="s_training" src="https://user-images.githubusercontent.com/3885659/98371029-5b576b80-1ff0-11eb-89ab-c002d3ea78ba.png">
<img width="373" alt="s_pred_loss" src="https://user-images.githubusercontent.com/3885659/98371036-5db9c580-1ff0-11eb-9988-d3af77d5ef4d.png">
<br/><br/>
<img width="904" alt="s_results" src="https://user-images.githubusercontent.com/3885659/98371058-67432d80-1ff0-11eb-8802-e28d49a2391e.png">
<br/><br/>
<img width="910" alt="s_aomalies" src="https://user-images.githubusercontent.com/3885659/98371063-6ad6b480-1ff0-11eb-9b01-ba0aa6510724.png">
<br/><br/>
<img width="678" alt="s_metrics" src="https://user-images.githubusercontent.com/3885659/98371069-6e6a3b80-1ff0-11eb-8f22-4a52834e8844.png">
<br/><br/>
<img width="330" alt="s_roc_plot" src="https://user-images.githubusercontent.com/3885659/98371076-71652c00-1ff0-11eb-988e-7d83f1d27dcc.png">
</p>

#### Multi-sensor (RHR, Steps, Sleep)

<p align="middle">
<img width="374" alt="ms_training" src="https://user-images.githubusercontent.com/3885659/98368897-f0586580-1fec-11eb-8990-44c87eb74e1c.png">
<img width="365" alt="ms_pred_loss" src="https://user-images.githubusercontent.com/3885659/98369816-67422e00-1fee-11eb-9570-9df4d650b4b6.png">
<br/><br/>
<img width="891" alt="ms_data" src="https://user-images.githubusercontent.com/3885659/98367610-c43be500-1fea-11eb-9be8-ade413f2d71e.png">
<br/><br/>
<img width="909" alt="ms_anomalies" src="https://user-images.githubusercontent.com/3885659/98367705-f3525680-1fea-11eb-9af3-3ea5ce6e3381.png">
<br/><br/>
<img width="680" alt="ms_metrics" src="https://user-images.githubusercontent.com/3885659/98367735-fe0ceb80-1fea-11eb-9dbb-851dcbd952e1.png">
<br/><br/>  
<img width="327" alt="ms_roc" src="https://user-images.githubusercontent.com/3885659/98367765-0a914400-1feb-11eb-8927-e6878191e6be.png">
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
