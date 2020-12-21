# LAAD
LSTM-based Autoencoder Anomaly Detection (LAAD) is primarily developed to detect abnormal resting heart rate (RHR) during the COVID-19 period. 


### Usage

```
python laad_covid19.py  --heart_rate data/ASFODQR_hr.csv --steps data/ASFODQR_steps.csv --myphd_id ASFODQR --symptom_date 2024-08-14
```



### LAAD architecutre

<p align="center">
<img width="800" alt="LAAD" src="https://user-images.githubusercontent.com/3885659/102735228-b4583600-42f6-11eb-9c2f-5af2ae614dab.png">
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

https://github.com/uchidalab/time_series_augmentation

https://github.com/terryum/Data-Augmentation-For-Wearable-Sensor-Data


(1) What defines an efficacious COVID-19 vaccine? A review of the challenges assessing the clinical efficacy of vaccines against SARS-CoV-2 Susanne H Hodgson, DPhil, The Lancet Infectious Diseases

(2) Detecting Mobile Traffic Anomalies through Physical Control Channel Fingerprinting: a Deep Semi-supervised Approach, HOANG DUY TRINH1,IEEE

(3) Case Study: Prolonged infectious SARS-CoV-2 shedding from an asymptomatic immunocompromised cancer patient, Avanzato et.al., Cell

(4) Temporal dynamics in viral shedding and transmissibility of COVID-19, Xi He et.al., Nat Medicine
(5) Clinical Course and Molecular Viral Shedding Among Asymptomatic and Symptomatic Patients With SARS-CoV-2 Infection in a Community Treatment Center in the Republic of Korea Seungjae Lee

(6) Asymptomatic Transmission, the Achilles’ Heel of Current Strategies to Control Covid-19 List of authors. Monica Gandhi et.al., NEJM
