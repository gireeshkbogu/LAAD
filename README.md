# LAAD
LSTM-based Autoencoder Anomaly Detection (LAAD) is primarily developed to detect abnormal resting heart rate (RHR) during the Coronavirus (SARS-CoV-2) infectious period.. 


### Usage

```
python laad_covid19.py  --heart_rate data/ASFODQR_hr.csv --steps data/ASFODQR_steps.csv --myphd_id ASFODQR --symptom_date 2024-08-14
```



### LAAD architecture

<p align="center">
<img width="800" alt="LAAD" src="https://user-images.githubusercontent.com/3885659/102735228-b4583600-42f6-11eb-9c2f-5af2ae614dab.png">
</p>



### Results

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

