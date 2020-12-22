# add this header to the results -  type,id,datetime,loss,threshold,anomaly,RHR

awk '{print FILENAME (NF?"\t":"") $0}' results/1_covid_positive/*_anomalies.csv  | grep -v RHR > rm1
awk '{print FILENAME (NF?"\t":"") $0}' results/2_non_covid/*_anomalies.csv  | grep -v RHR >> rm1
awk '{print FILENAME (NF?"\t":"") $0}' results/3_healthy/*_anomalies.csv  | grep -v RHR >> rm1
tr '/' '\t' <rm1| sed  's/_anomalies.csv//g' |sed 's/1_covid_positive/COVID-19/g' |awk '{print $2","$3","$4" "$5}' > rm2
tr '/' '\t' <rm2| sed  's/_anomalies.csv//g' |sed 's/2_non_covid/Non-COVID-19/g' > rm3
tr '/' '\t' <rm3| sed  's/_anomalies.csv//g' |sed 's/3_healthy/Healthy/g'  > all_anomalies.csv
rm rm1 rm2 rm3

grep COVID all_anomalies.csv|grep -v Non > all_covid_positive_anomalies.csv
grep Non all_anomalies.csv > all_non_covid_anomalies.csv
grep Healthy all_anomalies.csv > all_healthy_anomalies.csv


