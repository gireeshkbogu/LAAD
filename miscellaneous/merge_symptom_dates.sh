# in healthy change dates in time columns to 00:00:00
# add header type,id,datetime

cat results/1_covid_positive/*_data_split_dates.csv |awk '{print "COVID-19,"$1","$4" 00:00:00"}' |grep -v sym  > symptom_dates_covid19.csv
cat results/2_non_covid/*_data_split_dates.csv |awk '{print "Non-COVID-19,"$1","$4" 00:00:00"}' |grep -v sym  > symptom_dates_noncovid19.csv
cat results/3_healthy/*_data_split_dates.csv |awk '{print "Healthy,"$1","$4" "$5}' |grep -v sym  > symptom_dates_healthy.csv

cat symptom_dates_covid19.csv >> symptom_dates_all.csv
cat symptom_dates_noncovid19.csv >> symptom_dates_all.csv
cat symptom_dates_healthy.csv >> symptom_dates_all.csv 


