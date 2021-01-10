# add header id,type,start_date,symptom_date_before_20,symptom_date_before_7,symptom_date_before_10,symptom_date_after_21,end_date
#cat results/1_covid_positive/*_data_split_dates.csv |grep -v end |awk '{print $1",""COVID-19,"$2","$4","$6","$8","$10","$12}' > size_dates.csv
#cat results/2_non_covid/*_data_split_dates.csv |grep -v end |awk '{print $1",""Non-COVID-19,"$2","$4","$6","$8","$10","$12}' >> size_dates.csv
#cat results/3_healthy/*_data_split_dates.csv |grep -v end |awk '{print $1",""Healthy,"$2","$4","$6","$8","$10","$12}' >> size_dates.csv

# add header - id,type,train,test,test_noninfectious,test_infectious
cat results/1_covid_positive/*_data_size.csv  |grep -v anoma |tr -d '(' |tr ','  '\t' |tr -d ')'  |awk '{print $1",""COVID-19,"$2","$4","$6","$8}' > data_size.csv
cat results/2_non_covid/*_data_size.csv  |grep -v anoma |tr -d '(' |tr ','  '\t' |tr -d ')'  |awk '{print $1",""Non-COVID-19,"$2","$4","$6","$8}' >> data_size.csv
cat results/3_healthy/*_data_size.csv  |grep -v anoma |tr -d '(' |tr ','  '\t' |tr -d ')'  |awk '{print $1",""Healthy,"$2","$4","$6","$8}' >> data_size.csv




