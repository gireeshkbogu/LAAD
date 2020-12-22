cat results/1_covid_positive/*_data_split_dates.csv |grep -v end |awk '{print $1",""COVID-19,"$2","$4","$6","$8","$10","$12}' > size.csv
cat results/2_non_covid/*_data_split_dates.csv |grep -v end |awk '{print $1",""Non-COVID-19,"$2","$4","$6","$8","$10","$12}' >> size.csv
cat results/3_healthy/*_data_split_dates.csv |grep -v end |awk '{print $1",""Healthy,"$2","$4","$6","$8","$10","$12}' >> size.csv


