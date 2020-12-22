# add this header to the results -  type,id,datetime,delta_RHR

awk '{print FILENAME (NF?"\t":"") $0}' results/1_covid_positive/*_delta_RHR.csv  | grep -v index > rm1
awk '{print FILENAME (NF?"\t":"") $0}' results/2_non_covid/*_delta_RHR.csv  | grep -v index >> rm1
awk '{print FILENAME (NF?"\t":"") $0}' results/3_healthy/*_delta_RHR.csv  | grep -v index >> rm1

tr '/' '\t' <rm1| sed  's/_delta_RHR.csv//g' |sed 's/1_covid_positive/COVID-19/g' |awk '{print $2","$3","$4" "$5}' > rm2
tr '/' '\t' <rm2| sed  's/_delta_RHR.csv//g' |sed 's/2_non_covid/Non-COVID-19/g' > rm3
tr '/' '\t' <rm3| sed  's/_delta_RHR.csv//g' |sed 's/3_healthy/Healthy/g' > all_delta_RHR.csv

rm rm1 rm2 rm3
