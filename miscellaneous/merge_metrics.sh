# add this header  -   type,id,TP,FP,TN,FN,Sensitivity,Specificity,PPV,NPV,Precision,Recall,Fbeta
# fill empty cells with NA

cat results/1_covid_positive/*_metrics.csv |grep -v FN |tr -d ''\' |tr -d '[' |tr -d ']' |tr ',' '\t' | sort -nrk1 |awk '!a[$1]++'  |sort -k 1n | awk '{print "COVID-19,"$0}' |tr '\t' ',' > all_metrics.csv
cat results/2_non_covid/*_metrics.csv |grep -v FN |tr -d ''\' |tr -d '[' |tr -d ']' |tr ',' '\t' | sort -nrk1 |awk '!a[$1]++'  |sort -k 1n | awk '{print "Non-COVID-19,"$0}' |tr '\t' ',' >> all_metrics.csv
cat results/3_healthy/*_metrics.csv |grep -v FN |tr -d ''\' |tr -d '[' |tr -d ']' |tr ',' '\t' | sort -nrk1 |awk '!a[$1]++'  |sort -k 1n | awk '{print "Healthy,"$0}' |tr '\t' ',' >> all_metrics.csv

