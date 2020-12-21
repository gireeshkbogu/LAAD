## Date: 4 Dec 2020 
## Author: Gireesh Bogu
## Contact: gireesh.bogu@stanford.edu

#################################################################

library(ggplot2)
library(shiny)
library(plotly)
library(htmlwidgets)
library(tidyverse)
library("ggpubr")
library(lubridate)
library(cowplot)
library(pheatmap)


setwd("~/Desktop/laad/figures")

# anomaly scores -----------------------

# global figure --------------------------------------------------------------------------
anom1 <- read.table("all_covid_positive_anomalies_new.csv", sep=',', header = T)

# add dates
dates1 <- read.table("symptom_dates.txt", sep=',', head=T)

dis2 <- left_join(dates1, anom1)
dis2 <- dis2  %>% filter(datetime != 'NA') 
dis2$symptom_date <- ymd_hms(dis2$symptom_date)  
dis2$datetime <- ymd_hms(dis2$datetime)   

dis3_laad <- dis2 %>% group_by(id)  %>% 
  mutate(Deltas = as.double.difftime(datetime - symptom_date, units="days"))

write.table(dis3_laad, "LAAD_deltas_covid_positive.txt", sep=",")

dis3_laad_1 <- dis3_laad  %>% 
  filter(type =="COVID-19") %>% 
  group_by(id) %>% 
  #filter(Deltas >= -50 | Deltas <= 50  | is.na(Deltas)) %>% 
  mutate(loss_scaled=scales::rescale(loss,to=c(0, 1)))

cbPalette = c("#00AFBB", "#E7B800", "#DF4E07")


dis3_laad_1$id = factor(dis3_laad_1$id, 
                        levels=c("AJWW3IY",
                                 "A7EM0B6",
                                 "ASFODQR",
                                 "AA2KP1S",
                                 "AMV7EQF",
                                 "AYWIEKR",
                                 "A0NVTRV",
                                 "AKXN5ZZ",
                                 "AUY8KYW",
                                 "A4E0D03",
                                 "AIFDJZB",
                                 "AAXAA7Z",
                                 "A0VFT1N",
                                 "AQC0L71",
                                 "APGIB2T",
                                 "A1K5DRI",
                                 "A4G0044",
                                 "AHYIJDV",
                                 "AOYM4KG",
                                 "A3OU183",
                                 "AURCTAK",
                                 "AX6281V",
                                 "AV2GF3B",
                                 "AS2MVDL",
                                 'AYEFCWQ'
                        ))


# new
dis3_laad_1$id = factor(dis3_laad_1$id, 
                        levels=c("AJWW3IY",
                                 "A7EM0B6",
                                 "ASFODQR",
                                 "A4E0D03",
                                 "AKXN5ZZ",
                                 "AUY8KYW",
                                 "A0NVTRV",
                                 "AQC0L71",
                                 "APGIB2T",
                                 "AHYIJDV",
                                 "AIFDJZB",
                                 "AAXAA7Z",
                                 "AX6281V",
                                 "A1K5DRI",
                                 "AV2GF3B",
                                 "AS2MVDL",
                                 "AOYM4KG",
                                 "A4G0044",
                                 "AA2KP1S",
                                 "AYWIEKR",
                                 "A3OU183",
                                 "AURCTAK",
                                 "A0VFT1N",
                                 "AMV7EQF",
                                 'AYEFCWQ'
                        ))

dis3_laad_1 %>% 
  ggplot(aes(x=Deltas, y=loss)) + 
  geom_bar(stat="identity", position="identity", fill="#009E73",  width=0.1)+
  facet_grid(id~., scales = "free",  switch="both") +
  theme_pubr()+
  #scale_x_continuous(limits = c(-63, 63),breaks = seq(-63, 63, by = 7))+
  #scale_x_continuous(limits = c(-14, 28),breaks = seq(-14, 28, by = 1))+
  theme(axis.text.y=element_text(size=0))+ 
  theme(strip.text.y.left = element_text(size = 8, angle = 0))+
  scale_fill_manual(values=cbPalette)+
  theme(axis.text.x = element_text(size=8, angle = 90))+
  geom_hline(yintercept=0, colour="black", linetype="dashed", size=0.1)+ #takes care of baseline szie
  geom_vline(xintercept=-7, colour="orange", linetype="dashed", size=0.8)+ # pre-symptomatic 
  geom_vline(xintercept=0, colour="red", linetype="dashed", size=0.8)+ # symptomatic 
  geom_vline(xintercept=21, colour="purple", linetype="dashed", size=0.8)+ # post-symptomatic 
  ylab("")+ xlab("")+ ggtitle("")+
  theme(axis.ticks.y.left = element_blank(),panel.spacing.x=unit(0, "lines") , panel.spacing.y=unit(0.01,"lines")) # decrease the space betweeen grids

dis3_laad_1 %>% 
  ggplot(aes(x=Deltas, y=log(loss+1))) + 
  geom_bar(stat="identity", position="identity", fill="#009E73",  width=0.1)+
  facet_grid(id~., scales = "free",  switch="both") +
  theme_pubr()+
  #scale_x_continuous(limits = c(-63, 63),breaks = seq(-63, 63, by = 7))+
  #scale_x_continuous(limits = c(-14, 28),breaks = seq(-14, 28, by = 1))+
  theme(axis.text.y=element_text(size=0))+ 
  theme(strip.text.y.left = element_text(size = 8, angle = 0))+
  scale_fill_manual(values=cbPalette)+
  theme(axis.text.x = element_text(size=8, angle = 90))+
  geom_hline(yintercept=0, colour="black", linetype="dashed", size=0.1)+ #takes care of baseline szie
  geom_vline(xintercept=-7, colour="orange", linetype="dashed", size=0.8)+ # pre-symptomatic 
  geom_vline(xintercept=0, colour="red", linetype="dashed", size=0.8)+ # symptomatic 
  geom_vline(xintercept=21, colour="purple", linetype="dashed", size=0.8)+ # post-symptomatic 
  ylab("")+ xlab("")+ ggtitle("")+
  theme(axis.ticks.y.left = element_blank(),panel.spacing.x=unit(0, "lines") , panel.spacing.y=unit(0.01,"lines")) # decrease the space betweeen grids



anomaly_scores <- dis3_laad_1 %>% 
  ggplot(aes(x=Deltas, y=log(loss+1))) + 
  geom_bar(stat="identity", position="identity", fill="#009E73",  width=0.1)+
  facet_grid(id~., scales = "free",  switch="both") +
  theme_pubr()+
  #scale_x_continuous(limits = c(-63, 63),breaks = seq(-63, 63, by = 7))+
  scale_x_continuous(limits = c(-14, 28),breaks = seq(-14, 28, by = 1))+
  theme(axis.text.y=element_text(size=0))+ 
  theme(strip.text.y.left = element_text(size = 8, angle = 0))+
  scale_fill_manual(values=cbPalette)+
  theme(axis.text.x = element_text(size=8, angle = 90))+
  geom_hline(yintercept=0, colour="black", linetype="dashed", size=0.1)+ #takes care of baseline szie
  geom_vline(xintercept=-7, colour="orange", linetype="dashed", size=0.8)+ # pre-symptomatic 
  geom_vline(xintercept=0, colour="red", linetype="dashed", size=0.8)+ # symptomatic 
  geom_vline(xintercept=21, colour="purple", linetype="dashed", size=0.8)+ # post-symptomatic 
  ylab("")+ xlab("")+ ggtitle("")+
  theme(axis.ticks.y.left = element_blank(),panel.spacing.x=unit(0, "lines") , panel.spacing.y=unit(0.01,"lines")) # decrease the space betweeen grids



# metrics -------------------------

metrics1 <- read.table("all_metrics.txt", sep='\t', header = T)

metrics <- ggviolin(metrics1, x = "type", y = "Fbeta", fill = "type",
         palette = c("#009E73", "grey", "steelblue"),
         add = "boxplot", add.params = list(fill = "white")) +xlab("")+ylab("F-beta (0.1)") + 
  guides(fill=FALSE)  

# distance --------

anom1 <- read.table("all_anomalies.csv", sep=',', header = T)

anom1  %>% select(type,id) %>% 
distinct() %>% 
group_by(type)  %>% 
summarise(count = n()) 


# add dates
dates1 <- read.table("symptom_dates_all.txt", sep=',', head=T)

dis2 <- left_join(dates1, anom1)
dis2$symptom_date <- ymd_hms(dis2$symptom_date)  
dis2$datetime <- ymd_hms(dis2$datetime)   

dis3_laad_1 <- dis2 %>% group_by(id)  %>% 
mutate(Deltas = as.double.difftime(datetime - symptom_date, units="days"))

delta3 <- dis3_laad_1%>% 
group_by(id) %>% 
filter(Deltas >= -7 & Deltas <= 21  | is.na(Deltas))


delta4 <- delta3  %>% select(id,type,Deltas,loss) %>%   
# filtered anomaly scores during COVID-19 infectious period by takign the values between 7 days before and 21 days after symptom onset. 
# grouped patients who had anomaly signal before symptom onset as pre-symptomatic and after as post-symptomatic
# anomaly sginal strength was calculated by dividing the number of anomaly events with the total loss score (anomaly score) and further divided by 7 and multiplied by 100.
mutate(group = ifelse(Deltas <0, "Pre", "Post")) %>% 
group_by(id,type, group) %>% 
mutate("signal_count" = n()) %>% 
mutate("total_loss" = sum(loss)) %>% 
mutate("strength" = ((total_loss/signal_count)/7)*100) %>%   
mutate(group1 = ifelse(strength >6, "Strong", "Weak")) %>%
# grouped the patients who had signal before symptom onset as "Early" and either after on-set or
# before onset but with signal count less than or equal to 6  as "Late".
mutate(group2 = case_when(signal_count <= 6 | Deltas >=0 ~ "Late",
Deltas < 0 ~ "Early") )


delta5  <- delta4  %>% 
select(-loss)%>% 
filter(type=="COVID-19") %>% 
filter(signal_count >6) %>% 
# print the distance of the first detection in pre and post-symptomatic periods.
# print min if it's Early and max if it's Late
group_by(id, group) %>% 
mutate(group2_deltas=ifelse(group2=='Early', min(Deltas,na.rm = T),
ifelse(group2=='Late', min(Deltas,na.rm = T)))) %>%
distinct(group2_deltas, .keep_all=TRUE) %>% 
group_by(id) %>% 
slice(which.min(group2_deltas)) %>% 
rename(group_deltas =  Deltas) %>% 
select(-group, -group_deltas) %>%
# order by group1, group2, strength, distance
arrange(group1,group2,  desc(abs(group2_deltas)), desc(strength))


delta5  %>% group_by(type) %>% 
summarise(mean = mean(group2_deltas),
median =  median(group2_deltas)) 

delta5  %>% group_by(type, group2) %>% 
summarise(mean = mean(group2_deltas),
median =  median(group2_deltas)) 


distance <- gghistogram(delta5, x = "group2_deltas",
add = "mean", rug = TRUE,
fill = "group2", palette = c("#00AFBB", "#E7B800"),
add_density = TRUE, xlab = "Distance", ylab = "Count")


# duration ----------------------
# this file doesnn't have healthy anomalies
anom1 <- read.table("all_anomalies.csv", sep=',', header = T)

# add dates
dates1 <- read.table("symptom_dates_all.txt", sep=',', head=T)

dis2 <- left_join(dates1, anom1)
dis2$symptom_date <- ymd_hms(dis2$symptom_date)  
dis2$datetime <- ymd_hms(dis2$datetime)   

dis3_laad_1 <- dis2 %>% group_by(id)  %>% 
  mutate(Deltas = as.double.difftime(datetime - symptom_date, units="days"))


delta3 <- dis3_laad_1%>% 
  group_by(id) %>% 
  filter(Deltas >= -7 & Deltas <= 21  | is.na(Deltas))


delta1_dur_all <- delta3 %>% select(type, id)%>%  
  filter(!is.na(type)) %>% 
  group_by(id) %>% 
  mutate(count=n()) %>% 
  unique()

write.table(delta1_dur_all, "LAAD_duration_of_abnormal_RHR.txt", sep=",")

# Visualize: Specify the comparisons you want
my_comparisons <- list( c("COVID-19", "Healthy"), c("COVID-19", "Non-COVID-19"), c("Healthy", "Non-COVID-19") )
duration <- ggboxplot(delta1_dur_all, 
          x = "type", y = "count",
          color = "type", 
          palette = c("#009E73", "grey", "steelblue"),
          add = "jitter",
          add.params = list(size = 0.3, jitter = 0.2))+ 
  theme(legend.position="none") + 
  stat_compare_means(comparisons = my_comparisons, method  = 'wilcox.test')+ # Add pairwise comparisons p-value
  stat_compare_means(label.y = 700) +    # Add global p-value
  xlab("")+ylab("Duration of \nabnnormal RHR (hours)") 

# delta RHR ----------------------

anom1 <- read.table("all_anomalies.csv", sep=',', header = T)

# add dates
dates1 <- read.table("symptom_dates_all.txt", sep=',', head=T)

dis2 <- left_join(dates1, anom1)
dis2$symptom_date <- ymd_hms(dis2$symptom_date)  
dis2$datetime <- ymd_hms(dis2$datetime)   

dis3_laad_1 <- dis2 %>% group_by(id)  %>% 
  mutate(Deltas = as.double.difftime(datetime - symptom_date, units="days"))

write.table(dis3_laad_1, "LAAD_delta_RHR_all.txt", sep=",")


# delta RHR
delta1 <- read.table("all_delta_RHR.csv", sep=',', header = T)
delta1$datetime <- ymd_hms(delta1$datetime) 

# join distance
dist1 <- dis3_laad_1 %>% select(id, datetime, Deltas)
dist1$datetime <- ymd_hms(dist1$datetime) 

delta2 <- left_join(delta1, dist1, by = c("id", "datetime"))

delta3 <- delta2%>% 
  group_by(id) %>% 
  filter(Deltas >= -7 & Deltas <= 21  | is.na(Deltas))


# barplot-2 summary figure ---------------------------------------------------

delta5 <- delta3 %>% 
  group_by(id, type) %>%
  mutate(positive = delta_RHR >=0)  %>% 
  group_by(type,positive) %>% 
  summarise(mean = mean(delta_RHR),
            median = median(delta_RHR))

delta_rhr <- ggbarplot(delta5, x = "type", y = "median",
          fill = "positive",
          color = "white",
          palette = "jco",
          #x.text.angle = 90,
          ylab = "Delta RHR (Median)",
          xlab = " ",
          label = round(delta5$median,2),
          lab.pos = "in",
          font.label = list(size = 9,vjust = 0.1)) 




# stitch all

# metrics matrix for annomaly scores

matrix1 <- read.csv('metrics_covid19.txt', head=T, sep='\t')

matrix1$id = factor(matrix1$id, 
                        levels=c("AJWW3IY",
                                 "A7EM0B6",
                                 "ASFODQR",
                                 "AA2KP1S",
                                 "AMV7EQF",
                                 "AYWIEKR",
                                 "A0NVTRV",
                                 "AKXN5ZZ",
                                 "AUY8KYW",
                                 "A4E0D03",
                                 "AIFDJZB",
                                 "AAXAA7Z",
                                 "A0VFT1N",
                                 "AQC0L71",
                                 "APGIB2T",
                                 "A1K5DRI",
                                 "A4G0044",
                                 "AHYIJDV",
                                 "AOYM4KG",
                                 "A3OU183",
                                 "AURCTAK",
                                 "AX6281V",
                                 "AV2GF3B",
                                 "AS2MVDL",
                                 'AYEFCWQ'
                        ))

ggtexttable(matrix1, rows = NULL)


plot_grid(metrics, distance, duration, delta_rhr, labels = c('c', 'd', 'e', 'f'), ncol = 4)


ggarrange(anomaly_scores,                                               
          ggarrange(metrics, 
                    distance, 
                    duration, 
                    delta_rhr, 
                    ncol = 4, 
                    labels = c('b', 'c', 'd', 'e')), 
          nrow = 2, 
          labels = c("a")) 


