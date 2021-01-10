# Aim: all figures
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


# ANOMALY SCORES -------------------------------------------------------------------------

# 1. figure 2b - anomaly scores - covid-19 ---------------------------------------
anom1 <- read.table("all_covid_positive_anomalies.csv", sep=',', header = T)

# add dates
dates1 <- read.table("symptom_dates_all.csv", sep=',', head=T)

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
                        levels=c("A7EM0B6",
                                 "ASFODQR",
                                 "AJWW3IY",
                                 "AAXAA7Z",
                                 "A4E0D03",
                                 "AKXN5ZZ",
                                 "AIFDJZB",
                                 "AOYM4KG",
                                 "A1K5DRI",
                                 "APGIB2T",
                                 "AUY8KYW",
                                 "A0NVTRV",
                                 "AHYIJDV",
                                 "AQC0L71",
                                 "AV2GF3B",
                                 "AS2MVDL",
                                 "A4G0044",
                                 "AYWIEKR",
                                 "AX6281V",
                                 "AA2KP1S",
                                 "A3OU183",
                                 "AURCTAK",
                                 "A0VFT1N",
                                 "AMV7EQF",
                                 'AYEFCWQ'
                        ))


anomaly_scores_covid <- dis3_laad_1 %>% 
  ggplot(aes(x=Deltas, y=log(loss+1))) + 
  geom_bar(stat="identity", position="identity", fill="#009E73",  width=0.1)+
  facet_grid(id~., scales = "free",  switch="both") +
  theme_pubr()+
  #scale_x_continuous(limits = c(-63, 63),breaks = seq(-63, 63, by = 7))+
  #scale_x_continuous(limits = c(-14, 28),breaks = seq(-14, 28, by = 1))+
  scale_x_continuous(limits = c(-30, 60),breaks = seq(-30, 60, by = 10))+
  theme(axis.text.y=element_text(size=0))+ 
  theme(strip.text.y.left = element_text(size = 8, angle = 0))+
  scale_fill_manual(values=cbPalette)+
  theme(axis.text.x = element_text(size=8, angle = 90))+
  geom_hline(yintercept=0, colour="black", linetype="dashed", size=0.1)+ #takes care of baseline szie
  geom_vline(xintercept=-7, colour="orange", linetype="dashed", size=0.8, alpha= 0.6)+ # pre-symptomatic 
  geom_vline(xintercept=0, colour="red", linetype="dashed", size=0.8, alpha= 0.6)+ # symptomatic 
  geom_vline(xintercept=21, colour="purple", linetype="dashed", size=0.8, alpha= 0.6)+ # post-symptomatic 
  ylab("")+ xlab("")+ ggtitle("")+
  theme(axis.ticks.y.left = element_blank(),panel.spacing.x=unit(0, "lines") , panel.spacing.y=unit(0.01,"lines")) # decrease the space betweeen grids

anomaly_scores_covid


# 2. supp fig -  anomaly scores of non-covid19 --------------------------------------------

anom1 <- read.table("all_non_covid_anomalies.csv", sep=',', header = T)

# add dates
dates1 <- read.table("symptom_dates_all.csv", sep=',', head=T)

dis2 <- left_join(dates1, anom1)
dis2 <- dis2  %>% filter(datetime != 'NA') 
dis2$symptom_date <- ymd_hms(dis2$symptom_date)  
dis2$datetime <- ymd_hms(dis2$datetime)   

dis3_laad <- dis2 %>% group_by(id)  %>% 
  mutate(Deltas = as.double.difftime(datetime - symptom_date, units="days"))

write.table(dis3_laad, "LAAD_deltas_non_covid.txt", sep=",")

dis3_laad_1 <- dis3_laad  %>% 
  group_by(id) %>% 
  #filter(Deltas >= -500 | Deltas <= 500  | is.na(Deltas)) %>% 
  mutate(loss_scaled=scales::rescale(loss,to=c(0, 1)))

cbPalette = c("#00AFBB", "#E7B800", "#DF4E07")



dis3_laad_1$id = factor(dis3_laad_1$id, 
                        levels=c("AA0HAI1_2",
                                 "AA0HAI1_3",
                                 "AA0HAI1_1",
                                 "A35BJNV",
                                 "AR4FPCC",
                                 "A0KX894",
                                 "AD77K91",
                                 "AEOBCFJ",
                                 "AK7YRBU",
                                 'ALKAXMZ',
                                 'AFHOHOM',
                                 "AOGFRXL",
                                 'AUILKHG'
                        ))

anomaly_scores_noncovid <- dis3_laad_1 %>% 
  ggplot(aes(x=Deltas, y=loss_scaled)) + 
  geom_bar(stat="identity", position="identity", fill="steelblue",  width=0.1)+
  facet_grid(id~., scales = "free", space="free", switch="both") +
  theme_pubr()+
  #scale_x_continuous(limits = c(-56, 56),breaks = seq(-56, 56, by = 7))+
  #scale_x_continuous(limits = c(-14, 28),breaks = seq(-14, 28, by = 1))+
  scale_x_continuous(limits = c(-20, 50),breaks = seq(-20, 50, by = 10))+
  theme(axis.text.y=element_text(size=0))+ 
  theme(strip.text.y.left = element_text(size = 8, angle = 0))+
  scale_fill_manual(values=cbPalette)+
  theme(axis.text.x = element_text(size=8, angle = 90))+
  geom_hline(yintercept=0, colour="black", linetype="dashed", size=0.1)+ #takes care of baseline szie
  geom_vline(xintercept=-7, colour="orange", linetype="dashed", size=0.8)+ # pre-symptomatic 
  geom_vline(xintercept=0, colour="red", linetype="dashed", size=0.8)+ # symptomatic 
  geom_vline(xintercept=21, colour="purple", linetype="dashed", size=0.8)+ # post-symptomatic 
  ylab("")+ xlab("")+ 
  theme(axis.ticks.y.left = element_blank(),panel.spacing.x=unit(0, "lines") , panel.spacing.y=unit(0.01,"lines")) # decrease the space betweeen grids

anomaly_scores_noncovid

# 3. supp fig -  anomaly scores of healthy --------------------------------------------

anom1 <- read.table("all_healthy_anomalies.csv", sep=',', header = T)

# add dates
dates1 <- read.table("symptom_dates_all.csv", sep=',', head=T)

dis2 <- left_join(dates1, anom1)
dis2 <- dis2  %>% filter(datetime != 'NA') 
dis2$symptom_date <- ymd_hms(dis2$symptom_date)  
dis2$datetime <- ymd_hms(dis2$datetime)   

dis3_laad <- dis2 %>% group_by(id)  %>% 
  mutate(Deltas = as.double.difftime(datetime - symptom_date, units="days"))

write.table(dis3_laad, "LAAD_deltas_healthy.txt", sep=",")

dis3_laad_1 <- dis3_laad  %>% 
  group_by(id) %>% 
  #filter(Deltas >= -500 | Deltas <= 500  | is.na(Deltas)) %>% 
  mutate(loss_scaled=scales::rescale(loss,to=c(0, 1)))

cbPalette = c("#00AFBB", "#E7B800", "#DF4E07")


dis3_laad_1$id = factor(dis3_laad_1$id, 
                        levels=c("A5XL2IC",
                                 "A2XFW2N",
                                 "AW4EXXK",
                                 "AAGTWZK",
                                 "AKTGD8X",
                                 "A91HEZV",
                                 "A99ZKKW",
                                 "A45F9E6",
                                 "A4H7SNF",
                                 "AE0MQ94",
                                 "A6BUI4N",
                                 "APDJ1QP",
                                 "ATT9RR1",
                                 "AFEFA29",
                                 "A0822M0",
                                 "AYVQUF1",
                                 "AOQA85X",
                                 "AGA8XUN",
                                 "A06L7KF",
                                 "AXD3W8O",
                                 "AKV66US",
                                 "AFVAEC7",
                                 "AWA2KJK",
                                 "AY8TPMP",
                                 "A0L9BM2",
                                 "A17YCA2",
                                 "AQR8ZSS",
                                 "AZ2RYW7",
                                 "AEOHH30",
                                 "AX3KEW9",
                                 "AO20DS4",
                                 "A2P3LTM",
                                 "AXDWDEA",
                                 "AHP25OJ",
                                 "AQ25Y0L",
                                 "AAF9ACE",
                                 "ARYB2QO",
                                 "AL48GP3",
                                 "A2D7K4A",
                                 "ALZDAVZ",
                                 "AOB9SON",
                                 "AOS4BSJ",
                                 "AZKZ0AI",
                                 "A9ZG5GR",
                                 "AZ35PI5",
                                 "A6GEBIK",
                                 "AEZDKVO",
                                 "A7EAWA7",
                                 "ARFYLMK",
                                 "AFYLHG4",
                                 "ATHBFWX",
                                 "AJ0DKQ3",
                                 "AMQUHOQ",
                                 "AQ4TMLV",
                                 "ARR2IKE",
                                 "A8CBEJZ",
                                 "AGKI03N",
                                 "AWRBUQZ",
                                 "AL3KT5B",
                                 "A11V1FH",
                                 "A11SQQN",
                                 "AE2B3RH",
                                 "AXCO7I9",
                                 "A65HVGP",
                                 "AUCGUF3"
                        ))

anomaly_scores_healthy <- dis3_laad_1 %>% 
  filter(!is.na(id)) %>% 
  ggplot(aes(x=Deltas, y=loss_scaled)) + 
  geom_bar(stat="identity", position="identity", fill="grey",  width=0.1)+
  facet_grid(id~., scales = "free", space="free", switch="both") +
  theme_pubr()+
  #scale_x_continuous(limits = c(-56, 56),breaks = seq(-56, 56, by = 7))+
  #scale_x_continuous(limits = c(-14, 28),breaks = seq(-14, 28, by = 1))+
  scale_x_continuous(limits = c(-70, 70),breaks = seq(-70, 70, by = 10))+
  theme(axis.text.y=element_text(size=0))+ 
  theme(strip.text.y.left = element_text(size = 8, angle = 0))+
  scale_fill_manual(values=cbPalette)+
  theme(axis.text.x = element_text(size=8, angle = 90))+
  geom_hline(yintercept=0, colour="black", linetype="dashed", size=0.1)+ #takes care of baseline szie
  geom_vline(xintercept=-7, colour="orange", linetype="dashed", size=0.8)+ # pre-symptomatic 
  geom_vline(xintercept=0, colour="red", linetype="dashed", size=0.8)+ # symptomatic 
  geom_vline(xintercept=21, colour="purple", linetype="dashed", size=0.8)+ # post-symptomatic 
  ylab("")+ xlab("")+ 
  theme(axis.ticks.y.left = element_blank(),panel.spacing.x=unit(0, "lines") , panel.spacing.y=unit(0.01,"lines")) # decrease the space betweeen grids

anomaly_scores_healthy

# Metrics -------------------------------------------------------------------------------

metrics1 <- read.table("all_metrics_table.txt", sep='\t', header = T)

precision <- ggviolin(metrics1, x = "type", y = "Precision", fill = "type",
         palette = c("#009E73", "grey", "steelblue"), alpha=0.5,
         add = "boxplot", add.params = list(fill = "white"))+ 
  guides(fill=FALSE)  

precision

recall <- ggviolin(metrics1, x = "type", y = "Recall", fill = "type",
         palette = c("#009E73", "grey", "steelblue"), alpha=0.5,
         add = "boxplot", add.params = list(fill = "white"))+ 
  guides(fill=FALSE)  

recall

fbeta <- ggviolin(metrics1, x = "type", y = "Fbeta", fill = "type",
         palette = c("#009E73", "grey", "steelblue"),
         add = "boxplot", add.params = list(fill = "white")) +xlab("")+ylab("F-beta (0.1)") + 
  guides(fill=FALSE)  

fbeta

# 95% CI

metrics1  %>% 
  filter(!is.na(Precision)) %>% 
  group_by(type) %>%
  summarise(mean.Precision = mean(Precision, na.rm = TRUE),
            sd.Precision = sd(Precision, na.rm = TRUE),
            n.Precision = n()) %>%
  mutate(se.Precision = sd.Precision / sqrt(n.Precision),
         lower.ci.Precision = mean.Precision - qt(1 - (0.05 / 2), n.Precision - 1) * se.Precision,
         upper.ci.Precision = mean.Precision + qt(1 - (0.05 / 2), n.Precision - 1) * se.Precision)

metrics1  %>% 
  filter(!is.na(Recall)) %>% 
  group_by(type) %>%
  summarise(mean.Recall = mean(Recall, na.rm = TRUE),
            sd.Recall = sd(Recall, na.rm = TRUE),
            n.Recall = n()) %>%
  mutate(se.Recall = sd.Recall / sqrt(n.Recall),
         lower.ci.Recall = mean.Recall - qt(1 - (0.05 / 2), n.Recall - 1) * se.Recall,
         upper.ci.Recall = mean.Recall + qt(1 - (0.05 / 2), n.Recall - 1) * se.Recall)

metrics1  %>% 
  filter(!is.na(Fbeta)) %>% 
  group_by(type) %>%
  summarise(mean.Fbeta = mean(Fbeta, na.rm = TRUE),
            sd.Fbeta = sd(Fbeta, na.rm = TRUE),
            n.Fbeta = n()) %>%
  mutate(se.Fbeta= sd.Fbeta / sqrt(n.Fbeta),
         lower.ci.Fbeta = mean.Fbeta - qt(1 - (0.05 / 2), n.Fbeta - 1) * se.Fbeta,
         upper.ci.Fbeta = mean.Fbeta + qt(1 - (0.05 / 2), n.Fbeta - 1) * se.Fbeta)


# Distance to the detection time from the symptom onset (COVID-19) -----------------------------------------------

anom1 <- read.table("all_anomalies.csv", sep=',', header = T)

# add dates
dates1 <- read.table("symptom_dates_all.csv", sep=',', head=T)

# jointhe  ids with LAAD can run
metrics1 <- read.table("all_metrics_table.txt", sep='\t', header = T)
ids <- metrics1 %>% select(id, type)

dates1 <- left_join(ids, dates1, by="id")
dates1 <- dates1 %>% select(-type.y) %>% rename(type = "type.x")

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
  mutate(group2 = case_when(#signal_count <= 6 | 
    Deltas >=0 ~ "Late",
    Deltas < 0 ~ "Early") )


delta5  <- delta4  %>% 
  select(-loss)%>% 
  filter(type=="COVID-19") %>% 
  #filter(signal_count >6) %>% 
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

delta5  %>% group_by(type,group2) %>% 
  summarise(count = n()) 

delta5  %>% group_by(type, group2) %>% 
  summarise(mean = mean(group2_deltas),
            median =  median(group2_deltas)) 

delta5  %>% group_by(type) %>% 
  summarise(mean = mean(group2_deltas),
            median =  median(group2_deltas)) 

distance <- gghistogram(delta5, x = "group2_deltas",
            add = "mean", rug = TRUE,
            fill = "group2", palette = c("#00AFBB", "#E7B800"),
            #add_density = TRUE, 
            xlab = "Distance", ylab = "Number of individuals")+ 
  theme(legend.position="top") +
  theme(legend.title=element_blank())
distance

# all

delta5  <- delta4  %>% 
  select(-loss)%>% 
  #filter(type=="COVID-19") %>% 
  #filter(signal_count >6) %>% 
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


gghistogram(delta5, x = "group2_deltas",
            #add = "median", 
            rug = TRUE,
            fill = "group2", palette = c("#00AFBB", "#E7B800"),
            add_density = TRUE, xlab = "Distance", ylab = "Count")+
  facet_grid(type~., scales ="free")+ 
  theme(legend.position="top") +
  theme(legend.title=element_blank())

# Duration of abnormal RHR (in hours) --------------------------------------------------------

anom1 <- read.table("all_anomalies.csv", sep=',', header = T)

anom1  %>% select(type,id) %>% 
  distinct() %>% 
  group_by(type)  %>% 
  summarise(count = n()) 

ids <- anom1  %>% select(type,id) %>% 
  distinct() 

# add dates
dates1 <- read.table("symptom_dates_all.csv", sep=',', head=T)

# jointhe  ids with LAAD can run
metrics1 <- read.table("all_metrics_table.txt", sep='\t', header = T)
ids <- metrics1 %>% select(id, type)

dates1 <- left_join(ids, dates1, by="id")
dates1 <- dates1 %>% select(-type.y) %>% rename(type = "type.x")

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
  #filter(id != "AA0HAI1_1")  %>% 
  #filter(id != "AA0HAI1_2")  %>% 
  #filter(id != "AA0HAI1_3")  %>% 
  group_by(id) %>% 
  mutate(count=n()) %>% 
  unique()

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
  #stat_compare_means(label.y = 500) +    # Add global p-value
  xlab("")+ylab("Duration of \nabnormal RHR (hours)") 

duration

delta1_dur_all %>% 
  group_by(type) %>% 
  mutate(median = median(count)) %>% 
  select(type,median) %>% 
  unique()

## barplot -1 -----------------------------------------------------

delta1_dur_all %>% ggbarplot(x = "id", y = "count", 
                             fill = "type",
                             color = "white",
                             palette = c("#009E73", "grey", "steelblue"),
                             xlab = FALSE,
                             ylab = "No.of abnormal hours",
                             #sort.val = "desc",  
                             lab.size = 3,
                             font.label = list(size = 10,vjust = 0.5),
                             label = round(delta1_dur_all$count-1))+ 
  theme(legend.position="none") +
  theme(axis.text.x = element_text(size=8, angle = 90))+
  facet_grid(.~type, scales = "free", space = "free") +
  ylab("Duration of \n abnormal RHR (hours)")
#geom_hline(yintercept=24, colour="black", linetype="dashed", size=0.8)


# Delta RHR --------------------------------------------------------------

anom1 <- read.table("all_anomalies.csv", sep=',', header = T)

# add dates
dates1 <- read.table("symptom_dates_all.csv", sep=',', head=T)

# jointhe  ids with LAAD can run
metrics1 <- read.table("all_metrics_table.txt", sep='\t', header = T)
ids <- metrics1 %>% select(id, type)

dates1 <- left_join(ids, dates1, by="id")
dates1 <- dates1 %>% select(-type.y) %>% rename(type = "type.x")


dis2 <- left_join(dates1, anom1)
dis2$symptom_date <- ymd_hms(dis2$symptom_date)  
dis2$datetime <- ymd_hms(dis2$datetime)   

dis3_laad_1 <- dis2 %>% group_by(id)  %>% 
  mutate(Deltas = as.double.difftime(datetime - symptom_date, units="days"))


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


delta4 <- delta3 %>% 
  group_by(id, type) %>%
  mutate(positive = delta_RHR >=0)  %>% 
  group_by(id,type,positive) %>% 
  summarise(mean = mean(delta_RHR),
            median = median(delta_RHR))

write.table(delta4, "LAAD_delta_RHR_all.txt", sep=",")


# barplot-1 ------------------------------------------------------------------

ggbarplot(delta4, x = "id", y = "median",
          fill = "positive",
          color = "white",
          palette = "jco",
          x.text.angle = 90,
          lab.size = 3,
          lab.pos = "in",
          lab.vjust = -0.0002,
          ylab = "Delta RHR (Median)",
          xlab = FALSE,
          label = round(delta4$median))+
  theme(axis.text.x = element_text(size=8, angle = 90))+
  facet_grid(.~type, scales = "free", space = "free")+ 
  theme(legend.position="botopttom") +
  theme(legend.title=element_blank())


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
          xlab = "",
          label = round(delta5$median,2),
          font.label = list(size = 9,vjust = 0.1))+ 
  theme(legend.position="top") +
  theme(legend.title=element_blank())
delta_rhr

plot_grid(distance, fbeta, duration, delta_rhr, ncol = 4)




