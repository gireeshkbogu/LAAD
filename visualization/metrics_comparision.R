# Aim: Compare metrics across 3 different groups
## Date: 13 Nov 2020 
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

# ALL  ---------------------------------------------------

metrics1 <- read.table("all_metrics_table.txt", sep='\t', header = T)

ggdensity(metrics1, x = "TP",
            add = "median", rug = TRUE,
            color = "type", fill = "type", alpha=0.3,
            palette = c("red", "#E7B800",  "#00AFBB"))

ggdensity(metrics1, x = "FP",
          add = "median", rug = TRUE,
          color = "type", fill = "type", alpha=0.3,
          palette = c("red", "#E7B800",  "#00AFBB"))

ggdensity(metrics1, x = "TN",
          add = "median", rug = TRUE,
          color = "type", fill = "type", alpha=0.3,
          palette = c("red", "#E7B800",  "#00AFBB"))

ggdensity(metrics1, x = "FN",
          add = "median", rug = TRUE,
          color = "type", fill = "type", alpha=0.3,
          palette = c("red", "#E7B800",  "#00AFBB"))




ggviolin(metrics1, x = "type", y = "precision", fill = "type",
         palette = c( "#FC4E07", "#E7B800", "#00AFBB"), alpha=0.5,
         add = "boxplot", add.params = list(fill = "white"))

ggviolin(metrics1, x = "type", y = "recall", fill = "type",
         palette = c( "#FC4E07", "#E7B800", "#00AFBB"), alpha=0.5,
         add = "boxplot", add.params = list(fill = "white"))

ggviolin(metrics1, x = "type", y = "F1", fill = "type",
         palette = c( "#FC4E07", "#E7B800", "#00AFBB"), alpha=0.5,
         add = "boxplot", add.params = list(fill = "white"))

# print min, max, mean, median
metrics1  %>% 
  group_by(type) %>% 
  summarise(min = min(specificity),
            max = max(specificity),
            mean = mean(specificity))


metrics1  %>% 
  group_by(type) %>% 
  summarise(min = min(precision),
            max = max(precision),
            mean = mean(precision))

metrics1  %>% 
  group_by(type) %>% 
  summarise(min = min(recall),
            max = max(recall),
            mean = mean(recall))

metrics1  %>% 
  group_by(type) %>% 
  summarise(min = min(F1),
            max = max(F1),
            mean = mean(F1))





# global figure --------------------------------------------------------------------------
anom1 <- read.table("all_covid_positive_anomalies.csv", sep=',', header = T)

# add dates
dates1 <- read.table("symptom_dates.txt", sep=',', head=T)

dis2 <- left_join(dates1, anom1)
dis2$symptom_date <- ymd_hms(dis2$symptom_date)  
dis2$datetime <- ymd_hms(dis2$datetime)   

dis3_laad <- dis2 %>% group_by(id)  %>% 
  mutate(Deltas = as.double.difftime(datetime - symptom_date, units="days"))

write.table(dis3_laad, "LAAD_deltas_covid_positive.txt", sep="\t")

dis3_laad_1 <- dis3_laad  %>% 
  group_by(id) %>% 
  #filter(Deltas >= -50 | Deltas <= 50  | is.na(Deltas)) %>% 
  mutate(loss_scaled=scales::rescale(loss,to=c(0, 1)))

cbPalette = c("#00AFBB", "#E7B800", "#DF4E07")

dis3_laad_1$id = factor(dis3_laad_1$id, 
                                   levels=c('A0NVTRV',
                                            'A4E0D03',
                                            'AJWW3IY',
                                            'ASFODQR',
                                            'AMV7EQF',	
                                            'AFPB8J2',
                                            'AUY8KYW',
                                            'AKXN5ZZ',
                                            'AJMQUVV',
                                            'AYWIEKR',
                                            'AAXAA7Z',
                                            'AA2KP1S',
                                            'AIFDJZB',
                                            'AOYM4KG',
                                            'A4G0044',
                                            'AHYIJDV',
                                            'AURCTAK',
                                            'A3OU183',
                                            'AS2MVDL',
                                            'AV2GF3B',
                                            'A1K5DRI',
                                            'AZIK4ZA',
                                            'A7EM0B6',
                                            'AQC0L71',
                                            'APGIB2T',
                                            'AX6281V',
                                            'A36HR6Y',
                                            'A1ZJ41O',
                                            'AYEFCWQ'
                                            #'AJ7TSV9',
                                            #'ATHKM6V'
                                   ))

dis3_laad_1 %>% 
  ggplot(aes(x=Deltas, y=loss_scaled)) + 
  geom_bar(stat="identity", position="identity", fill="#009E73",  width=0.1)+
  facet_grid(id~., scales = "free", space="free", switch="both") +
  theme_pubr()+
  #scale_x_continuous(limits = c(-56, 56),breaks = seq(-56, 56, by = 7))+
  scale_x_continuous(limits = c(-14, 28),breaks = seq(-14, 28, by = 1))+
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


# delta RHR calculations -------------------------------------

delta1 <- read.table("all_covid_positive_delta_RHR.csv", sep=',', header = T)

delta2 <- delta1 %>% 
  group_by(id) %>%
  mutate(positive = delta_RHR >=0) 


delta2 %>% 
  ggboxplot(x = "id", y = "delta_RHR",ggtheme = theme_pubr(),
            fill = "positive")+
  theme(axis.text.x = element_text(size=8, angle = 90))+
  ylab("Delta RHR")+xlab("")+
  geom_hline(yintercept=0, colour="grey", linetype="dashed", size=0.8)+
  facet_wrap(id~., scales = 'free')+
  theme(axis.text.y=element_text(size=8))


delta3 <- delta1 %>% 
  group_by(id) %>%
  mutate(positive = delta_RHR >=0)  %>% 
  group_by(id,positive) %>% 
  summarise(mean = mean(delta_RHR),
            median = median(delta_RHR))


ggbarplot(delta3, x = "id", y = "median",fill = "positive",color = "white",palette = "jco",
          x.text.angle = 90,
          ylab = "Delta RHR (Median)",
          xlab = FALSE,
          label = round(delta3$median),
          font.label = list(size = 9,vjust = 1))+
  theme(axis.text.x = element_text(size=8, angle = 90))


## all
delta1 <- read.table("figures/all_delta_RHR.csv", sep=',', header = T)

delta3 <- delta1 %>% 
  group_by(id, type) %>%
  mutate(positive = delta_RHR >=0)  %>% 
  group_by(id,type,positive) %>% 
  summarise(mean = mean(delta_RHR),
            median = median(delta_RHR))


ggbarplot(delta3, x = "id", y = "median",fill = "positive",color = "white",palette = "jco",
          x.text.angle = 90,
          ylab = "Delta RHR (Median)",
          xlab = FALSE,
          label = round(delta3$median),
          font.label = list(size = 9,vjust = 1))+
  theme(axis.text.x = element_text(size=8, angle = 90))+
  facet_grid(.~type, scales = "free", space = "free")


delta4 <- delta1 %>% 
  group_by(id, type) %>%
  mutate(positive = delta_RHR >=0)  %>% 
  group_by(type,positive) %>% 
  summarise(mean = mean(delta_RHR),
            median = median(delta_RHR))

ggbarplot(delta4, x = "type", y = "median",fill = "positive",color = "white",palette = "jco",
          #x.text.angle = 90,
          ylab = "Delta RHR (Median)",
          xlab = FALSE,
          label = round(delta4$median,2),
          font.label = list(size = 9,vjust = 0.1))


# duration --------------------------------------

delta1_dur <- delta1 %>% select(type, id) %>% 
  group_by(id) %>% 
  mutate(count=n()) %>% 
  unique()

delta1_dur %>% ggbarplot(x = "id", y = "count", fill = "type",color = "white",palette = "jco",
            xlab = FALSE,
            ylab = "No.of abnormal hours",
            label = round(delta1_dur$count-1),
            font.label = list(size = 4,vjust = 1))+
  theme(axis.text.x = element_text(size=8, angle = 90))+
  facet_grid(.~type, scales = "free", space = "free")+
  geom_hline(yintercept=16, colour="black", linetype="dashed", size=0.8)

delta1_dur%>% 
  group_by(type) %>% 
  summarise(mean = mean(count),
            median = median(count))

delta1_dur%>% 
  group_by(type) %>% 
  mutate(mean = mean(count),
            median = median(count))%>% 
  ggboxplot(x = "type", y = "count",palette = "jco",
            fill = "type")+
  ylab("No.of abnormal hours")+xlab("")

# pre and post

anom1 <- read.table("figures/all_anomalies.csv", sep=',', header = T)

dis2 <- left_join(dates1, anom1)
dis2$symptom_date <- ymd_hms(dis2$symptom_date)  
dis2$datetime <- ymd_hms(dis2$datetime)   

dis3_laad <- dis2 %>% group_by(type,id)  %>% 
  mutate(Deltas = as.double.difftime(datetime - symptom_date, units="days"))

write.table(dis3_laad, "LAAD_deltas_all.txt", sep="\t")

dis3_laad_pre <- dis3_laad  %>% 
  group_by(type, id) %>% 
  filter(Deltas >= -7 & Deltas <= 0) 

dis3_laad_post <- dis3_laad  %>% 
  group_by(type, id) %>% 
  filter(Deltas >0 & Deltas < 21) 


delta1_dur_pre <- dis3_laad_pre %>% select(type, id) %>% 
  group_by(id) %>% 
  mutate(count=n()) %>% 
  unique()

delta1_dur_post <- dis3_laad_post %>% select(type, id) %>% 
  group_by(id) %>% 
  mutate(count=n()) %>% 
  unique()

delta1_dur_pre %>% ggbarplot(x = "id", y = "count", fill = "type",color = "white",palette = "jco",
                         xlab = FALSE,
                         title="Pre-symptomatic",
                         ylab = "No.of abnormal hours",
                         label = round(delta1_dur_pre$count-1),
                         font.label = list(size = 4,vjust = 1))+
  theme(axis.text.x = element_text(size=8, angle = 90))+
  facet_grid(.~type, scales = "free", space = "free")+
  geom_hline(yintercept=16, colour="black", linetype="dashed", size=0.8)

delta1_dur_post %>% ggbarplot(x = "id", y = "count", fill = "type",color = "white",palette = "jco",
                             xlab = FALSE,
                             title="Post-symptomatic",
                             ylab = "No.of abnormal hours",
                             label = round(delta1_dur_post$count-1),
                             font.label = list(size = 4,vjust = 1))+
  theme(axis.text.x = element_text(size=8, angle = 90))+
  facet_grid(.~type, scales = "free", space = "free")+
  geom_hline(yintercept=16, colour="black", linetype="dashed", size=0.8)


delta1_dur_pre%>% 
  group_by(type) %>% 
  summarise(mean = mean(count),
            median = median(count))


delta1_dur_post%>% 
  group_by(type) %>% 
  summarise(mean = mean(count),
            median = median(count))


delta1_dur_pre%>% 
  group_by(type) %>% 
  mutate(mean = mean(count),
         median = median(count))%>% 
  ggboxplot(x = "type", y = "count",palette = "jco",title="Pre-symptomatic",
            fill = "type")+
  ylab("No.of abnormal hours")+xlab("")

delta1_dur_post%>% 
  group_by(type) %>% 
  mutate(mean = mean(count),
         median = median(count))%>% 
  ggboxplot(x = "type", y = "count",palette = "jco",title="Post-symptomatic",
            fill = "type")+
  ylab("No.of abnormal hours")+xlab("")

# distance -----------------------------

anom1 <- read.table("figures/all_anomalies.csv", sep=',', header = T)

# add dates
dates1 <- read.table("figures/all_symptom_dates.txt", sep=',', head=T)

dis2 <- left_join(dates1, anom1)
dis2$symptom_date <- ymd_hms(dis2$symptom_date)  
dis2$datetime <- ymd_hms(dis2$datetime)   

dis3_laad <- dis2 %>% group_by(type,id)  %>% 
  mutate(Deltas = as.double.difftime(datetime - symptom_date, units="days"))

dis3_laad_dist <- dis3_laad  %>% 
  select('id','type','Deltas')%>% 
  group_by(type,id) %>% 
  filter(Deltas >= -7 & Deltas <= 21) %>% 
  filter(row_number()==1) 

dis3_laad_dist %>%
  group_by(type)%>% 
  summarise(mean  = mean(Deltas),
            median = median(Deltas),
            count = n())

dis3_laad_pre <- dis3_laad_dist  %>% 
  group_by(type, id) %>% 
  filter(Deltas >= -7 & Deltas <0) 

dis3_laad_post <- dis3_laad_dist  %>% 
  group_by(type, id) %>% 
  filter(Deltas >0 & Deltas < 21) 

dis3_laad_pre %>%
  group_by(type)%>% 
  summarise(mean  = mean(Deltas),
            median = median(Deltas),
            count = n())

dis3_laad_post %>%
  group_by(type)%>% 
  summarise(mean  = mean(Deltas),
            median = median(Deltas),
            count = n())






# New Figures  - NOV 27, 2020 --------------------------------------------------------------------------////

setwd('/Users/gireeshbogu/Desktop/laad/figures/')
# COVID-19 positive
# global figure --------------------------------------------------------------------------
anom1 <- read.table("all_covid_positive_anomalies.csv", sep=',', header = T)

# add dates
dates1 <- read.table("symptom_dates.txt", sep=',', head=T)

dis2 <- left_join(dates1, anom1)
dis2 <- dis2  %>% filter(datetime != 'NA') 
dis2$symptom_date <- ymd_hms(dis2$symptom_date)  
dis2$datetime <- ymd_hms(dis2$datetime)   

dis3_laad <- dis2 %>% group_by(id)  %>% 
  mutate(Deltas = as.double.difftime(datetime - symptom_date, units="days"))

write.table(dis3_laad, "LAAD_deltas_covid_positive.txt", sep="\t")

dis3_laad_1 <- dis3_laad  %>% 
  group_by(id) %>% 
  #filter(Deltas >= -50 | Deltas <= 50  | is.na(Deltas)) %>% 
  mutate(loss_scaled=scales::rescale(loss,to=c(0, 1)))

cbPalette = c("#00AFBB", "#E7B800", "#DF4E07")

dis3_laad_1$id = factor(dis3_laad_1$id, 
                        levels=c('A0VFT1N',
                                 'A0NVTRV',
                                 'AA2KP1S',
                                 'A4E0D03',
                                 'AJWW3IY',
                                 'ASFODQR',
                                 'AMV7EQF',
                                 'AIFDJZB',	
                                 'AFPB8J2',
                                 'AUY8KYW',
                                 'AKXN5ZZ',
                                 'AJMQUVV',
                                 'AYWIEKR',
                                 'AAXAA7Z',
                                 'AQC0L71',
                                 'AURCTAK',
                                 'AOYM4KG',
                                 'A4G0044',
                                 'AHYIJDV',
                                 'A3OU183',
                                 'A1K5DRI',
                                 'APGIB2T',
                                 'AX6281V',
                                 'AS2MVDL',
                                 'AV2GF3B',
                                 'AZIK4ZA',
                                 'A7EM0B6',
                                 'A36HR6Y',
                                 'A1ZJ41O',
                                 'AYEFCWQ'
                                 #'AJ7TSV9',
                                 #'ATHKM6V',
                        ))

dis3_laad_1 %>% 
  ggplot(aes(x=Deltas, y=loss_scaled)) + 
  geom_bar(stat="identity", position="identity", fill="#009E73",  width=0.1)+
  facet_grid(id~., scales = "free", space="free", switch="both") +
  theme_pubr()+
  #scale_x_continuous(limits = c(-56, 56),breaks = seq(-56, 56, by = 7))+
  #scale_x_continuous(limits = c(-21, 35),breaks = seq(-21, 35, by = 1))+
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

dis3_laad_1 %>% 
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
  ylab("")+ xlab("")+ 
  theme(axis.ticks.y.left = element_blank(),panel.spacing.x=unit(0, "lines") , panel.spacing.y=unit(0.01,"lines")) # decrease the space betweeen grids


dis3_laad_2 <- dis3_laad  %>% 
  group_by(id) %>% 
  filter(Deltas >= -7 & Deltas <= 21  | is.na(Deltas)) %>% 
  mutate(loss_scaled=scales::rescale(loss,to=c(0, 1)))

dis3_laad_3 <- dis3_laad_2 %>% 
  select(id, Deltas) %>% 
  arrange(id) %>% 
  distinct(id, .keep_all = TRUE)


# atleast 6 continuous hour of signal is used to calculate the early vs late













# Non-COVID
# global figure --------------------------------------------------------------------------
anom1 <- read.table("all_non_covid_anomalies.csv", sep=',', header = T)

# add dates
dates1 <- read.table("symptom_dates.txt", sep=',', head=T)

dis2 <- left_join(dates1, anom1)
dis2 <- dis2  %>% filter(datetime != 'NA') 
dis2$symptom_date <- ymd_hms(dis2$symptom_date)  
dis2$datetime <- ymd_hms(dis2$datetime)   

dis3_laad <- dis2 %>% group_by(id)  %>% 
  mutate(Deltas = as.double.difftime(datetime - symptom_date, units="days"))

write.table(dis3_laad, "LAAD_deltas_non_covid.txt", sep="\t")

dis3_laad_1 <- dis3_laad  %>% 
  group_by(id) %>% 
  #filter(Deltas >= -500 | Deltas <= 500  | is.na(Deltas)) %>% 
  mutate(loss_scaled=scales::rescale(loss,to=c(0, 1)))

cbPalette = c("#00AFBB", "#E7B800", "#DF4E07")

dis3_laad_1$id = factor(dis3_laad_1$id, 
                        levels=c('AA0HAI1_1',
                                 'AA0HAI1_2',
                                 'A0KX894',
                                 'A35BJNV',
                                 'AEOBCFJ',
                                 'AF3J1YC',
                                 'AA0HAI1_3',
                                 'AR4FPCC',
                                 'AXI1PBS',
                                 'AK7YRBU',
                                 'AOGFRXL',
                                 'AUILKHG',
                                 'ALKAXMZ',
                                 'AFHOHOM',
                                 'AD77K91'
                        ))

dis3_laad_1 %>% 
  ggplot(aes(x=Deltas, y=loss_scaled)) + 
  geom_bar(stat="identity", position="identity", fill="steelblue",  width=0.1)+
  facet_grid(id~., scales = "free", space="free", switch="both") +
  theme_pubr()+
  #scale_x_continuous(limits = c(-56, 56),breaks = seq(-56, 56, by = 7))+
  scale_x_continuous(limits = c(-14, 28),breaks = seq(-14, 28, by = 1))+
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

##  Healthy

# global figure --------------------------------------------------------------------------
anom1 <- read.table("all_healthy_anomalies.csv", sep=',', header = T)

# add dates
dates1 <- read.table("symptom_dates_healthy.csv", sep=',', head=T)

dis2 <- left_join(dates1, anom1)
dis2 <- dis2  %>% filter(datetime != 'NA') 
dis2$symptom_date <- ymd_hms(dis2$symptom_date)  
dis2$datetime <- ymd_hms(dis2$datetime)   

dis3_laad <- dis2 %>% group_by(id)  %>% 
  mutate(Deltas = as.double.difftime(datetime - symptom_date, units="days"))

write.table(dis3_laad, "LAAD_deltas_healthy.txt", sep="\t")

dis3_laad_1 <- dis3_laad  %>% 
  group_by(id) %>% 
  #filter(Deltas >= -500 | Deltas <= 500  | is.na(Deltas)) %>% 
  mutate(loss_scaled=scales::rescale(loss,to=c(0, 1)))

cbPalette = c("#00AFBB", "#E7B800", "#DF4E07")

dis3_laad_1 %>% 
  ggplot(aes(x=Deltas, y=loss_scaled)) + 
  geom_bar(stat="identity", position="identity", fill="grey",  width=0.1)+
  facet_grid(id~., scales = "free", space="free", switch="both") +
  theme_pubr()+
  #scale_x_continuous(limits = c(-56, 56),breaks = seq(-56, 56, by = 7))+
  scale_x_continuous(limits = c(-14, 28),breaks = seq(-14, 28, by = 1))+
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


# Delta  RHR plots ---------------------
# aim: plot delta RHR plots duringn COVID-19 infectious period

## all
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


ggbarplot(delta4, x = "id", y = "median",
          fill = "positive",
          color = "white",
          palette = "jco",
          x.text.angle = 90,
          ylab = "Delta RHR (Median)",
          xlab = FALSE,
          label = round(delta4$median),
          font.label = list(size = 9,vjust = 1))+
  theme(axis.text.x = element_text(size=8, angle = 90))+
  facet_grid(.~type, scales = "free", space = "free")


delta5 <- delta3 %>% 
  group_by(id, type) %>%
  mutate(positive = delta_RHR >=0)  %>% 
  group_by(type,positive) %>% 
  summarise(mean = mean(delta_RHR),
            median = median(delta_RHR))

ggbarplot(delta5, x = "type", y = "median",
          fill = "positive",
          color = "white",
          palette = "jco",
          #x.text.angle = 90,
          ylab = "Delta RHR (Median)",
          xlab = FALSE,
          label = round(delta5$median,2),
          font.label = list(size = 9,vjust = 0.1))


# distance -----------------------------

# COVID-19

anom1 <- read.table("all_covid_positive_anomalies.csv", sep=',', header = T)

# add dates
dates1 <- read.table("symptom_dates.txt", sep=',', head=T)

dis2 <- left_join(dates1, anom1)
dis2 <- dis2  %>% filter(datetime != 'NA') 
dis2$symptom_date <- ymd_hms(dis2$symptom_date)  
dis2$datetime <- ymd_hms(dis2$datetime)   

dis3_laad <- dis2 %>% group_by(id)  %>% 
  mutate(Deltas = as.double.difftime(datetime - symptom_date, units="days"))

dis3_laad_1 <- dis3_laad  %>% 
  group_by(id) %>% 
  filter(Deltas >= -7 & Deltas <= 21  | is.na(Deltas)) %>% 
  select(id, loss, Deltas) %>% 
  group_by(id) %>%  
  mutate(pos_events = sum(Deltas>0),
         neg_events = sum(Deltas<0)) %>% 
  mutate(location = Deltas>0)

dis3_laad_2 <- dis3_laad_1%>% 
  group_by(id, location) %>% 
  mutate(total_signal = sum(loss))
  

dis3_laad_3 <- dis3_laad_2 %>% 
  filter(row_number()==1) %>% 
  select(id, Deltas, pos_events, neg_events, location, total_signal)

write.table(dis3_laad_3, "covid19_groups.txt", sep="\t")

groups1 <- read.table("covid_groups_modified.txt", sep='\t', header = T)

groups2 <- groups1  %>% 
  summarise(mean = mean(Deltas),
            median =  median(Deltas))

a <- ggplot(groups1, aes(x = Deltas))
a+geom_histogram(aes(y = ..density.., color = location), 
                   fill = "white",
                   position = "identity")+
  geom_density(aes(color = location), size = 1) +
  #stat_density(aes(color = location), geom="line") +
  scale_color_manual(values = c("#868686FF", "#EFC000FF"))



























groups1  %>% 
  group_by(location) %>% 
  summarise(mean_deltas = mean(Deltas),
            median_deltas =  median(Deltas),
            mean_loss = mean(loss),
            median_loss = median(loss),
            mean_events = mean(events ),
            median_events  = median(events),
            mean_strength = mean(strength ),
            median_strength  = median(strength)
  )



# Metrics

metrics1 <- read.table("all_metrics.txt", sep='\t', header = T)

metrics2  <- metrics1%>% 
  group_by(type) %>% 
  summarise(count = sum(TP>72),
            total = sum(TP>=0),
            percentage = (count/total)*100
  )

metrics2%>% ggbarplot(x = "type", y = "percentage", 
                 fill = "type",
                 color = "white",
                 palette = c("#009E73", "grey", "steelblue"),
                 xlab = FALSE,
                 ylab = "Percentage of individuals\n with >3 days of abnormal RHR \nduring COVID-19 period",
                 label = round(metrics2$percentage-1),
                 font.label = list(size = 1,vjust = 1))

