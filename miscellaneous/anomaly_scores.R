# Aim: Plot annomaly scocres during COVID-19 infectious period
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


setwd('/Users/gireeshbogu/Desktop/laad/figures/')
# COVID-19 positive
# global figure --------------------------------------------------------------------------
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
                                 "A1K5DRI",
                                 "APGIB2T",
                                 "AUY8KYW",
                                 "A0NVTRV",
                                 "AHYIJDV",
                                 "AQC0L71",
                                 "AV2GF3B",
                                 "AS2MVDL",
                                 "AOYM4KG",
                                 "A4G0044",
                                 "AYWIEKR",
                                 "AX6281V",
                                 "A3OU183",
                                 "AA2KP1S",
                                 "AURCTAK",
                                 "A0VFT1N",
                                 "AMV7EQF",
                                 'AYEFCWQ'
                        ))

dis3_laad_1 %>% 
  ggplot(aes(x=Deltas, y=loss_scaled)) + 
  geom_bar(stat="identity", position="identity", fill="#009E73",  width=0.1)+
  facet_grid(id~., scales = "free", space="free", switch="both") +
  theme_pubr()+
  #scale_x_continuous(limits = c(-56, 56),breaks = seq(-56, 56, by = 7))+
  scale_x_continuous(limits = c(-21, 35),breaks = seq(-21, 35, by = 1))+
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
  #scale_x_continuous(limits = c(-14, 28),breaks = seq(-14, 28, by = 1))+
  scale_x_continuous(limits = c(-30, 90),breaks = seq(-30, 90, by = 10))+
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


dis3_laad_2 <- dis3_laad  %>% 
  group_by(id) %>% 
  filter(Deltas >= -7 & Deltas <= 21  | is.na(Deltas)) %>% 
  mutate(loss_scaled=scales::rescale(loss,to=c(0, 1)))

dis3_laad_3 <- dis3_laad_2 %>% 
  select(id, Deltas) %>% 
  arrange(id) %>% 
  distinct(id, .keep_all = TRUE)


# Non-COVID
# global figure --------------------------------------------------------------------------
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

dis3_laad_1 %>% 
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

##  Healthy

# global figure --------------------------------------------------------------------------
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
                        levels=c("A99ZKKW",
                                 "AW4EXXK",
                                 "AAGTWZK",
                                 "A5XL2IC",
                                 "A91HEZV",
                                 "ATT9RR1",
                                 "A2P3LTM",
                                 "A45F9E6",
                                 "AWA2KJK",
                                 "A06L7KF",
                                 "AFVAEC7",
                                 "AXD3W8O",
                                 "AKV66US",
                                 "AQR8ZSS",
                                 "AWRBUQZ",
                                 "AY8TPMP",
                                 "AE0MQ94",
                                 "AEOHH30",
                                 "A2XFW2N",
                                 "AZKZ0AI",
                                 "AQ25Y0L",
                                 "AX3KEW9",
                                 "AXDWDEA",
                                 "AZ2RYW7",
                                 "AL48GP3",
                                 "A6BUI4N",
                                 "A11V1FH",
                                 "AQ4TMLV",
                                 "A7EAWA7",
                                 "A11SQQN",
                                 "AZ35PI5",
                                 "AFEFA29",
                                 "AFYLHG4",
                                 "AEZDKVO",
                                 "AHP25OJ",
                                 "ARFYLMK",
                                 "AMQUHOQ",
                                 "AJ0DKQ3",
                                 "AUCGUF3",
                                 "ALZDAVZ",
                                 "A8QLAB0",
                                 "APHNRSV",
                                 "APDJ1QP",
                                 "AGKI03N",
                                 "A0822M0",
                                 "AOQA85X",
                                 "AYVQUF1",
                                 "A0L9BM2",
                                 "AO20DS4",
                                 "A4H7SNF",
                                 "AAF9ACE",
                                 "A6GEBIK",
                                 "A2D7K4A",
                                 "AL3KT5B",
                                 "ARYB2QO",
                                 "AOS4BSJ",
                                 "A9ZG5GR",
                                 "AOB9SON",
                                 "A17YCA2",
                                 "AXCO7I9",
                                 "A65HVGP",
                                 "ARR2IKE",
                                 "A8CBEJZ"
                        ))


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

dis3_laad_1 %>% 
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



