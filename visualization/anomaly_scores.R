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
dates1 <- read.table("symptom_dates.txt", sep=',', head=T)

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
                        levels=c("AA0HAI1_1",
                                 "A0KX894",
                                 "AK7YRBU",
                                 "AR4FPCC",
                                 "AEOBCFJ",
                                 "AA0HAI1_3",
                                 "AA0HAI1_2",
                                 "A35BJNV",
                                 "AOGFRXL",
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

write.table(dis3_laad, "LAAD_deltas_healthy.txt", sep=",")

dis3_laad_1 <- dis3_laad  %>% 
  group_by(id) %>% 
  #filter(Deltas >= -500 | Deltas <= 500  | is.na(Deltas)) %>% 
  mutate(loss_scaled=scales::rescale(loss,to=c(0, 1)))

cbPalette = c("#00AFBB", "#E7B800", "#DF4E07")


dis3_laad_1$id = factor(dis3_laad_1$id, 
                        levels=c("AAGTWZK",
                                 "A5XL2IC",
                                 "AAF9ACE",
                                 "AKV66US",
                                 "AFEFA29",
                                 "ATT9RR1",
                                 "AQ4TMLV",
                                 "AUCGUF3",
                                 "A11SQQN",
                                 "AXCO7I9",
                                 "AE0MQ94",
                                 "AL48GP3",
                                 "AEOHH30",
                                 "A6GEBIK",
                                 "A8CBEJZ",
                                 "A7EAWA7",
                                 "ARR2IKE",
                                 "A2D7K4A",
                                 "AQ25Y0L",
                                 "A6BUI4N",
                                 "AHP25OJ",
                                 "A45F9E6",
                                 "A2P3LTM",
                                 "AW4EXXK",
                                 "AFVAEC7",
                                 "A0822M0",
                                 "AXD3W8O",
                                 "AE2B3RH",
                                 "AWRBUQZ",
                                 "AY8TPMP",
                                 "AWA2KJK",
                                 "A91HEZV",
                                 "A99ZKKW",
                                 "AZKZ0AI",
                                 "ATHBFWX",
                                 "A17YCA2",
                                 "AXDWDEA",
                                 "AOQA85X",
                                 "AGA8XUN",
                                 "AZ35PI5",
                                 "AMQUHOQ",
                                 "A9ZG5GR",
                                 "A2XFW2N",
                                 "ARFYLMK",
                                 "A11V1FH",
                                 "AGYQJEL",
                                 "AQR8ZSS",
                                 "AO20DS4",
                                 "AZ2RYW7",
                                 "A06L7KF",
                                 "AF3MXM1",
                                 "AF8R0I6",
                                 "AX3KEW9",
                                 "ARYB2QO",
                                 "APHNRSV",
                                 "AGKI03N",
                                 "ALZDAVZ",
                                 "A8QLAB0",
                                 "A65HVGP",
                                 "APDJ1QP",
                                 "A0L9BM2",
                                 "AFYLHG4",
                                 "AL3KT5B",
                                 "AOS4BSJ",
                                 "A0N9NV4",
                                 "A4H7SNF",
                                 "AYVQUF1",
                                 "AEZDKVO",
                                 "AOB9SON",
                                 "AJ0DKQ3"
                        ))


dis3_laad_1 %>% 
  filter(!is.na(id)) %>% 
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



