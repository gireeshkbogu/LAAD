# Aim: Count number of delta RHR  during COVID-19 infectious period
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

# Delta RHR  -----------------------------------------------------


# all --------------------------------------------------------------------------

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

delta4 <- delta3 %>% 
  group_by(id, type) %>%
  mutate(positive = delta_RHR >=0)  %>% 
  group_by(id,type,positive) %>% 
  summarise(mean = mean(delta_RHR),
            median = median(delta_RHR))

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
  facet_grid(.~type, scales = "free", space = "free")


# barplot-2 summary figure ---------------------------------------------------

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





