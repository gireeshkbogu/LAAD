# Aim: Count the distance of ealry and late detection times
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

######################################

# this file doesnn't have healthy anomalies
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

write.table(delta5, "LAAD_distance_covid19_positive.txt", sep=",")

gghistogram(delta5, x = "group2_deltas",
            add = "mean", rug = TRUE,
            fill = "group2", palette = c("#00AFBB", "#E7B800"),
            add_density = TRUE, xlab = "Distance", ylab = "Count")


##  non-covid19

delta5  <- delta4  %>% 
  select(-loss)%>% 
  filter(type=="Non-COVID-19") %>% 
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

write.table(delta5, "LAAD_distance_noncovid19.txt", sep=",")

gghistogram(delta5, x = "group2_deltas",
            add = "mean", rug = TRUE,
            fill = "group2", palette = c("#00AFBB", "#E7B800"),
            add_density = TRUE, xlab = "Distance", ylab = "Count")

# healthy -------------
##  non-covid19

delta5  <- delta4  %>% 
  select(-loss)%>% 
  filter(type=="Healthy") %>% 
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

write.table(delta5, "LAAD_distance_healthy.txt", sep=",")

gghistogram(delta5, x = "group2_deltas",
            add = "mean", rug = TRUE,
            fill = "group2", palette = c("#00AFBB", "#E7B800"),
            add_density = TRUE, xlab = "Distance", ylab = "Count")

# all -----------------------------------------------------------------------

delta5  <- delta4  %>% 
  select(-loss)%>% 
  #filter(type=="COVID-19") %>% 
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


delta5  %>% group_by(type,group2) %>% 
  summarise(count = n()) 

delta5  %>% group_by(type, group2) %>% 
  summarise(mean = mean(group2_deltas),
            median =  median(group2_deltas)) 

delta5  %>% group_by(type) %>% 
  summarise(mean = mean(group2_deltas),
            median =  median(group2_deltas)) 

write.table(delta5, "LAAD_distance_all.txt", sep=",")

gghistogram(delta5, x = "group2_deltas",
            #add = "median", 
            rug = TRUE,
            fill = "group2", palette = c("#00AFBB", "#E7B800"),
            add_density = TRUE, xlab = "Distance", ylab = "Count")+
  facet_grid(type~.)

