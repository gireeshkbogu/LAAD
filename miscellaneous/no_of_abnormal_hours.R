# Aim: Count number of abnormal RHR hours during COVID-19 infectious period
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

# abnormal hours  -----------------------------------------------------

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


dis3_laad_1 %>% select(type,id) %>% unique()%>%  
  filter(!is.na(type)) %>% group_by(type) %>% count()

dis3_laad_1 %>% select(type,id) %>% unique()%>%  
  filter(!is.na(type))

delta3 <- dis3_laad_1%>% 
  group_by(id) %>% 
  filter(Deltas >= -7 & Deltas <= 21  | is.na(Deltas))


delta3 %>% select(type,id) %>% unique()%>%  
  filter(!is.na(type)) %>% group_by(type) %>% count()


delta1_dur_all <- delta3 %>% select(type, id)%>%  
  filter(!is.na(type)) %>% 
  group_by(id) %>% 
  mutate(count=n()) %>% 
  unique()

write.table(delta1_dur_all, "LAAD_duration_of_abnormal_RHR.txt", sep=",")

# Visualize: Specify the comparisons you want
my_comparisons <- list( c("COVID-19", "Healthy"), c("COVID-19", "Non-COVID-19"), c("Healthy", "Non-COVID-19") )
ggboxplot(delta1_dur_all, 
          x = "type", y = "count",
          color = "type", 
          palette = c("#009E73", "grey", "steelblue"),
          add = "jitter",
          add.params = list(size = 0.3, jitter = 0.2))+ 
  theme(legend.position="none") + 
  stat_compare_means(comparisons = my_comparisons, method  = 'wilcox.test')+ # Add pairwise comparisons p-value
  #stat_compare_means(label.y = 500) +    # Add global p-value
  xlab("")+ylab("Duration of \nabnormal RHR (hours)") 


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

# barplot-2 ------------------------------------------------------

delta1_dur_all_1 <- delta1_dur_all%>% 
  group_by(type) %>% 
  summarise(count1 = sum(count>24),
            total = sum(count>=0),
            percentage = (count1/total)*100
  )

delta1_dur_all_1

delta1_dur_all_1%>% ggbarplot(x = "type", y = "percentage", 
                      fill = "type",
                      color = "white",
                      palette = c("#009E73", "grey", "steelblue"),
                      xlab = FALSE,
                      ylab = "Percentage of individuals\n with more than one day of abnormal RHR \nduring COVID-19 period",
                      label = round(delta1_dur_all_1$percentage-1),
                      font.label = list(size = 1,vjust = 1))+ 
  theme(legend.position="none") 

# signal strength --------------------------------------------------

delta4 <- delta3  %>% select(id,type,Deltas,loss) %>%   
  # filtered anomaly scores during COVID-19 infectious period by takign the values between 7 days before and 21 days after symptom onset. 
  # grouped patients who had anomaly signal before symptom onset as pre-symptomatic and after as post-symptomatic
  # anomaly sginal strength was calculated by dividing the number of anomaly events with the total loss score (anomaly score) and further divided by 7 and multiplied by 100.
  mutate(group = ifelse(Deltas >=0, "Post", "Pre")) %>% 
  group_by(id,type, group) %>% 
  mutate("signal_count" = n()) %>% 
  mutate("total_loss" = sum(loss)) %>% 
  mutate("strength" = ((total_loss/signal_count)/7)*100) %>%   
  # 
  mutate(group1 = ifelse(strength >6, "Strong", "Weak")) %>%
  # grouped the patients who had signal before symptom onset as "Early" and either after on-set or
  # before onset but with signal count less than or equal to 6  as "Late".
  mutate(group2 = case_when(Deltas >=0 ~ "Late",
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
  

delta5  <- delta4  %>% 
  select(-loss)%>% 
  filter(type=="Non-COVID-19") %>% 
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


delta5  <- delta4  %>% 
  select(-loss)%>% 
  filter(type=="Healthy") %>% 
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

