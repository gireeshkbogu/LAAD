# Aim: 
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

size1 <- read.table("size_dates.csv", sep=',', header = T)

size1$start_date <- ymd(size1$start_date) 
size1$end_date <- ymd(size1$end_date)
size1$symptom_date_before_20 <- ymd(size1$symptom_date_before_20) 
size1$symptom_date_after_21 <- ymd(size1$symptom_date_after_21) 
size1$symptom_date_before_10 <- ymd(size1$symptom_date_before_10) 
size1$symptom_date_before_7 <- ymd(size1$symptom_date_before_7) 


size2 =  size1 %>% 
  mutate(train = symptom_date_before_20 - start_date,
         test = end_date - symptom_date_before_20,
         test_normal = symptom_date_before_10 - symptom_date_before_20,
         test_anomaly = symptom_date_after_21- symptom_date_before_7,
         test_recovery = end_date - symptom_date_after_21)


# 
size3 <- read.table("data_size.csv", sep=',', header = T)


# jointhe  ids with LAAD can run
metrics1 <- read.table("all_metrics_table.txt", sep='\t', header = T)
ids <- metrics1 %>% select(id, type)

size4 <- left_join(ids, size3, by="id")
size4 <- size4 %>% select(-type.y) %>% rename(type = "type.x")


size4 %>% select(type) %>% 
  group_by(type) %>% 
  count()

size4 %>% 
  group_by(type) %>% 
  mutate(train_mean = mean(train),
         train_sd= sd(train),
         test_mean = mean(test),
         test_sd = sd(test),
         testn_mean = mean(test_noninfectious),
         testn_sd = sd(test_noninfectious),
         testi_mean = mean(test_infectious),
         testi_sd = sd(test_infectious)) %>% 
  select(type, train_mean, train_sd, test_mean, test_sd, testn_mean, testn_sd, testi_mean, testi_sd) %>% 
  unique()

# symptom  dates

sympt1 <- read.table("symptom_dates_all.csv", sep=',', header = T)

# jointhe  ids with LAAD can run
metrics1 <- read.table("all_metrics_table.txt", sep='\t', header = T)
ids <- metrics1 %>% select(id, type)

sympt2 <- left_join(ids, sympt1, by="id")
sympt2 <- sympt2 %>% select(-type.y) %>% rename(type = "type.x")

