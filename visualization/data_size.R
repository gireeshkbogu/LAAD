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

# this file doesnn't have healthy anomalies
size1 <- read.table("data_sizes.csv", sep=',', header = T)

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





