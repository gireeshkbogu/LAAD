# Aim: Compare metrics across 3 different groups
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




ggviolin(metrics1, x = "type", y = "Precision", fill = "type",
         palette = c("#009E73", "grey", "steelblue"), alpha=0.5,
         add = "boxplot", add.params = list(fill = "white"))+ 
  guides(fill=FALSE)  

ggviolin(metrics1, x = "type", y = "Recall", fill = "type",
         palette = c("#009E73", "grey", "steelblue"), alpha=0.5,
         add = "boxplot", add.params = list(fill = "white"))+ 
  guides(fill=FALSE)  

ggviolin(metrics1, x = "type", y = "Fbeta", fill = "type",
         palette = c("#009E73", "grey", "steelblue"),
         add = "boxplot", add.params = list(fill = "white")) +xlab("")+ylab("F-beta (0.1)") + 
  guides(fill=FALSE)  

# print min, max, mean, median

metrics1  %>% 
  filter(!is.na(Precision)) %>% 
  group_by(type) %>% 
  summarise(min = min(Precision),
            max = max(Precision),
            mean = mean(Precision),
            median = median(Precision))

metrics1   %>% 
  filter(!is.na(Recall)) %>% 
  group_by(type) %>% 
  summarise(min = min(Recall),
            max = max(Recall),
            mean = mean(Recall),
            median = median(Recall))

metrics1   %>% 
  filter(!is.na(Fbeta)) %>%  
  group_by(type) %>% 
  summarise(min = min(Fbeta),
            max = max(Fbeta),
            mean = mean(Fbeta),
            median = median(Fbeta))

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


# Visualize: Specify the comparisons you want
my_comparisons <- list( c("COVID-19", "Healthy"), c("COVID-19", "Non-COVID-19"), c("Healthy", "Non-COVID-19") )
ggboxplot(metrics1, 
          x = "type", y = "Fbeta",
          color = "type", 
          palette = c("#009E73", "grey", "steelblue"),
          add = "jitter",
          add.params = list(size = 0.3, jitter = 0.2))+ 
  theme(legend.position="none") + 
  stat_compare_means(comparisons = my_comparisons, method  = 'wilcox.test')+ # Add pairwise comparisons p-value
  stat_compare_means(label.y = 1.6)  +   # Add global p-value
  xlab("")+ylab("F-beta (0.1)") 


