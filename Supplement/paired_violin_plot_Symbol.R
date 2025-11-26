rm(list=ls())

library(ggplot2)
library(reshape2)
library(ggpubr)
library(dplyr)
library(tidyverse)
library(patchwork)
library(stringr)

data <- read.csv(paste0(modelname,".csv"))

########################################################
########## Plotting
########################################################
plot_list <- map(seq(1, 14, by = 2), function(i) {
  g1_name <- colnames(data)[i]
  g2_name <- colnames(data)[i + 1]
  g1      <- data[[i]]
  g2      <- data[[i + 1]]
  
  df_long <- tibble(
    Value = c(g1, g2),
    Type  = rep(c(g1_name, g2_name), each = length(g1)),
    Group = str_replace(g1_name, "_without", "")
  )
  
  # Single Image
  p <- ggplot(df_long, aes(x = Type, y = Value, fill = Type)) +
    geom_violin(trim = FALSE, alpha = 0.7) +
    geom_boxplot(width = 0.1, outlier.shape = NA, colour = "grey20") +
    facet_wrap(~ Group, nrow = 1) +
    labs(x = "Type", y = str_replace(g1_name, "_without", ""),
         title = paste0(str_replace(g1_name, "_without", ""))) +
    theme_minimal(base_size = 14) +
    theme(legend.position = "bottom",
          legend.direction = "horizontal",
          legend.box = "horizontal",
          legend.justification = "center",
          legend.box.just = "center") +
    scale_fill_manual(values = c("#00AFBB", "#E7B800"))
  p <- p + stat_compare_means(
    aes(group = Type),
    method = "t.test",
    paired = TRUE,
    label = "p.signif"
  )
})
#################################
###### Combine Images ######
final_plot <- wrap_plots(plot_list, ncol = 7) #Horizontal layout

pdf(paste0(modelname,"/",modelname,"_Symbol.pdf"),width=25,height=5)
print(final_plot)
dev.off()


