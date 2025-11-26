rm(list=ls())

library(ggplot2)
library(reshape2)
library(ggpubr)
library(dplyr)
library(tidyverse)
library(patchwork)
library(stringr)
library(ggsci)
library(cowplot)

data <- bind_rows(Concatenation, gate, Residual_gate)

########################################################
########## Plotting
########################################################

plot_list <- map(seq(1, 7, by = 1), function(i) {
  p <- ggplot(data = data, aes(x = Group, y = !!sym(colnames(data)[i]))) +
    geom_violin(aes(fill = Group)) +
    geom_boxplot(width = 0.1) +
    geom_signif(
      comparisons = list(c("Concatenation", "gate"),
                         c("gate", "Residual_gate"),
                         c("Concatenation", "Residual_gate")),
      y_position = c(max(data[[i]])+0.3, max(data[[i]])+0.5,max(data[[i]])+1),
      test = "t.test",
      map_signif_level = TRUE
    ) +
    theme_classic() +
    theme(legend.position = "bottom",
          legend.direction = "horizontal",
          legend.box = "horizontal",
          legend.justification = "center",
          legend.box.just = "center") +
    scale_fill_npg()
})

final_plot <- wrap_plots(plot_list, ncol = 7)

pdf(paste0("Total/Total_Symbol.pdf"),width=25,height=5)
print(final_plot)
dev.off()

