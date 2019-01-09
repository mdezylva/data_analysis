# Loads base packages
library(ggplot2)
library(reshape2)
library(plyr)


ht_weight_df <- read.csv(file = "data.csv")
# str is short for structure(). It reports what's in the data.frame
str(ht_weight_df)

plot(x = ht_weight_df$Height, y = ht_weight_df$Weight)
colnames(ht_weight_df)
colnames(ht_weight_df)[1] <- "Weight"
colnames(ht_weight_df)[2] <- "Height"
lm_ht_weight <- lm(Weight ~ Height, data = ht_weight_df)
summary(lm_ht_weight)
