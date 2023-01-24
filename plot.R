rm(list = ls())
library(ggplot2)

latent_tsne <- read.table("tsne_2D.txt", sep = ",")
colnames(latent_tsne) <- c("TSNE_1", "TSNE_2")
y_pred <- as.numeric(readLines("pred_labels.txt"))
y_pred <- factor(y_pred, levels = 0:max(y_pred))

dat <- data.frame(latent_tsne, y_pred = y_pred)

m <- ggplot(dat, aes(x = TSNE_1, y = TSNE_2, color = y_pred)) +
    geom_point() +
    theme_classic()
print(m)
