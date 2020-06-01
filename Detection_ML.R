version

# Replication of Breast Cancer Detection with KNN and SVM

# Initialize environment
library(data.table)
library(dplyr)
library(kknn)
# library(libsvm)

# Read in dataset
df <- fread('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data')
names(df) <- c('id','clump_thickness','uniform_cell_size','uniform_cell_shape','marginal_adhesion','single_epithelial_size','bare_nuclei','bland_chromatin','normal_nucleoli','mitosis','class')
head(df)

str(df)

# Preprocess the data
table(df$bare_nuclei)
df$bare_nuclei[df$bare_nuclei == "?"] <- NA

rem <- which(is.na(df$bare_nuclei))
df <- df[-rem,]
rm(rem)


df$bare_nuclei <- as.integer(df$bare_nuclei)
df$class <- as.factor(df$class)

df$class <- revalue(df$class, c("2"="benign", "4"="malignant"))
table(df$class)

df <- df[,-1]
str(df)

# Wisualize the dataset
psych::describe(df)

scatterplotMatrix(df,
                  diagonal="histogram",
                  smooth=FALSE)

# KNN ---------------------------------------------------------------------

set.seed(6)

library(kknn)

m <- dim(df)[1]
val <- sample(1:m, size = round(m/5), replace = FALSE, 
              prob = rep(1/m, m)) 
df.learn <- df[-val,]
df.valid <- df[val,]
df.kknn <- kknn(class~., df.learn, df.valid, distance = 5,
                kernel = "triangular")
summary(df.kknn)
fit <- fitted(df.kknn)
table(df.valid$class, fit)
pcol <- as.character(as.numeric(df.valid$class))
pairs(df.valid[1:4], pch = pcol, col = c("green3", "red")
      [(df.valid$class != fit)+1])


library(caret)
library(e1071)

xtab <- table(df.valid$class, fit)
confusionMatrix(xtab)





