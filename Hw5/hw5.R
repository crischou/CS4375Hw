#David Tran and Cris Chou
#09/26/2021

library(tidyverse)
library(caret)
library(e1071)

#Load data
df <- read.csv("titanic_project.csv")

#Separate Train/Split to have 900 variables for Train
smp_size <- floor(0.861 * nrow(df))
set.seed(1234)
train_ind <- sample(seq_len(nrow(df)), size = smp_size)
train <- df[train_ind,]
test <- df[-train_ind,]

#Logistic Regression
ptm <- proc.time();
glm1 <- glm(survived ~ pclass, data = train, family = binomial)
ptm2 <- proc.time();
cat("The elapsed time is: ",ptm2 - ptm)

summary(glm1)

#Prediction, Accuracy, and Sensitivity, and Specificity
glm2 <- glm(survived ~ pclass, data = test, family = binomial)
pred <- predict(glm2, newdata = test, type = "response")
pred1 <- ifelse(pred > .5, 1, 0)
acc1 <- mean(pred1 == as.integer(test$survived))
cat("Accuracy: ", acc1, "\n")

confusionMatrix(as.factor(pred1), as.factor(test$survived), positive = "1")

sens <- sensitivity(as.factor(pred1), as.factor(test$survived), positive = "1")
cat("Sensitivity: ", sens, "\n")

spec <- specificity(as.factor(pred1), as.factor(test$survived), negative = "0")
cat("Specificity: ", spec, "\n")

#Naive Bayes
nb1 <- naiveBayes(survived ~ pclass + sex + age, data = train)
ptm2 <- proc.time();
cat("The elapsed time is: ",ptm2 - ptm)
nb1

#Test Naive Bayes
nb2 <- naiveBayes(survived ~ pclass + sex + age, data = test)
ptm2 <- proc.time();
cat("The elapsed time is: ",ptm2 - ptm)
nb2

#Prediction, Accuracy, and Sensitivity, and Specificity
predBayes <- predict(nb2, newdata = test, type = "class")
confusionMatrix(as.factor(predBayes), as.factor(test$survived), positive="1")
accBayes <- mean(predBayes == as.integer(test$survived))
cat("Accuracy: ", accBayes, "\n")

sensBayes <- sensitivity(as.factor(predBayes), as.factor(test$survived), positive = "1")
cat("Sensitivity: ", sensBayes, "\n")

specBayes <- specificity(as.factor(predBayes), as.factor(test$survived), negative = "0")
cat("Specificity: ", specBayes, "\n")


