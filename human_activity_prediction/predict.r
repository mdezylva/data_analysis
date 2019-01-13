library(randomForest)
library(gmodels)
library(neuralnet)
library(RSNNS)
library(Rcpp)
library(lattice)
library(ggplot2)
library(caret)
set.seed(123)

setwd("/home/mitchell/Documents/scratch/data_analysis/human_activity_prediction/UCI HAR Dataset/")
train_data<-read.table("train/X_train.txt")
train_lables<-read.table("train/y_train.txt")

test_data<-read.table("test/X_test.txt")
test_lables<-read.table("test/y_test.txt")

col_names <- readLines("features.txt")
colnames(train_data)<-make.names(col_names)
colnames(test_data)<-make.names(col_names)
colnames(train_lables)<-"lable"
colnames(test_lables)<-"lable"

train_final<-cbind(train_lables,train_data)
test_final<-cbind(test_lables,test_data)
final_data<-rbind(train_final,test_final)
final_data$lable<-factor(final_data$lable)


trial = final_data[ttt,]

model_mlp<-caret::train(lable~.,data=final_data[ttt,],method="mlp")
pre_mlp<-predict(model_mlp,final_data['-ttt',])
table(model_mlp,final_data['-ttt',1])
