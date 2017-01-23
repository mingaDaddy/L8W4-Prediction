---
title: "Human Activity Recognition"
output: github_document
---

# HTML version of this document
[This page](https://mingadaddy.github.io/L8W4-Prediction/gh-pages/) (located in github.io) contains an HTML version of this document, where performance and prediction tables are evaluated. 

## Executive Summary

Given two datasets (*pml-training.csv* and *pml-testing.csv*) correspongding to the study of **Human Activity Recognition** performed by Velloso, E et al., we developed three diferent predicting models (random forest, decission tree and Quinlans C5.0).

All three models were cross-validated with 30% of the data included in the training dataset. Duration of training, sample error, accuracy and correctly predicted variables from the testing dataset were also noted down to determine which of the three models performed at best.

Random forest and Quinlans C5.0 were able to predict correctly 100% of the data in the testing dataset (decission tree model was only able to achieve a 60% correctness). However, random forest took nearly a 1000% more time to train, and therefore, Quinlans c5.0 model was considered as a most proper predicting model for this kind of task (considering the given data)

## Libraries used

For this project, following libraries were used:

```{r libraries}
library("caret")
library("rpart")
library("C50")
library("plyr")
library("randomForest")
```

## Downloading data and loading in environment

After setting the working directory, we checked if the needed files already existed. If not, they were downloaded:

```{r wdGet}
# set working directory (linux)
setwd("/home/agustin/ml/")

#get data from original source
        if (!file.exists("pml-data/pml-training.csv")) {
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv",
                      destfile = "pml-data/pml-training.csv")
}
if (!file.exists("pml-data/pml-testing.csv")) {
        download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv",
                      destfile = "pml-data/pml-testing.csv")
}
```

Files were then loaded in the environment using the `read.csv` function:

```{r readCsv}
# read the csv files and create data.frames in environment
# Interpret NA and blank fields as NA Strings
pml_training <- read.csv("pml-data/pml-training.csv", na.strings = c("NA",""," "))
pml_testing <- read.csv("pml-data/pml-testing.csv", na.strings = c("NA",""," "))
```

## Cleaning data

The original dataset contains more than 150 variables. However, many of them are empty. We delete those columns from the data.frames:

```{r removeNAs}
# Remove columns with only NAs
pml_training <- pml_training[,colSums(is.na(pml_training)) == 0]
pml_testing <- pml_testing[,colSums(is.na(pml_testing)) == 0]
```

Also, initial variables are only used for identification, and those are not usefull for our project. We delete them also:

```{r removeIds}
# remove identification columns as they are not needed for the project
pml_training <- pml_training[, -c(1:7)]
pml_testing <- pml_testing[, -c(1:7)]
```

## Spliting data for cross-validation:

We are splitting data from the dataset `training` into a train subSet and a test subSet. The proportions are 0.70 and 0.30 respectively. While splitting the data, we are able to make some cross-validation for the different models before predicting them in the final `testing` dataset:

```{r split}
# split the cleaned data into training and testing (70%-30%)
# for doing some cross validation
inTrain <- createDataPartition(y = pml_training$classe, p = 0.7, list = FALSE)
training <- pml_training[inTrain, ]
testing <- pml_training[-inTrain, ]
```

## Model 1 - Random Forest:

We trained the model

```{r rfTrain}
#set seed for reproducibility
set.seed(1978)
#### Model 1: Random Forest
# fit a random forest model to predict classe
# calculate training time
rfStartTraining <- Sys.time()
rfModel <- randomForest(as.factor(training$classe) ~ ., data = training)
rfEndTraining <- Sys.time()
```

and cross-validated it.

```{r rfcv}
# crossvalidate the model using the remaining 30% of data
rfPredictor <- predict(rfModel, testing)
summaryrfPredictor <- confusionMatrix(testing$classe, rfPredictor)
```

## Model 2 - Decission trees:

We trained the model

```{r dtTrain}
#set seed for reproducibility
set.seed(1978)
#### Model 2: decission tree
# fit a rpart model to predict classe
dtStartTraining <- Sys.time()
dtModel <- rpart(as.factor(training$classe) ~ ., data = training)
dtEndTraining <- Sys.time()
```

and cross-validated it.

```{r dtcv}
# crossvalidate the model using the remaining 30% of data
dtPredictor <- predict(dtModel, testing, na.action = na.pass, type = "class")
summarydtPredictor <- confusionMatrix(testing$classe, dtPredictor)
```

## Model 3 - Quilan's c5.0:

We trained the model

```{r c50Train}
#set seed for reproducibility
set.seed(1978)
#### Model 3: Quinlan’s C5.0 algorithm
# fit a Quinlan’s C5.0 algorithm model to predict classe
c50StartTraining <- Sys.time()
c50Model <- C5.0(as.factor(training$classe) ~ ., data = training, rules = TRUE)
c50EndTraining <- Sys.time()
```

and cross-validated it.

```{r c50cv}
# crossvalidate the model using the remaining 30% of data
c50Predictor <- predict(c50Model, testing, na.action = na.pass, type = "class")
summaryc50Predictor <- confusionMatrix(testing$classe, c50Predictor)
```

## Predicting the testing dataset with the trained models:

We predict the testing dataset using each of the trained models. Note that the correct prediction was noted down from the responses of the quiz corresponding to this project:

```{r prediction}
### Predict from testing data.frame using the three models
predictionTesting <- data.frame("id" = pml_testing$problem_id,
                           "correct" = c("B","A","B","A","A","E","D","B","A","A","B","C","B","A","E","E","A","B","B","B"),
                           "rf"  = predict(rfModel, pml_testing, type = "class"),
                           "dt" = predict(dtModel, pml_testing, type = "class"),
                           "c50" = predict(c50Model, pml_testing, type = "class"))
predictionTesting
```

We observe that random forest and Quilan's c5.0 performed better than the decission tree model (100% of the predictions were correct), but we still have to decide which model is better taking other factors into consideration.

## Measuring sample error, accuracy and training duration:

While performing the training of the different models, we calculated the training duration using the `sys.Time()` function from base R. Along with the sample error and accuracy calculation, those metrics will give us better arguments to decide which of the two best models performed better.

```{r performance}
## calculate out of sample error (random Forest)
rfcorrectClassificatedObservations <- sum(rfPredictor == testing$classe)
rftotalObservations <- length(testing$classe)
rfaccuracyPrediction <- rfcorrectClassificatedObservations / rftotalObservations
rfsampleError <- 1 - rfaccuracyPrediction

## calculate out of sample error (dt)
dtcorrectClassificatedObservations <- sum(dtPredictor == testing$classe)
dttotalObservations <- length(testing$classe)
dtaccuracyPrediction <- dtcorrectClassificatedObservations / dttotalObservations
dtsampleError <- 1 - dtaccuracyPrediction

## calculate out of sample error (c50)
c50correctClassificatedObservations <- sum(c50Predictor == testing$classe)
c50totalObservations <- length(testing$classe)
c50accuracyPrediction <- c50correctClassificatedObservations / c50totalObservations
c50sampleError <- 1 - c50accuracyPrediction

# create performance Table
performanceTable <- data.frame("model" = c("rf","dt","c50"),
                               "model Accuracy" = c(summaryrfPredictor$overall[[1]],
                                                    summarydtPredictor$overall[[1]],
                                                    summaryc50Predictor$overall[[1]]),
                               "training Duration" = c(as.numeric(rfEndTraining - rfStartTraining),
                                                       as.numeric(dtEndTraining - dtStartTraining),
                                                       as.numeric(c50EndTraining - c50StartTraining)),
                               "correctly Assigned" = c(length(which(predictionTesting$rf == predictionTesting$correct)),
                                                        length(which(predictionTesting$dt == predictionTesting$correct)),
                                                        length(which(predictionTesting$c50 == predictionTesting$correct))),
                               "sample error" = c(rfsampleError,
                                                  dtsampleError,
                                                  c50sampleError))

performanceTable
```

From the table above we observe that although the random forest has slightly better figures in matter of sample error and model accuracy than the Quilian's c5.0, the duration of training is considerable bigger (random forest is 950% slower than Quilian's Algorythm).

## Conclusion:

Although for the given datasets the random forest and Quilan's models perform extraordinary good, the training duration of the former is much bigger. Therefore, for this kind of tasks and data, we recommend using the Quilan's algorythm as it has a better balance in accuracy and performance.
