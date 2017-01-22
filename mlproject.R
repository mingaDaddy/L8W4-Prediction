library("caret")
library("rpart")
library("C50")
library("plyr")

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

# read the csv files and create data.frames in environment
# Interpret NA and blank fields as NA Strings
pml_training <- read.csv("pml-data/pml-training.csv", na.strings = c("NA",""," "))
pml_testing <- read.csv("pml-data/pml-testing.csv", na.strings = c("NA",""," "))

# Remove columns with only NAs
pml_training <- pml_training[,colSums(is.na(pml_training)) == 0]
pml_testing <- pml_testing[,colSums(is.na(pml_testing)) == 0]

# remove identification columns as they are not needed for the project
pml_training <- pml_training[, -c(1:7)]
pml_testing <- pml_testing[, -c(1:7)]


# split the cleaned data into training and testing (70%-30%)
# for doing some cross validation
inTrain <- createDataPartition(y = pml_training$classe, p = 0.7, list = FALSE)
training <- pml_training[inTrain, ]
testing <- pml_training[-inTrain, ]

#set seed for reproducibility
set.seed(1978)

#### Model 1: Random Forest
# fit a random forest model to predict classe
# calculate training time
rfStartTraining <- Sys.time()
rfModel <- randomForest(as.factor(training$classe) ~ ., data = training)
rfEndTraining <- Sys.time()

# crossvalidate the model using the remaining 30% of data
rfPredictor <- predict(rfModel, testing)
summaryrfPredictor <- confusionMatrix(testing$classe, rfPredictor)

#### Model 2: decission tree
# fit a rpart model to predict classe
dtStartTraining <- Sys.time()
dtModel <- rpart(as.factor(training$classe) ~ ., data = training)
dtEndTraining <- Sys.time()

# crossvalidate the model using the remaining 30% of data
dtPredictor <- predict(dtModel, testing, na.action = na.pass, type = "class")
summarydtPredictor <- confusionMatrix(testing$classe, dtPredictor)

#### Model 3: Quinlan’s C5.0 algorithm
# fit a Quinlan’s C5.0 algorithm model to predict classe
c50StartTraining <- Sys.time()
c50Model <- C5.0(as.factor(training$classe) ~ ., data = training, rules = TRUE)
c50EndTraining <- Sys.time()

# crossvalidate the model using the remaining 30% of data
c50Predictor <- predict(c50Model, testing, na.action = na.pass, type = "class")
summaryc50Predictor <- confusionMatrix(testing$classe, c50Predictor)

### Predict from testing data.frame using the three models
predictionTesting <- data.frame("id" = pml_testing$problem_id,
                           "correct" = c("B","A","B","A","A","E","D","B","A","A","B","C","B","A","E","E","A","B","B","B"),
                           "rf"  = predict(rfModel, pml_testing, type = "class"),
                           "dt" = predict(dtModel, pml_testing, type = "class"),
                           "c50" = predict(c50Model, pml_testing, type = "class"))

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
