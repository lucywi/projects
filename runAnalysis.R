## 1. Merges the training and the test sets to create one data set.
## 2. Extracts only the measurements on the mean and standard 
## deviation for each measurement. 
## 3. Uses descriptive activity names to name the activities 
## in the data set
## 4. Appropriately labels the data set with descriptive variable names. 
## 5. Creates a second, independent tidy data set with the 
## average of each variable for each activity and each subject. 
  
      ## read in train data
      subjectTrain <- read.table("/Users/lucykwilliams/Desktop/R/1 data/cleaning data/wearables/train/subject_train.txt", header = F)
      xTrain <- read.table("/Users/lucykwilliams/Desktop/R/1 data/cleaning data/wearables/train/X_train.txt", header = F)
      yTrain <- read.table("/Users/lucykwilliams/Desktop/R/1 data/cleaning data/wearables/train/Y_train.txt", header = F)
  
      ## create a train df
      train <- cbind(subjectTrain, xTrain, yTrain)
  
      ## read in test data
      subjectTest <- read.table("/Users/lucykwilliams/Desktop/R/1 data/cleaning data/wearables/test/subject_test.txt", header = F)
      xTest <- read.table("/Users/lucykwilliams/Desktop/R/1 data/cleaning data/wearables/test/X_test.txt", header = F)
      yTest <- read.table("/Users/lucykwilliams/Desktop/R/1 data/cleaning data/wearables/test/Y_test.txt", header = F)
      
      ## create a test df
      test <- cbind(subjectTest, xTest, yTest)
      
      ## merge train and test datasets
      trainTest <- rbind(test, train)
      
      ## read in the features file with column names
      features <- read.table("/Users/lucykwilliams/Desktop/R/1 data/cleaning data/wearables/features.txt", header=F, as.is=T, col.names=c("MeasurementID", "MeasurementName"))
      
      ## assign column names to trainTest data set
      names(trainTest)[2:562] <- features[, 2]
      subjectID <- names(trainTest)[1]<-"Subject.ID"
      names(trainTest)[563] <- "Activity.ID"
      
      ## read in activity labels
      activity <- read.table("/Users/lucykwilliams/Desktop/R/1 data/cleaning data/wearables/activity_labels.txt", header=F, as.is=T, col.names=c("Activity.ID", "Activity.Name"))
      
      ## merge trainTest and activity by ActivityID to have the corresponding ActivityName listed
      trainTest <- merge(activity, trainTest, by = "Activity.ID")
      trainTest <- trainTest[, 2:564]
      
      ## extract only the mean and standard deviations for each measurement
      mean.std <- grep(".*mean\\(\\)|.*std\\(\\)", features$MeasurementName)
      trainTest <- trainTest[ , mean.std]
      
      
#########

      ## Create a second, independent tidy data set with the average of 
      ## each variable for each activity and each subject. 
      
      ## melt trainTest data frame      
      trainTestMelt <- melt(trainTest, id=c("Subject.ID", "Activity.Name"))
      
      ## recast and create a tidy data set which contains the the average of 
      ## each variable for each activity and each subject. 
      tidy <- dcast(trainTestMelt, Activity.Name + Subject.ID ~ variable, mean)     
      
      ## write table and save as a text file
      tidy <- write.table(tidy, "/Users/lucykwilliams/Desktop/R/1 data/cleaning data/tidy.txt")
