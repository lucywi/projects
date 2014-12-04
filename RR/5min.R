## Make a time series plot (i.e. type = "l") of the 5-minute interval (x-axis) 
## and the average number of steps taken, averaged across all days (y-axis)

## Which 5-minute interval, on average across all the days in the dataset, 
## contains the maximum number of steps?

## read in data
activity <- read.csv("~/Desktop/R/1 data/reproducible research/activity.csv")

## remove NAs
activity <- na.omit(activity)

## subset time interval
x <- activity[, 3]

## subset average # of steps taken averaged across all days
y <- aggregate(.~interval, FUN = mean, data=activity)


## plot
plot(x, y, type = "l")
