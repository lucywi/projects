## read 2.1.2007 data
pc_2.1.2007 <- read.csv.sql(file="/Users/lucykwilliams/Desktop/R/1 data/household_power_consumption.txt", sql = "select * from file where Date = '1/2/2007'", header = TRUE, sep = ";")

## read 2.2.2007 data
pc_2.2.2007 <- read.csv.sql(file="/Users/lucykwilliams/Desktop/R/1 data/household_power_consumption.txt", sql = "select * from file where Date = '2/2/2007'", header = TRUE, sep = ";")

## create data frame which rbinds 2.1.2007 and 2.2.2007 together
df <- rbind(pc_2.1.2007, pc_2.2.2007)

## create histogram of global active power v. frequency
hist(df$Global_active_power, col ="red", 
     main = paste("Global Active Power"),
     xlab = paste("Global Active Power (kilowatts)"), 
     ylab = paste("Frequency"), 
     axes = T)
