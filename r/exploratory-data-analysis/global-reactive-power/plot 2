## read 2.1.2007 data
pc_2.1.2007 <- read.csv.sql(file="/Users/lucykwilliams/Desktop/R/1 data/household_power_consumption.txt", sql = "select * from file where Date = '1/2/2007'", header = TRUE, sep = ";")

## read 2.2.2007 data
pc_2.2.2007 <- read.csv.sql(file="/Users/lucykwilliams/Desktop/R/1 data/household_power_consumption.txt", sql = "select * from file where Date = '2/2/2007'", header = TRUE, sep = ";")

## create data frame which rbinds 2.1.2007 and 2.2.2007 together
df <- rbind(pc_2.1.2007, pc_2.2.2007)

## create plot of global active power in kilowatts for 2.1.2007 to 2.2.2007
x <- as.POSIXct(strptime(paste(df$Date, df$Time), "%d/%m/%Y %H:%M:%S"))  
y <- df[, 3]
plot(x, y, type = "l", ylab = paste("Global Active Power (kilowatts)"), xlab = "")
