par(mfcol = c(2,2))

## plot2
pc_2.1.2007 <- read.csv.sql(file="/Users/lucykwilliams/Desktop/R/1 data/household_power_consumption.txt", sql = "select * from file where Date = '1/2/2007'", header = TRUE, sep = ";")
pc_2.2.2007 <- read.csv.sql(file="/Users/lucykwilliams/Desktop/R/1 data/household_power_consumption.txt", sql = "select * from file where Date = '2/2/2007'", header = TRUE, sep = ";")
df <- rbind(pc_2.1.2007, pc_2.2.2007)
x <- as.POSIXct(strptime(paste(df$Date, df$Time), "%d/%m/%Y %H:%M:%S"))  
y <- df[, 3]
plot(x, y, type = "l", ylab = paste("Global Active Power (kilowatts)"), xlab = "")

## plot3
y <- df[, 7] 
plot(x, y, type = "l", ylab = paste("Energy sub metering"), xlab = "")
y <- df[, 8]
lines(x, y, type = "l", col = "red")
y <- df[, 9] 
lines(x, y, type = "l", col = "blue")
legend("topright", lty = 1, col = c("black", "red", "blue"), legend = c("Sub_metering_1","Sub_metering_2","Sub_metering_3"), bty="n",  cex = 0.65, xjust = 1, y.intersp = 1.1, inset = 0.15)

## voltage
y <- df[, 5] 
plot(x, y, type = "l", ylab = paste("Voltage"), xlab = "datetime")

## plot grp
y <- df[, 4] 
plot(x, y, type = "l", ylab = paste("Global_reactive_power"), xlab = "datetime")
