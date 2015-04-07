pc_2.1.2007 <- read.csv.sql(file="/Users/lucykwilliams/Desktop/R/1 data/household_power_consumption.txt", sql = "select * from file where Date = '1/2/2007'", header = TRUE, sep = ";")

pc_2.2.2007 <- read.csv.sql(file="/Users/lucykwilliams/Desktop/R/1 data/household_power_consumption.txt", sql = "select * from file where Date = '2/2/2007'", header = TRUE, sep = ";")

df <- rbind(pc_2.1.2007, pc_2.2.2007)

x <- as.POSIXct(strptime(paste(df$Date, df$Time), "%d/%m/%Y %H:%M:%S"))  
y <- df[, 5] 

plot(x, y, type = "l", ylab = paste("Voltage"), xlab = "datetime")
