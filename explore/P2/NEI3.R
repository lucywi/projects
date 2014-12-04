## Fine particulate matter (PM2.5) is an ambient air pollutant for which there 

## is strong evidence that it is harmful to human health. Approximately every 3 years, 

## the EPA releases its database on emissions of PM2.5. This database is known as 

## the National Emissions Inventory (NEI). For each year and for each type of PM 

## source, the NEI records how many tons of PM2.5 were emitted from that source 

## over the course of the entire year. This plot helps to determine whether

## four sources: point, nonpoint, onroad, or nonroad has increased or decreased

## emissions from 1999 to 2008in Baltimore City.

## read in data

NEI <- readRDS("/Users/lucykwilliams/Desktop/R/1 data/NEI_data/summarySCC_PM25.rds")

## subset emissions, type, and year

emissions <- NEI[, 4]

type <- (NEI[, 5])

year <- NEI[, 6]

## create new dataframe

e.t.y <- cbind(emissions, type, year)

e.t.y <- as.data.frame(e.t.y) 

## subset Baltimore City, Maryland (fips == "24510")

b <- subset(e.t.y, fips == "24510")

## plot emissions types v. year by type for Baltimore City

qplot(year, data = b, fill = type)
