## Fine particulate matter (PM2.5) is an ambient air pollutant for which there 

## is strong evidence that it is harmful to human health. Approximately every 3 years, 

## the EPA releases its database on emissions of PM2.5. This database is known as 

## the National Emissions Inventory (NEI). For each year and for each type of PM 

## source, the NEI records how many tons of PM2.5 were emitted from that source 

## over the course of the entire year. This plot helps to determine how 

## emissions from motor vehicle sources have changed from 1999–2008 in Baltimore City. 

## read in data

NEI <- readRDS("/Users/lucykwilliams/Desktop/R/1 data/NEI_data/summarySCC_PM25.rds")

## subset Baltimore City, Maryland (fips == "24510")

b <- subset(NEI, fips == "24510")

## subset type = ON-ROAD

onroad <- subset(b, type == "ON-ROAD")

year <- onroad[, 6]

emissions <- onroad[, 4]

## create new df with year, onroad, and emissions

onroad.emissions <- cbind(year, onroad, emissions)

onroad.emissions <- as.data.frame(onroad.emissions)

## plot

barplot(table(onroad.emissions$year), col = "blue", main = "Onroad emissions in Baltimore by 

Year")
