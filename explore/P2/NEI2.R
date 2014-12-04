## Fine particulate matter (PM2.5) is an ambient air pollutant for which there 

## is strong evidence that it is harmful to human health. Approximately every 3 years, 

## the EPA releases its database on emissions of PM2.5. This database is known as 

## the National Emissions Inventory (NEI). For each year and for each type of PM 

## source, the NEI records how many tons of PM2.5 were emitted from that source 

## over the course of the entire year. This plot helps to determine whether total 

## emissions from PM2.5 decreased in Baltimore City, MD from 1999 to 2008.

## read in data

NEI <- readRDS("/Users/lucykwilliams/Desktop/R/1 data/NEI_data/summarySCC_PM25.rds")

## subset year and emissions

year <- NEI[, 6]

emissions <- NEI[, 4]

## new dataframe with year and emissions

emissions.year <- cbind(year, emissions)

emissions.year <- as.data.frame(emissions.year)

## subset Baltimore City, Maryland (fips == "24510")

b <- subset(emissions.year, fips == "24510")

## plot emissions v. year

barplot(table(b$year), col = "blue", main = "Total PM2.5 emissions in Baltimore City, MD by 

year")
