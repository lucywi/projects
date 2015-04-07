## Fine particulate matter (PM2.5) is an ambient air pollutant for which there 

## is strong evidence that it is harmful to human health. Approximately every 3 years, 

## the EPA releases its database on emissions of PM2.5. This database is known as 

## the National Emissions Inventory (NEI). For each year and for each type of PM 

## source, the NEI records how many tons of PM2.5 were emitted from that source 

## over the course of the entire year. This plot compare emissions from motor vehicle 

## sources in Baltimore City with emissions from motor vehicle sources in Los 

## Angeles County, California and shows which city has seen greater changes 

## over time in motor vehicle emissions.

## read in data

NEI <- readRDS("/Users/lucykwilliams/Desktop/R/1 data/NEI_data/summarySCC_PM25.rds")

## subset Baltimore City, Maryland (fips == "24510")

b <- subset(NEI, fips == "24510")

## subset type = ON-ROAD

onroad.b <- subset(b, type == "ON-ROAD")

## subset LA (fips == "06037")

LA <- subset(NEI, fips == "06037")

## subset type = ON-ROAD

onroad.LA <- subset(LA, type == "ON-ROAD")
