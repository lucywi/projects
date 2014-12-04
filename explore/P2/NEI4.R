## Fine particulate matter (PM2.5) is an ambient air pollutant for which there 

## is strong evidence that it is harmful to human health. Approximately every 3 years, 

## the EPA releases its database on emissions of PM2.5. This database is known as 

## the National Emissions Inventory (NEI). For each year and for each type of PM 

## source, the NEI records how many tons of PM2.5 were emitted from that source 

## over the course of the entire year. This plot helps to determine how 

## emissions from coal combustion-related sources have changed from 1999–2008 across

## the US.

## read in data

NEI <- readRDS("/Users/lucykwilliams/Desktop/R/1 data/NEI_data/summarySCC_PM25.rds")

SCC <- readRDS("/Users/lucykwilliams/Desktop/R/1 data/NEI_data/

Source_Classification_Code.rds")

# merge NEI and SCC data using the SCC number as the key variable

NEI.SCC <- merge(NEI, SCC, by="SCC")

## subset year and coal emissions

year <- NEI.SCC[, 6]

emissions <- NEI.SCC[, 4]

anthracite.coal <- NEI.SCC[, 13]

pulverized.coal <- NEI.SCC[, 14]

## create new df binding year, emissions, anthacite coal, and pulverized coal

coal <- cbind(year, emissions, anthracite.coal, pulverized.coal)

coal <- as.data.frame(coal)

## plot coal emissions v. year

qplot(year, data = coal, ylab = "Coal Emissions", xlab= " Year")
