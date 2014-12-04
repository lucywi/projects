pollutantmean2 <- function(directory, pollutant, id = 1:332, removeNA = TRUE) {					                         
  for (i in 1:332) {
    ## location of files
    directory <- ("/Users/lucykwilliams/Desktop/R/specdata")
    
    ## read csv into dataframe
    data <- ldply(dir("specdata", full=T), read.csv)	
    
    ## subset sulfate
    sulfate <- data[ ,2]
    
    ## subset nitrate
    nitrate <- data[ , 3]	
    
    ## mean
    mean(data[,pollutant],na.rm=TRUE)
  }
}
