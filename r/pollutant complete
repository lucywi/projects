complete <- function(directory, pollutant, id = 1:332, removeNA = TRUE) {  				                         
  for (i in 1:332) {
    ## location of files
    directory <- ("/Users/lucykwilliams/Desktop/R/specdata")
    
    ## read csv into dataframe
    data <- ldply(dir("specdata", full=T), read.csv)	
    
    ## subset ID
    ID <- data[ ,4]
    
    ## subset nobs
    nobs <- complete.cases(data)
    
    ## bind ID and nobs into a new dataframe 
    id_nobs <- data.frame(col1 = ID, col2 = nobs, na.rm=TRUE)  
  }
}
