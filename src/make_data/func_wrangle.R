func_wrangle <- function(data){
  
  # deletes 20th Dec for more equal comparison across years
  # re-aggregates days into more sensible monthly period for analysis
  # so that days after 20th Dec 2019 are not conflated into same monthly period as days before 20th Dec 2019
  # Ref: From cookies analysis by Alessia Tosi
  
  #'@param data_bq: data fetched from BigQuery, results of src/ingest_BQdata.R 
  #'@return original dataset with additional columns: 
  #' date_period, date_period_year, date_period_month, date_week_n, date_period_month_week 
  
  pckgs <- c("dplyr", "stringr")
  require("dplyr", "stringr")
  if(!"date" %in% colnames(data)) stop(paste0("column 'date' is missing from dataset"))
  sapply(pckgs, function(p) if(!p %in% rownames(installed.packages())) stop(paste0("package ", p, " required but not found!")))
  
  data <- data %>%
    arrange(desc(date)) %>% 
    mutate(date_year = factor(date_year, levels = rev(sort(unique(date_year)))),
           date_period = paste0(date_year, " ", date_month_nm),
           date_period = factor(x = date_period, levels = unique(date_period), ordered = TRUE))
  
  # order also date_period
  # data$date_period <- factor(data$date_period, 
  #                            levels = unique(data$date_period[order(data$date_period_year, data$date_period_month)]), 
  #                            ordered = TRUE)
  
  return(data)
}
