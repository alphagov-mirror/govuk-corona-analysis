func_importdatabq <- function(sql_query){
  
  #' queries BigQuery's "cookie_impact_work" sql master tables using the `sql_query` saved in `queries/` folder
  #' 
  #' @include func_readsql.R
  #' @param sql_query name of sql file containing the query saved in `queries/`.
  
  pckgs <- c("bigrquery", "DBI", "dplyr")
  source("src/make_data/func_readsql.R")
  
  # 1. check dependencies and data type
  sapply(pckgs, function(p) if(!p %in% tolower((.packages()))) stop(paste0("package ", p, " required but not found!")))
  
  if(missing(sql_query)) stop(paste0("argument 'sql_query' required but not provided!"))
  if(!is.character(sql_query)) stop(paste0("'sql_query' must be of type character"))
  if(!endsWith(sql_query, ".sql")) stop(paste0("'",sql_query, "' must be a sql file so must ends with '.sql'"))
  
  # 2. establish connection
  id_project <- "govuk-bigquery-analytics"
  conn_pageviews <- bigrquery::dbConnect(drv = bigquery(),
                                         project = "govuk-bigquery-analytics",
                                         dataset = "cookie_impact_work",
                                         billing = id_project)
  
  # 3. read in SQL query 
  query_bigquery <- func_readsql(file = paste0("queries/", sql_query))
  
  # 4. pull in data
  data_pageviews <- DBI::dbGetQuery(conn = conn_pageviews, statement = query_bigquery) 
  
  return(data_pageviews)
  
} 