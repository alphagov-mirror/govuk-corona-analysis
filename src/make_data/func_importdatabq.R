func_importdatabq <- function(file_query, name_project, name_dataset){
  
  #' queries BigQuery's "cookie_impact_work" sql master tables using the `sql_query` saved in `queries/` folder
  #' 
  #' @include func_readsql.R
  #' @param file_query name of sql file containing the query.
  #' @param name_project name of GCP BQ project.
  #' @param name_dataset name of dataset that file_query calls the table
  
  pckgs <- c("bigrquery", "DBI", "dplyr")
  source("src/make_data/func_readsql.R")
  
  # 1. check dependencies and data type
  sapply(pckgs, function(p) if(!p %in% tolower((.packages()))) stop(paste0("package ", p, " required but not found!")))
  
  if(missing(file_query)) stop(paste0("argument 'sql_query' required but not provided!"))
  if(!is.character(file_query)) stop(paste0("'sql_query' must be of type character"))
  if(!endsWith(file_query, ".sql")) stop(paste0("'",file_query, "' must be a sql file so must ends with '.sql'"))
  
  # 2. establish connection
  id_project <- "govuk-bigquery-analytics"
  conn_pageviews <- bigrquery::dbConnect(drv = bigquery(),
                                         project = name_project,
                                         dataset = name_dataset,
                                         billing = id_project)
  
  # 3. read in SQL query 
  query_bigquery <- func_readsql(file_query = file_query)
  
  # 4. pull in data
  data <- DBI::dbGetQuery(conn = conn_pageviews, statement = query_bigquery) 
  
  return(data)
  
} 