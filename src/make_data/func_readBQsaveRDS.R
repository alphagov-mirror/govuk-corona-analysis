func_readBQsaveRDS <- function(file_query, name_project, name_dataset, output_filepath){
  # reads from SQL file, imports from BQ, then saves output as .RDS file
  #'@param file_query (string): file of query that you want to save the results of as an RDS file
  #'@param name_project name of GCP BQ project.
  #'@param name_dataset name of dataset that file_query calls the table
  #'@param output_filepath (string): name of file and path where to save the .RDS (without .RDS extension)
  #'@example func_readBQsaveRDS(file_query = myquery, name_project = "myproject", name_dataset = "namedataset", output_filepath = "myfilepath")
  #'@return message saying .RDS datafile saved at provided output_filepath
  
  source("src/make_data/func_importdatabq.R")
  
  data <- func_importdatabq(file_query = file_query, name_project = name_project, name_dataset = name_dataset)
  
  saveRDS(object = data, file = paste0(output_filepath, ".RDS"))
  
  message("Data saved successfully in folder, ", output_filepath)
}
