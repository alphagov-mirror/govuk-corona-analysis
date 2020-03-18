func_saveRDS <- function(data, output_filepath){
  # Exports data as .RDS to local folder
  #'@param data: data to be saved as an .RDS file
  #'@param output_filepath (string): name of file and path where to save the .RDS (without .RDS extension)
  #'@example save_to_RDS(mydata, "data/copy_of_mydata")
  #'@return .RDS datafile saved at provided filpath
  
  saveRDS(object = data, 
          file = paste0(output_filepath, ".RDS"))
  
  message("Data saved successfully in specified folder")
}
