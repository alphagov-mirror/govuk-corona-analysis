library(here)
library(dplyr)
library(ggplot2)
library(ggridges)

data_bq <- readRDS(file = here("data", "bg_pgviews_devicecategory.RDS"))


func_plot_densities_pgvwsShares_timeline <- function(data, 
                                                     characteristic_var=""){
  
  #'@param data : aggregated dataset with proportion of pageviews by `characteristic_var`, result of `func_calc_hshares_by_monthCat.R`
  #'@param characteristic_var
  #'@description Returns a ridgeline plot of the density distribution of proportions of hourly
  #' pageviews by each characteristic category over months
  
  # checks
  required_cols <- c("date_period", "datetime_hour", "total_pageviews", "prop_pageviews")
  sapply(required_cols, function(col) if(!col %in% colnames(data)) stop(paste0("column", col, " is missing from dataset"))
  )
  
  # to allow use with dplyr and ggplot2: covert to symbol and then unquote it using !!
  sym_charac <- ggplot2::sym(characteristic_var)
  
  # number of unique categories
  n_cols <- length(unique(data[[characteristic_var]]))
  
  # timeline of distributions of hourly pageview shares over months by category
  data %>%
    ggplot(., 
           aes(x = prop_pageviews, y = date_period, fill=stat(x)) #fill shades change according to x-values
    ) + 
    ggridges::geom_density_ridges_gradient(
      quantile_lines = TRUE, quantiles=2,
      scale=0.9
    ) +
    scale_fill_distiller(palette="Reds", direction = 1) +
    scale_y_discrete(expand_scale(mult = c(0.01, 1))) +
    scale_x_continuous(expand=c(0,0)) +
    coord_cartesian(clip = "off") +
    facet_wrap(reformulate(characteristic_var), ncol=n_cols) +
    labs(
      x = "share (i.e., proportion) of hourly page views",
      y = "year and month",
      title = paste("Density distribution of shares of hourly pageviews by", characteristic_var, "over monthly periods"),
      subtitle = "Monthly periods start on the 21st of month 'k' and end on 20th of month 'k+1'",
      caption = "20th of December was removed from all years"
    )
}
