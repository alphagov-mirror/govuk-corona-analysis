library(here)

library(dplyr)
library(stringr)

library(ggplot2)
library(ggridges)

data_bq <- readRDS(file = here("data", "bg_pgviews_devicecategory.RDS"))

data_bq <- data_bq %>%
  filter(!date %in% c(as.Date("2018-12-20"), as.Date("2019-12-20"))) %>%
  mutate(date_year = factor(x = date_year, levels = rev(sort(unique(date_year)))),
         date_period = case_when(
           (date >= "2017-12-21" & date <= "2018-01-20") ~ "2017/18 Dec/Jan",
           (date >= "2018-01-21" & date <= "2018-02-20") ~ "2018 Jan/Feb",
           (date >= "2018-02-21" & date <= "2018-03-20") ~ "2018 Feb/Mar",
           (date >= "2018-03-21" & date <= "2018-04-20") ~ "2018 Mar/Apr",
           (date >= "2018-04-21" & date <= "2018-05-20") ~ "2018 Apr/May",
           (date >= "2018-05-21" & date <= "2018-06-20") ~ "2018 May/Jun",
           (date >= "2018-06-21" & date <= "2018-07-20") ~ "2018 Jun/Jul",
           (date >= "2018-07-21" & date <= "2018-08-20") ~ "2018 Jul/Aug",
           (date >= "2018-08-21" & date <= "2018-09-20") ~ "2018 Aug/Sep",
           (date >= "2018-09-21" & date <= "2018-10-20") ~ "2018 Sep/Oct",
           (date >= "2018-10-21" & date <= "2018-11-20") ~ "2018 Oct/Nov",
           (date >= "2018-11-21" & date <= "2018-12-19") ~ "2018 Nov/Dec",
           (date >= "2018-12-21" & date <= "2019-01-20") ~ "2018/19 Dec/Jan",
           (date >= "2019-01-21" & date <= "2019-02-20") ~ "2019 Jan/Feb",
           (date >= "2019-02-21" & date <= "2019-03-20") ~ "2019 Feb/Mar",
           (date >= "2019-03-21" & date <= "2019-04-20") ~ "2019 Mar/Apr",
           (date >= "2019-04-21" & date <= "2019-05-20") ~ "2019 Apr/May",
           (date >= "2019-05-21" & date <= "2019-06-20") ~ "2019 May/Jun",
           (date >= "2019-06-21" & date <= "2019-07-20") ~ "2019 Jun/Jul",
           (date >= "2019-07-21" & date <= "2019-08-20") ~ "2019 Jul/Aug",
           (date >= "2019-08-21" & date <= "2019-09-20") ~ "2019 Aug/Sep",
           (date >= "2019-09-21" & date <= "2019-10-20") ~ "2019 Sep/Oct",
           (date >= "2019-10-21" & date <= "2019-11-20") ~ "2019 Oct/Nov",
           (date >= "2019-11-21" & date <= "2019-12-19") ~ "2019 Nov/Dec",
           (date >= "2019-12-21" & date <= "2020-01-20") ~ "2019/20 Dec/Jan",
           (date >= "2020-01-21" & date <= "2020-02-20") ~ "2020 Jan/Feb",
           (date >= "2020-02-21" & date <= "2020-03-20") ~ "2020 Feb/Mar",
           TRUE ~ NA_character_)) %>%
  mutate(date_period = factor(date_period, levels = unique(date_period))) 


library(assertthat)

func_calc_hshares_by_monthCat <- function(data, 
                                          characteristic_var=""){
  
  #'@param data (data.frame) : dataset  
  #'@param characteristic_var (character string) : name of the characteristic variable 
  #'@description Calculates shares/proportions of hourly pageviews for 
  #'each characteristic category for each monthly period
  #'
  #'@return aggregated dataframe containing number and proportion of hourly pageviews by 'date_period' and characteristic_var 
  
  # checks
  required_cols <- c("date_period", "datetime_hour", "pageviews")
  
  data %>%
    assertthat::not_empty() %>%
    assertthat::has_name(required_cols)
  
  
  # to allow use with dplyr and ggplot2: covert to symbol and then unquote it using !!
  sym_charac <- dplyr::sym(characteristic_var)
  
  agg_data = tryCatch({
    
    data %>%
      group_by(
        date_period, 
        datetime_hour, 
        !!sym_charac) %>% 
      # sum of all pageviews for each day-hour for each category
      summarise(
        total_pageviews = sum(as.numeric(pageviews), na.rm = TRUE)
      ) %>%
      # calculate proportion of hourly pageviews by category  
      mutate(
        prop_pageviews = total_pageviews/(sum(total_pageviews))
      )
    
  }, warning = function(w) {
    message(paste0('** WARNING when applying `func_calc_hshares_by_monthCat()` with "', characteristic_var, '" **'))
    print(w)
    
  }, error = function(e) {
    message(paste0('** ERROR when applying `func_calc_hshares_by_monthCat()` with "', characteristic_var, '" **'))
    print(e)
    
  }, finally={
  })
  
  return(agg_data)
  
}




# number of unique categories
n_cols <- length(unique(data_bq[["characteristic_var"]]))
n_categories <- length(x = unique(x = data_bq$deviceCategory))

data_bq %>%
  #fill shades change according to x-values
  ggplot(mapping = aes(x = prop_pageviews, y = date_period, fill = stat(x))) + 
  geom_density_ridges_gradient(quantile_lines = TRUE, quantiles = 2, scale = 0.9) +
  scale_fill_distiller(palette="Reds", direction = 1) +
  scale_y_discrete(expand_scale(mult = c(0.01, 1))) +
  scale_x_continuous(expand=c(0,0)) +
  coord_cartesian(clip = "off") +
  facet_wrap(reformulate(deviceCategory), ncol = n_categories) +
  labs(
    x = "share (i.e., proportion) of hourly page views",
    y = "year and month",
    title = paste("Density distribution of shares of hourly pageviews by", characteristic_var, "over monthly periods"),
    subtitle = "Monthly periods start on the 21st of month 'k' and end on 20th of month 'k+1'",
    caption = "20th of December was removed from all years"
  )






func_plot_densities_pgvwsShares_timeline(data=monthly_hshares_device,
                                         characteristic_var="deviceCategory")
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
