library(dplyr)
library(stringr)

library(ggplot2)
library(ggridges)
library(govstyle)

source(here::here("src/make_data", "func_wrangle.R"))


# Load and Wrangle --------------------------------------------------------
data_bq <- readRDS(file = here::here("data", "bg_pgviews_devicecategory_xdomain.RDS"))

data_bq <- data_bq %>% 
  mutate(date = as.Date(visitStartTimestamp),
         date_year = lubridate::year(x = date),
         hour = lubridate::hour(x = date),
         datetime_hour = case_when(
           stringr::str_length(hour) == 1    ~    paste0(date, ' 0', hour),
           stringr::str_length(hour) == 2    ~    paste0(date, ' ', hour),
           TRUE                              ~    NA_character_),
         datetime_hour = lubridate::ymd_h(datetime_hour)) %>% 
  group_by(date, date_year, datetime_hour, deviceCategory) %>% 
  summarise(pageviews = sum(x = pageviews, na.rm = TRUE)) %>% 
  ungroup()
data_bq <- func_wrangle(data = data_bq)

# compute proportions
data_bq <- data_bq %>%
  group_by(date, date_period, datetime_hour, deviceCategory) %>% 
  # sum of all pageviews for each day-hour for each category
  summarise(total_pageviews = sum(x = as.numeric(pageviews), na.rm = TRUE)) %>%
  # calculate proportion of hourly pageviews by category  
  mutate(prop_pageviews = total_pageviews/(sum(total_pageviews)))


# Plot --------------------------------------------------------------------

  # Plot: Time ---------------------------------------------------------------
data_timeplot <- data_bq %>% 
  group_by(date, deviceCategory) %>% 
  summarise(total_pageviews = sum(x = total_pageviews, na.rm = TRUE)) %>% 
  mutate(prop_pageviews = total_pageviews/sum(x = total_pageviews))

plot_devicecategory <- ggplot(data = data_timeplot, mapping = aes(x = date, y = prop_pageviews, colour = deviceCategory)) +
  geom_line() +
  geom_vline(xintercept = as.Date("2020-03-15")) +
  facet_grid(rows = vars(deviceCategory)) +
  geom_vline(xintercept = as.Date(c("2017-12-21", "2018-12-21", "2019-12-21")),
             linetype = "dotted", colour = "black", size = 0.5) +
  labs(title = "Time Plot of GOV.UK Daily Shares of Pageviews by Device Category", 
       x = "Date", 
       y = "Share of page views",
       colour = guide_legend(title = "Key:")) +
  theme_gov()

# save to 16:9 aspect ratio suitable for full-bleed slides
ggsave(filename = "reports/figures/devicecategory_time_cv.jpg", plot = plot_devicecategory, width = 9, height = 5.0625, units = "in")


  # Plot: Density ------------------------------------------------------------
n_categories <- length(x = unique(x = data_bq$deviceCategory))

plot_devicecategory <- data_bq %>%
  #fill shades change according to x-values
  ggplot(mapping = aes(x = prop_pageviews, y = date_period, fill = stat(x))) + 
  geom_density_ridges_gradient(quantile_lines = TRUE, quantiles = 2, scale = 0.9) +
  scale_fill_distiller(palette = "Reds", direction = 1) +
  scale_y_discrete(expansion(mult = c(0.01, 1))) +
  scale_x_continuous(expand = c(0,0)) +
  coord_cartesian(clip = "off") +
  facet_wrap(reformulate(termlabels = "deviceCategory"), ncol = n_categories) +
  labs(
    x = "Share of hourly page views",
    y = "Year and month",
    title = paste("Density Distribution Plot of GOV.UK Hourly Shares of Pageviews by Device Category"),
    caption = "Monthly periods start on the 21st of month 'k' and end on 20th of month 'k+1' \n20th of December was removed from all years"
  ) +
  theme_gov() +
  theme(axis.title.y = element_blank())

# save to 16:9 aspect ratio suitable for full-bleed slides
ggsave(filename = "reports/figures/devicecategory_density_cv.jpg", plot = plot_devicecategory, width = 9, height = 5.0625, units = "in")
