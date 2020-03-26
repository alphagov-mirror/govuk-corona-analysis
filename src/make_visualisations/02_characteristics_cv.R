library(dplyr)
library(stringr)
library(lubridate)

library(ggplot2)
library(ggridges)
library(khroma)

source(here::here("src/make_data", "func_wrangle.R"))

# create custom plotting theme
theme_custom <- theme(plot.title = element_text(face = "bold", hjust = 0.5),
                      panel.background = element_blank(),
                      axis.line = element_line(colour = "black"),
                      legend.position = "bottom",
                      legend.direction = "horizontal")

# Load and Wrangle --------------------------------------------------------
data_bq <- readRDS(file = here::here("data", "bg_pgviews_devicecategory_xdomain.RDS"))

data_bq <- data_bq %>% 
  mutate(date = as.Date(visitStartTimestamp),
         # create columns so we can pass it through `func_wrangle()` function
         date_year = year(x = date),
         date_month_nm = month(x = date, label = TRUE, abbr = TRUE),
         hour = hour(x = date),
         datetime_hour = case_when(
           str_length(hour) == 1    ~    paste0(date, ' 0', hour),
           str_length(hour) == 2    ~    paste0(date, ' ', hour),
           TRUE                     ~    NA_character_),
         datetime_hour = ymd_h(datetime_hour)) %>% 
  group_by(date, date_year, date_month_nm, datetime_hour, deviceCategory) %>% 
  summarise(pageviews = sum(x = pageviews)) %>% 
  ungroup()
data_bq <- func_wrangle(data = data_bq)

# compute proportions
data_bq <- data_bq %>%
  group_by(date, date_period, datetime_hour, deviceCategory) %>% 
  # sum of all pageviews for each day-hour for each category
  summarise(total_pageviews = sum(x = pageviews)) %>%
  # calculate proportion of hourly pageviews by category  
  mutate(prop_pageviews = total_pageviews/(sum(total_pageviews)))


# Plot --------------------------------------------------------------------

  # Plot: Time ---------------------------------------------------------------
data_timeplot <- data_bq %>% 
  group_by(date, deviceCategory) %>% 
  summarise(total_pageviews = sum(x = total_pageviews, na.rm = TRUE)) %>% 
  mutate(prop_pageviews = total_pageviews/sum(x = total_pageviews))

# counts
plot_devicecategory <- ggplot(data = data_timeplot, mapping = aes(x = date, y = total_pageviews, colour = deviceCategory)) +
  geom_point() +
  geom_line() +
  geom_vline(xintercept = as.Date("2020-03-15"),
             linetype = "dotted", colour = "black", size = 0.5) +
  labs(title = "Time Plot of GOV.UK Daily Pageviews by Device Category", 
       x = "Date", 
       y = "Total page views",
       colour = guide_legend(title = "Key:")) +
  scale_colour_bright() +
  theme_custom
# save to 16:9 aspect ratio suitable for full-bleed slides
ggsave(filename = "reports/figures/devicecategory_time_cvcounts.jpg", plot = plot_devicecategory, width = 9, height = 5.0625, units = "in")

# proportions
plot_devicecategory <- ggplot(data = data_timeplot, mapping = aes(x = date, y = prop_pageviews, colour = deviceCategory)) +
  geom_line() +
  geom_vline(xintercept = as.Date(c("2020-03-15")),
             linetype = "dotted", colour = "black", size = 0.5) +
  labs(title = "Time Plot of GOV.UK Daily Shares of Pageviews by Device Category", 
       x = "Date", 
       y = "Share of page views",
       colour = guide_legend(title = "Key:")) +
  scale_colour_bright() +
  theme_custom

# save to 16:9 aspect ratio suitable for full-bleed slides
ggsave(filename = "reports/figures/devicecategory_time_cv.jpg", plot = plot_devicecategory, width = 9, height = 5.0625, units = "in")


  # Plot: Density ------------------------------------------------------------
n_categories <- length(x = unique(x = data_bq$deviceCategory))

plot_devicecategory <- data_bq %>%
  #fill shades change according to x-values
  ggplot(mapping = aes(x = prop_pageviews, y = date_period, fill = stat(x))) + 
  geom_density_ridges_gradient(quantile_lines = TRUE, quantiles = 2, scale = 0.9) +
  scale_fill_YlOrBr() +
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
  theme_custom +
  theme(axis.title.y = element_blank())

# save to 16:9 aspect ratio suitable for full-bleed slides
ggsave(filename = "reports/figures/devicecategory_density_cv.jpg", plot = plot_devicecategory, width = 9, height = 5.0625, units = "in")
