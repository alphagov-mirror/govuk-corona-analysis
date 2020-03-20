library(dplyr)
library(stringr)
library(lubridate)

library(ggplot2)
library(ggridges)
library(govstyle)

# load data in
data_bq <- readRDS(file = here::here("data", "bg_pgviews_devicecategory.RDS"))


# Wrangle -----------------------------------------------------------------
# add additional column to specify period
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
  facet_grid(rows = vars(deviceCategory)) +
  geom_vline(xintercept = as.Date(c("2017-12-21", "2018-12-21", "2019-12-21")),
             linetype = "dotted", colour = "black", size = 0.5) +
  labs(title = "Time Plot of GOV.UK Daily Shares of Pageviews by device category", 
       x = "Date", 
       y = "Share of page views",
       colour = guide_legend(title = "Key:")) +
  theme_gov()

# save to 16:9 aspect ratio suitable for full-bleed slides
ggsave(filename = "reports/figures/devicecategory_time_all.jpg", plot = plot_devicecategory, width = 9, height = 5.0625, units = "in")


  # Plot: Density ------------------------------------------------------------
n_categories <- length(x = unique(x = data_bq$deviceCategory))

plot_devicecategory <- data_bq %>%
  #fill shades change according to x-values
  ggplot(mapping = aes(x = prop_pageviews, y = date_period, fill = stat(x))) + 
  geom_density_ridges_gradient(quantile_lines = TRUE, quantiles = 2, scale = 0.9) +
  scale_fill_distiller(palette = "Reds", direction = 1) +
  #scale_y_discrete(expansion(mult = c(0.01, 1))) +
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
ggsave(filename = "reports/figures/devicecategory_density_all.jpg", plot = plot_devicecategory, width = 9, height = 5.0625, units = "in")
