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
    filter(!date %in% c(as.Date("2018-12-20"), as.Date("2019-12-20"))) %>%
    mutate(
      date_year = factor(date_year, levels = rev(sort(unique(date_year)))),
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
        (date >= "2020-03-21" & date <= "2020-04-20") ~ "2020 Mar/Apr",
        TRUE ~ NA_character_
      )) %>%
    mutate(date_period = factor(date_period, levels=unique(date_period))) 
  
  # create two new variable breaking down the new monthly period into constituing parts: period_year, period_month
  ordered_period_months <- c("Jan/Feb","Feb/Mar","Mar/Apr","Apr/May","May/Jun","Jun/Jul",
                             "Jul/Aug","Aug/Sep","Sep/Oct","Oct/Nov","Nov/Dec","Dec/Jan")
  ordered_period_years <- c("2017", "2017/18", "2018", "2018/19", "2019", "2019/20", "2020")
  
  data <- data %>%
    mutate(date_period_year = sapply(str_split(data$date_period, " "), function(x) x[1]),
           date_period_month = sapply(str_split(data$date_period, " "), function(x) x[2])) %>%
    mutate(date_period_year = factor(date_period_year, levels = rev(ordered_period_years), ordered = TRUE),
           date_period_month = factor(date_period_month, levels = rev(ordered_period_months), ordered = TRUE))
  
  # order also date_period
  data$date_period <- factor(data$date_period, 
                                levels = unique(data$date_period[order(data$date_period_year, data$date_period_month)]), 
                                ordered = TRUE)
  
  # Create weekly periods within each monthly period
  
  # add week number, by grouping data in 7 day intervals starting on first day of each monthly series
  # ref: https://stackoverflow.com/questions/22559322/how-to-group-data-in-7-day-intervals-starting-on-a-particular-weekday
  # as the last day in each monthly series (the 19th) will be the only one in week 5, we merge week5 into week 4
  # in this way, we have 4 weekly periods in each monthly periods (even if week 4 is made of 8 days rather than 7)
  
  data <- data %>%
    group_by(date_period) %>%
    mutate(date_week_n = 1 + as.numeric(date - min(date)) %/% 7) %>%
    mutate(date_week_n = case_when(
      date_week_n == 5 ~ 4,
      TRUE ~ date_week_n))
  
  data <- data %>%
    mutate(date_period_month_week = interaction(date_period, date_week_n, lex.order = F),
           date_week_n = factor(date_week_n, levels = c(4,3,2,1), ordered = TRUE)) 
  
  data$date_period_month_week <- factor(data$date_period_month_week, 
                                           levels = unique(data$date_period_month_week[order(data$date_period, data$date_week_n)]), 
                                           ordered = TRUE)
  
  return(data)
}
