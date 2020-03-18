# Data
## Characteristics of users during the corona period
The purpose of this is to describe how the functions and queries work for the "Characteristics of users during the corona period" workstrand. This workstrand exists on the Corona Trello board [here](https://trello.com/c/rIIksM1H).

The following functions and queries help set-up the third-stage of this work, namely:

> Tell a data story of how the proportion of sessions via mobile has changed for corona period

Below, we will describe the functions and query that will enable this analysis:

1. `query_characteristic_device.sql` - this script queries from BigQuery(BQ) session-level data by device category. It will provide the data necessary for doing this stage of the analysis.
1. `func_readsql.R` - this script parses the query in `query_characteristic_device.sql` and re-formats it so that it can be read into R for use in the `func_importdatabq()` function
1. `func_importdatabq.R` - this script connects the R session to BQ, connects to the specified project and dataset, and then runs the query from the `func_readsql()` function to pull the data into R.
    + Note, it calls func_readsql(), which is defined in `func_readsql.R`
1. `func_readBQsaveRDS.R` - this script takes the data imported from BQ as executed by `func_importdatabq.R` and saves this as an `.RDS` file.
    + The reason why we save it as an `.RDS` file is because we do not want to unecessarily pull data from BQ each time we run the analysis. In this way, we avoid increasing querying costs.