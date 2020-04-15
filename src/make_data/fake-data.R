# Create fake data for this schema
# https://docs.google.com/spreadsheets/d/1lzO296uS1he0wjXIfNqRbrrsbZ3jhMxuzAgZY0nqcOA

# Additional notes:
# * The NHS vulnerable people fake should include people who have submitted via
# web and IVR. There should also be people on the NHS list who haven't
# submitted, and people in the web form submissions who aren't on the NHS list.
# IVR blocks those who aren't on the NHS list.
# * The live data has addresses in the NHS data and in the web form submission,
# and there are some instances where they match, some where they are different
# due to formatting and some where it's just totally different.
# * Ideally we'd be looking at a few thousand for each.

# Preparation:
# Download  the dataset `epraccur` from
# https://digital.nhs.uk/services/organisation-data-service/data-downloads/gp-and-gp-practice-related-data
# and extract it into a folder called "data/nhs-gp-codes"

# Configuration ----------------------------------------------------------------

# Total number of records to generate across all datasets
total_records <- 1000

# Percentage to appear in NHS list only
nhs_perc <- .1

# Percentage to appear in Web list only
web_perc <- .1

# Percentage to appear in both NHS and Web lists
nhs_and_web_perc <- 1 - nhs_perc - web_perc

# Percentage to appear in both NHS and IVR lists, drawn from the whole NHS list,
# so including some that are/aren't in the Web list.
ivr_perc <- .1

# End of config ----------------------------------------------------------------

library(tidyverse)
library(charlatan)
library(wakefield)
library(ids)
library(PostcodesioR)
library(generator)
library(here)

# Master records ---------------------------------------------------------------

n <- total_records
set.seed(2019-04-06)

# random_postcodes <- function(n) {
#   purrr::map_chr(seq_len(n), ~ random_postcode()$postcode)
# }
# random_postcodes(2)

# Generate some random addresses and places
addresses <- map(seq_len(n), ~ AddressProvider$new("en_GB"))
places <- map_dfr(seq_len(n), ~ random_place())

#' Create random reference IDs
reference_ids <- function(n) {
  paste(
    str_replace_all(date_stamp(n, random = TRUE, k = 31, by = "-1 days"), "-", ""),
    ch_integer(n = n, min = 100000, max = 999999),
    toupper(random_id(n = n, bytes = 3)),
    sep = "-"
  )
}

#' NHS practice codes
nhs_gp_practice_codes <-
  here("data/nhs-gp-codes/epraccur.csv") %>%
  read_csv(col_names = FALSE) %>%
  select(c(1, 2, 18)) %>%
  set_names(c("code", "name", "telephone")) %>%
  sample_n(n)

#' Dates of birth
dobs <- dob(n)

#' NHS numbers
nhs_numbers <- ch_integer(n = n, min = 1000000000, max = 9999999999)

#' Phone numbers
phone_number_calls = ch_phone_number(n, locale = "en_GB")
phone_number_texts = ch_phone_number(n, locale = "en_GB")

#' Names
first_names = ch_name(n)
other_names = ch_name(n)
last_names = ch_name(n)

# Master records ---------------------------------------------------------------
master <-
  tibble(
    #
    # NHS columns
    #
    nhs_nhs_number = nhs_numbers,
    nhs_dob = dobs,
    nhs_dob_year = lubridate::year(dobs),
    nhs_dob_month = lubridate::month(dobs),
    nhs_dob_day = lubridate::day(dobs),
    nhs_patient_title = r_sample(n, c("Mr", "Ms", "Dr", "Prof")),
    nhs_patients_first_name = first_names,
    nhs_patients_other_name = other_names,
    nhs_patients_surname = last_names,
    nhs_patients_address_line1 = str_replace_all(map_chr(addresses, ~ .x$street_address()), "\\n", ""),
    nhs_patients_address_line2 = map_chr(addresses, ~ .x$city()),
    nhs_patients_address_line3 = map_chr(addresses, ~ .x$street_name()),
    nhs_patients_address_line4 = map_chr(places$name_1, ~ ifelse(is.null(.x), NA_character_, .x)),
    nhs_patients_address_line5 = map_chr(places$county_unitary, ~ ifelse(is.null(.x), NA_character_, .x)),
    nhs_postcode = map_chr(addresses, ~ .x$postcode()),
    nhs_practice_code = nhs_gp_practice_codes$code,
    nhs_practice_name = nhs_gp_practice_codes$name,
    nhs_contact_telephone = nhs_gp_practice_codes$telephone,
    #
    # Web columns
    #
    live_in_england = r_sample(n, c("yes", "no")), # Should this affect addresses?
    first_name = first_names,
    middle_name = other_names,
    last_name = last_names,
    city = map_chr(places$name_1, ~ ifelse(is.null(.x), NA_character_, .x)),
    # Use the same addresses as created above
    address_l1 = str_replace_all(map_chr(addresses, ~ .x$street_address()), "\\n", ""),
    address_l2 = map_chr(addresses, ~ .x$street_name()),
    county = map_chr(places$county_unitary, ~ ifelse(is.null(.x), NA_character_, .x)),
    postcode = map_chr(addresses, ~ .x$postcode()),
    nhs_number = nhs_numbers,
    carry_supplies = r_sample(n, c("yes", "no")),
    reference_id = reference_ids(n),
    dob_day = lubridate::day(dobs),
    dob_month = lubridate::month(dobs),
    dob_year = lubridate::year(dobs),
    full_dob = dobs,
    session_id = uuid(n, drop_hyphens = TRUE),
    csrf_token = uuid(n, drop_hyphens = TRUE),
    phone_number_calls = phone_number_calls,
    phone_number_texts = phone_number_texts,
    contact = r_email_addresses(n),
    know_nhs_number = r_sample(n, c("YES, I KNOW MY NHS NUMBER", "NO, I DO NOT KNOW MY NHS NUMBER")),
    check_answers_seen = r_sample_logical(n),
    nhs_letter = r_sample(n, c("yes", "no")),
    basic_care_needs = r_sample(n, c("yes", "no")),
    dietary_requirements = r_sample(n, c("yes", "no")),
    medical_conditions = r_sample(n, c("YES, I HAVE A MEDICAL CONDITION THAT MAKES ME EXTREMELY VULNERABLE TO CORONAVIRUS", "YES, I HAVE ONE OF THE MEDICAL CONDITIONS ON THE LIST")),
    essential_supplies = r_sample(n, c("yes", "no")),
    updated_at = ch_date_time(n) %>% reduce(c),
    referenceid = reference_id, # A duplicate column that exists in prod
    unixtimestamp = ch_date_time(n) %>% reduce(c),
    created_at = ch_date_time(n) %>% reduce(c),
    #
    # IVR columns
    #
    ivr_nhs_number = nhs_numbers,
    ivr_postcode = map_chr(addresses, ~ .x$postcode()),
    ivr_dob = dob(n),
    ivr_customer_callling_number = ch_phone_number(n, locale = "en_GB"),
    ivr_current_item_id = ch_integer(n, min = 1, max = 99),
    ivr_transfer = r_sample_logical(n),
    ivr_fallback_time = ch_integer(n = n, min = 0, max = 9),
    ivr_nhs_known = r_sample(n, c("yes", "no")),
    ivr_contact_id = uuid(n),
    ivr_preferred_phone_number = r_sample(n, c(phone_number_calls, phone_number_texts)),
    ivr_phone_number_calls = phone_number_calls,
    ivr_postal_code_verified = r_sample_logical(n),
    ivr_delivery_supplies = r_sample_logical(n),
    ivr_carry_supplies = r_sample_logical(n),
    ivr_have_help = r_sample_logical(n),
    ivr_call_timestamp = ch_date_time(n) %>% reduce(c),
    ivr_umet_needs = rep(NA_character_, n)
    ) %>%
  mutate_all(as.character)

glimpse(master)

write_delim(master, here("data/fake-data/master.csv"), delim = "|")

# Column names -----------------------------------------------------------------

nhs_column_names <-
  c("nhs_nhs_number" ,
    "nhs_dob" ,
    "nhs_dob_year" ,
    "nhs_dob_month" ,
    "nhs_dob_day" ,
    "nhs_patient_title" ,
    "nhs_patients_first_name" ,
    "nhs_patients_other_name" ,
    "nhs_patients_surname" ,
    "nhs_patients_address_line1" ,
    "nhs_patients_address_line2" ,
    "nhs_patients_address_line3" ,
    "nhs_patients_address_line4" ,
    "nhs_patients_address_line5" ,
    "nhs_postcode" ,
    "nhs_practice_code" ,
    "nhs_practice_name" ,
    "nhs_contact_telephone")

web_column_names <-
  c("live_in_england",
    "first_name",
    "middle_name",
    "last_name",
    "city",
    "address_l1",
    "address_l2",
    "county",
    "postcode",
    "nhs_number",
    "carry_supplies",
    "reference_id",
    "dob_day",
    "dob_month",
    "dob_year",
    "full_dob",
    "session_id",
    "csrf_token",
    "phone_number_calls",
    "phone_number_texts",
    "contact",
    "know_nhs_number",
    "check_answers_seen",
    "nhs_letter",
    "basic_care_needs",
    "dietary_requirements",
    "medical_conditions",
    "essential_supplies",
    "updated_at",
    "referenceid",
    "unixtimestamp",
    "created_at")

ivr_column_names <-
  c("ivr_nhs_number",
    "ivr_postcode",
    "ivr_dob",
    "ivr_customer_callling_number",
    "ivr_current_item_id",
    "ivr_transfer",
    "ivr_fallback_time",
    "ivr_nhs_known",
    "ivr_contact_id",
    "ivr_preferred_phone_number",
    "ivr_phone_number_calls",
    "ivr_postal_code_verified",
    "ivr_delivery_supplies",
    "ivr_carry_supplies",
    "ivr_have_help",
    "ivr_call_timestamp",
    "ivr_umet_needs")

# Individual lists -------------------------------------------------------------

nhs_list <- sample_frac(master, nhs_perc + nhs_and_web_perc)
web_list <- sample_frac(master, web_perc + nhs_and_web_perc)
ivr_list <- sample_frac(nhs_list, ivr_perc)

nhs_list <- select_at(nhs_list, nhs_column_names)
web_list <- select_at(web_list, web_column_names)
ivr_list <- select_at(ivr_list, ivr_column_names)

write_delim(nhs_list, here("data/fake-data/nhs.csv"), delim = "|")
write_delim(web_list, here("data/fake-data/web.csv"), delim = "|")
write_delim(ivr_list, here("data/fake-data/ivr.csv"), delim = "|")

# Check overlaps between lists -------------------------------------------------

# NHS only
nhs_list %>%
  anti_join(web_list, by = c("nhs_nhs_number" = "nhs_number")) %>%
  anti_join(ivr_list, by = c("nhs_nhs_number" = "ivr_nhs_number")) %>%
  nrow()

# NHS and Web
nhs_list %>%
  inner_join(web_list, by = c("nhs_nhs_number" = "nhs_number")) %>%
  nrow()

# Web only
web_list %>%
  anti_join(nhs_list, by = c("nhs_number" = "nhs_nhs_number")) %>%
  nrow()

# IVR and Web
ivr_list %>%
  inner_join(web_list, by = c("ivr_nhs_number" = "nhs_number")) %>%
  nrow()

# IVR not Web
ivr_list %>%
  anti_join(web_list, by = c("ivr_nhs_number" = "nhs_number")) %>%
  nrow()

# Web not IVR
web_list %>%
  anti_join(ivr_list, by = c("nhs_number" = "ivr_nhs_number")) %>%
  nrow()
