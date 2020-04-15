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
# Download the dataset `epraccur` from
# https://digital.nhs.uk/services/organisation-data-service/data-downloads/gp-and-gp-practice-related-data
# and extract it into a folder called "data/nhs-gp-codes"

# Download the dataset `Clinical_Commissioning_Groups_April_2020_Names_and_Codes_in_England.csv` from
# https://geoportal.statistics.gov.uk/datasets/clinical-commissioning-groups-april-2020-names-and-codes-in-england
# and extract it into a folder called "data/ccg-codes"

# Configuration ----------------------------------------------------------------

# Total number of records to generate across all datasets
total_records <- 1000

# Total flags to have as columns in NHS list only
total_flags <- 8

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
m <- total_flags
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

#' CCG codes
ccg_codes <-
  here("data/ccg-codes/Clinical_Commissioning_Groups_April_2020_Names_and_Codes_in_England.csv") %>%
  read_csv() %>% 
  select(1:3) %>% 
  rename(ward = CCG20CD,
         ccgcode = CCG20CDH,
         ccgname = CCG20NM)

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

#' Flags
flags <- matrix(data = sample(x = 0:1, size = m * n, replace = TRUE), 
                nrow = m, ncol = n) %>% 
  t() %>% 
  as.data.frame() %>% 
  rename(flag_chemo_radiotherapy = V1,
         flag_respiratory = V2,
         flag_haemotologicalcancers = V3,
         flag_pregnantwithcongentialheartdefect = V4,
         flag_transplant = V5,
         flag_rarediseases = V6,
         flag_pdssensitive = V7,
         flag_pdsinformallydeceased = V8)

# Master records ---------------------------------------------------------------
master <-
  tibble(
    #
    # NHS columns
    #
    nhsnumber = nhs_numbers,
    dateofbirth = dobs,
    patientfirstname = first_names,
    patientothername = other_names,
    patientsurname = last_names,
    patientaddress_line1 = str_replace_all(map_chr(addresses, ~ .x$street_address()), "\\n", ""),
    patientaddress_line2 = map_chr(addresses, ~ .x$city()),
    patientaddress_line3 = map_chr(addresses, ~ .x$street_name()),
    patientaddress_line4 = map_chr(places$name_1, ~ ifelse(is.null(.x), NA_character_, .x)),
    patientaddress_line5 = map_chr(places$county_unitary, ~ ifelse(is.null(.x), NA_character_, .x)),
    patientaddress_postcode = map_chr(addresses, ~ .x$postcode()),
    gppractice_code = nhs_gp_practice_codes$code,
    practice_name = nhs_gp_practice_codes$name,
    contact_telephone = nhs_gp_practice_codes$telephone,
    mobile = ch_phone_number(n, locale = "en_GB"),
    patient_landline = ch_phone_number(n, locale = "en_GB"),
    oslaua = map_chr(places$county_unitary, ~ ifelse(is.null(.x), NA_character_, .x)),
    ccg = pull(sample_n(tbl = ccg_codes[, "ccgcode"], size = n, replace = TRUE)),
    
    flag_chemo_radiotherapy = flags$flag_chemo_radiotherapy,
    flag_respiratory = flags$flag_respiratory,
    flag_haematologicalcancers = flags$flag_haemotologicalcancers,
    flag_pregnantwithcongentialheartdefect = flags$flag_pregnantwithcongentialheartdefect,
    flag_transplant = flags$flag_transplant,
    flag_rarediseases = flags$flag_rarediseases,
    
    gender = pull(sample_n(tbl = tibble(gender = c(0, 1, 2, 9, NA)), size = n, replace = TRUE)),
    
    flag_pdssensitive = flags$flag_pdssensitive,
    flag_pdsinformallydeceased = flags$flag_pdsinformallydeceased,
    
    oscty = map_chr(places$name_1, ~ ifelse(is.null(.x), NA_character_, .x)),
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
  c("nhsnumber",
    "dateofbirth",
    "patientfirstname",
    "patientothername",
    "patientsurname",
    "patientaddress_line1",
    "patientaddress_line2",
    "patientaddress_line3",
    "patientaddress_line4",
    "patientaddress_line5",
    "patientaddress_postcode",
    "gppractice_code",
    "practice_name",
    "contact_telephone",
    "mobile",
    "landline",
    "oslaua",
    "ccg",
    "flag_chemo_radiotherapy",
    "flag_respiratory",
    "flag_haematologicalcancers",
    "flag_pregnantwithcongentialheartdefect",
    "flag_transplant",
    "flag_rarediseases",
    "gender",
    "flag_pdssensitive",
    "flag_pdsinformallydeceased",
    "oscty")

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
  anti_join(web_list, by = c("nhsnumber" = "nhs_number")) %>%
  anti_join(ivr_list, by = c("nhsnumber" = "ivr_nhs_number")) %>%
  nrow()

# NHS and Web
nhs_list %>%
  inner_join(web_list, by = c("nhsnumber" = "nhs_number")) %>%
  nrow()

# Web only
web_list %>%
  anti_join(nhs_list, by = c("nhs_number" = "nhsnumber")) %>%
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
