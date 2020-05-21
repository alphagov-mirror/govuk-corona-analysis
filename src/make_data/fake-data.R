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

# Download the ONS Postcode Director from
# http://geoportal.statistics.gov.uk/datasets/ons-postcode-directory-may-2019
# and extract the file ONSPD_MAY_2019_UK.txt (or whatever the date is) into a
# folder called "data/postcodes".

# Configuration ----------------------------------------------------------------

# Total number of records to generate across all datasets
total_records <- 5000

# Total flags to have as columns in NHS list only
total_flags <- 8

# Percentage to appear in NHS list only
nhs_perc <- .2

# Percentage to appear in the NHS deductions list
nhs_deductions_perc <- .05

# Percentage to appear in Web list only
web_perc <- .2

# Percentage to appear in both NHS and Web lists
nhs_and_web_perc <- 1 - nhs_perc - web_perc

# Percentage to appear in both NHS and IVR lists, drawn from the whole NHS list,
# so including some that are/aren't in the Web list.
ivr_perc <- .7

# End of config ----------------------------------------------------------------

library(tidyverse)
library(charlatan)
library(wakefield)
library(ids)
library(PostcodesioR)
library(generator)
library(vroom) # to read postcodes file without importing it into memory
library(here)

# Master records ---------------------------------------------------------------

n <- total_records
m <- total_flags
set.seed(2019-04-06)

# Generate some random addresses and places
addresses <- AddressProvider$new()
places <- map_dfr(seq_len(n), ~ random_place())

## sample real UK postcodes from the ONS
postcodes_path <- here("data/postcodes/Data/ONSPD_FEB_2020_UK.csv")
postcodes <-
  vroom(postcodes_path,
        col_types = cols(.default = col_skip(),
                         pcd = col_character()))$pcd %>%
  sample(n)

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
  select(-FID) %>%
  rename(ward = CCG20CD,
         ccgcode = CCG20CDH,
         ccgname = CCG20NM)

#' Dates of birth
dobs <- dob(n)
dobs <- str_remove_all(string = dobs, pattern = "-")

#' NHS numbers
nhs_numbers <- ch_integer(n = n, min = 1000000000, max = 9999999999)
nhs_numbers_web <- c(nhs_numbers[1:(n*(m-1)/m)], ch_integer(n = n/m, min = 1000000000, max = 9999999999))

#' Phone numbers
phone_number_calls = ch_phone_number(n, locale = "en_GB")
phone_number_texts = ch_phone_number(n, locale = "en_GB")

#' Names
## generate unique vector of prefixes and suffixes
x <- PersonProvider$new()
vec_prefix <- vector(mode = "character", length = n)
vec_suffix <- vector(mode = "character", length = n)
for (i in 1:n) {
  vec_prefix[i] <- x$prefix()
  vec_suffix[i] <- x$suffix()
}
## take unique entries
vec_prefix <- unique(x = vec_prefix)
vec_suffix <- unique(x = vec_suffix)
## collapse vector of elements into one string separate with regex "|"
vec_prefix <- paste0(vec_prefix, " ", collapse = "|")
vec_suffix <- paste0(" ", vec_suffix, collapse = "|")

## generate fake names
names <- ch_name(n)

## remove prefixes and suffixes
names <- str_replace(string = names, pattern = regex(vec_prefix, ignore_case = FALSE), replacement = "")
names <- str_replace(string = names, pattern = regex(vec_suffix, ignore_case = FALSE), replacement = "")

first_names = str_split(string = names, pattern = " ", simplify = TRUE)[,1]
other_names = str_split(string = ch_name(n), pattern = " ", simplify = TRUE)[,2]
last_names = str_split(string = names, pattern = " ", simplify = TRUE)[,2]

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
         flag_pdsinformallydeceased = V8
  ) %>%
  # Resample flag_pdsinformallydeceased to have more 0s than 1s
  mutate(flag_pdsinformallydeceased = sample(x = 0:1,
                                             size = n,
                                             replace = TRUE,
                                             prob = c(0.75, 0.25)))

# Master records ---------------------------------------------------------------
master <-
  tibble(
    #
    # NHS columns
    #
    Traced_NHSNUMBER = nhs_numbers,
    DateOfBirth = dobs,
    PatientFirstName = first_names,
    PatientOtherName = other_names,
    PatientSurname = last_names,
    PatientAddress_Line1 = str_replace_all(map_chr(seq_len(n), ~ addresses$street_address()), "\\n", ""),
    PatientAddress_Line2 = map_chr(seq_len(n), ~ addresses$city()),
    PatientAddress_Line3 = map_chr(seq_len(n), ~ addresses$street_name()),
    PatientAddress_Line4 = map_chr(places$name_1, ~ ifelse(is.null(.x), NA_character_, .x)),
    PatientAddress_Line5 = map_chr(places$county_unitary, ~ ifelse(is.null(.x), NA_character_, .x)),
    PatientAddress_PostCode = postcodes,
    GPPractice_Code = nhs_gp_practice_codes$code,
    Practice_NAME = nhs_gp_practice_codes$name,
    contact_telephone = nhs_gp_practice_codes$telephone,
    mobile = ch_phone_number(n, locale = "en_GB"),
    landline = ch_phone_number(n, locale = "en_GB"),
    oslaua = map_chr(places$county_unitary, ~ ifelse(is.null(.x), NA_character_, .x)),
    ccg = pull(sample_n(tbl = ccg_codes[, "ccgcode"], size = n, replace = TRUE)),

    `Flag_Chemo/Radiotherapy` = flags$flag_chemo_radiotherapy,
    Flag_Respiratory = flags$flag_respiratory,
    Flag_HeamatologicalCancers = flags$flag_haemotologicalcancers,
    Flag_PregnantWithCongentialHeartDefect = flags$flag_pregnantwithcongentialheartdefect,
    Flag_Transplant = flags$flag_transplant,
    Flag_RareDiseases = flags$flag_rarediseases,

    Gender = pull(sample_n(tbl = tibble(gender = c(0, 1, 2, 9, NA)), size = n, replace = TRUE)),

    Flag_PDSInformallyDeceased = flags$flag_pdsinformallydeceased,

    oscty = map_chr(places$name_1, ~ ifelse(is.null(.x), NA_character_, .x)),
    Data_Source = sample(c("COG_TRUST_UPDATE", "GP WEEKLY SPL", "HES", "Initial"),
                         size = n,
                         replace = TRUE,
                         prob = c(340814, 637709, 2197, 1220468)), # Counts from real data
    InceptionDate = sample(seq.Date(from = as.Date("2020-04-12"),
                                    to = Sys.Date(),
                                    by = "1 day"),
                           replace = TRUE,
                           size = n),
    #
    # Web columns
    #
    live_in_england = r_sample(n, c("yes", "no")), # Should this affect addresses?
    first_name = first_names,
    middle_name = other_names,
    last_name = last_names,
    city = map_chr(places$name_1, ~ ifelse(is.null(.x), NA_character_, .x)),
    # Use the same addresses as created above
    address_l1 = str_replace_all(map_chr(seq_len(n), ~ addresses$street_address()), "\\n", ""),
    address_l2 = map_chr(seq_len(n), ~ addresses$street_name()),
    county = map_chr(places$county_unitary, ~ ifelse(is.null(.x), NA_character_, .x)),
    postcode = postcodes,
    nhs_number = nhs_numbers_web,
    carry_supplies = r_sample(n, c("yes", "no")),
    reference_id = reference_ids(n),
    full_dob = dob(n = n),
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
    ivr_postcode = postcodes,
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

write.csv(x = master, file = here("data/fake-data/master.csv"), quote = TRUE, row.names = FALSE)

# Individual lists -------------------------------------------------------------

nhs_list <- sample_frac(master, nhs_perc + nhs_and_web_perc)
nhs_deductions_list <- sample_frac(master, nhs_deductions_perc)
web_list <- sample_frac(master, web_perc + nhs_and_web_perc)
ivr_list <- sample_frac(nhs_list, ivr_perc)

nhs_list <- select_at(nhs_list, vars(Traced_NHSNUMBER:InceptionDate))
nhs_deductions_list <- select_at(nhs_deductions_list, vars(Traced_NHSNUMBER:InceptionDate))
web_list <- select_at(web_list, vars(live_in_england:created_at))
ivr_list <- select_at(ivr_list, vars(ivr_nhs_number:ivr_umet_needs))

# More requirements of different types of duplicates and overlaps
#
# Web Data : Duplicate entries with same nhs number matching BOTH nhs dataset and ivr dataset.
# Web Data : Duplicate entries with same nhs number matching ONLY nhs dataset
# Web Data : Duplicate entries with no nhs number but fuzzy logic matching nhs data set (matching on first name, last name, dob, post code)
# Web Data : Containing nhs number not in nhs set
# IVR Data : Entries with nhs number in nhs data set where ivr_current_item_id = 17
# IVR Data : Multiple entries with same nhs number in nhs data set where ivr_current_item_id = 17 and ivr_current_item_id != 17
# IVR Data : Entries where data is only in nhs data set (and not available in Web data set)

# Create web duplicates matching ivr and/or nhs
web_nhs_ivr_duplicates <-
  web_list %>%
  semi_join(nhs_list, by = c("nhs_number" = "Traced_NHSNUMBER")) %>%
  semi_join(ivr_list, by = c("nhs_number" = "ivr_nhs_number")) %>%
  sample_n(100)
web_nhs_duplicates <-
  web_list %>%
  semi_join(nhs_list, by = c("nhs_number" = "Traced_NHSNUMBER")) %>%
  anti_join(ivr_list, by = c("nhs_number" = "ivr_nhs_number")) %>%
  sample_n(100)

# Create web duplicates but without NHS number
web_duplicates_no_nhs_number <-
  web_list %>%
  semi_join(nhs_list, by = c("nhs_number" = "Traced_NHSNUMBER")) %>%
  sample_n(100) %>%
  mutate(nhs_number = NA_character_)

# Ensure more IVR rows have ivr_current_item_id == 17
ivr_list <-
  ivr_list %>%
  mutate(ivr_current_item_id  = if_else(runif(n()) < .05,
                                        "17",
                                        ivr_current_item_id))

# Duplicate IVR and ensure some have ivr_current_item_id == 17 and some don't.
ivr_duplicates <-
  ivr_list %>%
  sample_n(100) %>%
  mutate(ivr_current_item_id  = if_else(runif(n()) < .5,
                                        "17",
                                        ivr_current_item_id))

drop_ivr_web <-
  web_list %>%
  sample_n(100) %>%
  anti_join(ivr_list, by = c("nhs_number" = "ivr_nhs_number"))

## 3. rowbind to original lists
nhs_list <-
  nhs_list %>%
  anti_join(nhs_deductions_list, by = "Traced_NHSNUMBER")

web_list <-
  bind_rows(
    web_list,
    web_nhs_ivr_duplicates,
    web_nhs_duplicates,
    web_duplicates_no_nhs_number
  ) %>%
anti_join(drop_ivr_web, by = "nhs_number")

ivr_list <-
  bind_rows(
    ivr_list,
    ivr_duplicates
  )

write.csv(x = nhs_list, file = here("data/fake-data/nhs.csv"), quote = TRUE, row.names = FALSE)
write.csv(x = nhs_deductions_list, file = here("data/fake-data/nhs-deductions.csv"), quote = TRUE, row.names = FALSE)
write.csv(x = web_list, file = here("data/fake-data/web.csv"), quote = TRUE, row.names = FALSE)
write.csv(x = ivr_list, file = here("data/fake-data/ivr.csv"), quote = TRUE, row.names = FALSE)
