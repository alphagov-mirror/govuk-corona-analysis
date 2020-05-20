import pandas
import re


def standardise_columns(df: pandas.DataFrame) -> pandas.DataFrame:
    """Replace punctuation from pandas DataFrame columns, and set to lowercase.

    All punctuation in the column names are replaced with an underscore; adjacent punctuation are all replaced with a
    single underscore.

    :param df: A pandas DataFrame, potentially with column names containing punctuation and/or uppercase characters.
    :return: A pandas DataFrame with all punctuation replaced with an underscore, with multiple adjacent instances of
        punctuation replaced with a single underscore, and all column headers in lowercase.

    """
    return df.rename(columns=lambda n: re.sub(r"\W+", "_", n.lower()))


def convert_object_to_datetime(df: pandas.DataFrame, col_datetime: str = "text_date") -> pandas.DataFrame:
    """Convert a pandas Series of datetime strings to a pandas datetime Series.

    The datetime strings are assumed to start with a datetime in the format "YYYY-mm-dd HH:MM:SS" followed by some
    text. This function will remove the extra text outside of the datetime, and then parse the result as a datetime
    object.

    :param df: A pandas DataFrame containing at least `col_datetime`.
    :param col_datetime: Default: "text_date". A column name containing strings, where each string begins with a
        datetime string of the format "YYYY-mm-dd HH:MM:SS", possibly followed by some additional text.
    :return: A near-identical copy of `df`, except `col_datetime` will be a datetime object with correctly parsed
        datetime values.

    """

    # Define a regular expression pattern that extracts text of "YYYY-mm-dd HH:MM:SS" at the beginning of a string, and
    # labels the group as `datetime`. This text must occur at the beginning of the string
    regex_pattern = r"^(?P<datetime>\d{4}\-\d{2}\-\d{2} \d{2}\:\d{2}\:\d{2}).*$"

    # Extract the datetime strings from the `col_datetime` column of `df`
    s = df[col_datetime].str.replace(regex_pattern, lambda m: m.group('datetime'))

    # Convert `s` into a datetime pandas Series, and overwrite `col_datetime`
    df_out = df.assign(**{col_datetime: pandas.to_datetime(s, format="%Y-%m-%d %H:%M:%S")})

    # Assert that there are no NULL time values in the new `col_datetime` column of `df_out`
    assert df_out[col_datetime].notnull().all(), f"There are some missing datetime values in {col_datetime!r} of df!"

    # Return the revised pandas DataFrame
    return df_out
