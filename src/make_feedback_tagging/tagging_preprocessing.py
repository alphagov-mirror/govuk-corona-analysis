from typing import Dict, List, Union
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


def find_duplicated_rows(df: pandas.DataFrame, col_duplicates: List[str]) -> pandas.DataFrame:
    """Find all duplicated rows within certain columns of a pandas DataFrame,

    :param df: A pandas DataFrame with at least the columns in `col_duplicates`. `df` potentially contains duplicate
        rows of data across just these columns `col_duplicates`
    :param col_duplicates: A list of column names in `df` where there could be potentially duplicate rows of data.
    :return: A pandas DataFrame of all the duplicate rows of data in `df`, when finding duplicates just in the
        `col_duplicates` columns.

    """

    # Get all boolean pandas Series for any rows that have duplicates
    bool_duplicate_rows = df[col_duplicates].duplicated(keep=False)

    # Return just the duplicated rows
    return df[bool_duplicate_rows]


def rank_rows(df: pandas.DataFrame, col_key: str, method: str = "first", ascending: bool = False) -> pandas.Series:
    """Rank the rows along a column of a pandas DataFrame.

    :param df: A pandas DataFrame for ranking
    :param col_key: A column for the ranking.
    :param method: Default: "first". A method for ranking - see pandas.Series.rank for options, and further details.
    :param ascending: Default: False. If True, ranks in ascending order.
    :return: A ranked pandas Series according to the chosen method and order.

    """
    return df[col_key].rank(method=method, ascending=ascending)


def rank_tags(df: pandas.DataFrame, col_tag: str, s_ranked: pandas.Series,
              set_tag_ranks: Dict[Union[float, str], int]) -> pandas.Series:
    """Rank tags using a predefined hierarchy, back-filled with calculated ranks.

    The predefined hierarchy of ranks is a dictionary is used to set ranks based on the value of the `col_tag` column
    in `df`. This ensures that desirable/undesirable ranks are forced to a set rank. Any other values not listed in
    the hierarchy are back-filled using already calculated ranks from `s_ranked`.

    The outputted ranking is always a positive, and non-zero integer between 1 and n, where n is the largest possible
    rank.

    :param df: A pandas DataFrame for ranking.
    :param col_tag: A column in `df` containing tags for ranking.
    :param s_ranked: A pandas Series of calculated ranks that ignores the hierarchy.
    :param set_tag_ranks: A dictionary of a predefined hierarchy or ranks for values in `col_tag`, which can be
        different to `s_ranked`. The dictionary keys are values from `col_tag`, and the dictionary values are their
        predefined rankings. Not all values of `col_tag` need to be represented here - those that are missing will be
        replaced by their corresponding value from `s_ranked`. The values should all be less than 0, and in ascending
        order of priority.
    :return: A pandas Series of the ranks of `col_tag` using a combination of predefined ranks from `set_tag_ranks`,
        back-filled with calculated ranks from `s_ranked`.

    """

    # Assert that `s_ranked` must be the same length as `df`
    assert len(s_ranked) == len(df), f"'s_ranked', and 'df' must be the same length!: {len(s_ranked):,} != {len(df):,}"

    # Create a boolean pandas Series of where all the tags in `col_tag` matches the keys of `set_tag_ranks`
    bool_set_tags = df[col_tag].isin(set_tag_ranks.keys())

    # Where `bool_set_tags` is True, replace `s_ranked` with the values from `set_tag_ranks`, otherwise use the
    # values from `s_ranked`
    s_ranked_tag = s_ranked.where(~bool_set_tags, df[col_tag].replace(set_tag_ranks)).astype(int)

    # Adjust the ranks, so that the minimum value is 1
    return s_ranked_tag.add(abs(s_ranked_tag.min()) + 1) if s_ranked_tag.min() < 1 else s_ranked_tag
