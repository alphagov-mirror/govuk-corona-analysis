from functools import reduce
from nltk.corpus import stopwords
from src.make_feedback_tool_data.preprocess import PreProcess
from src.utils.parallelise_pandas import parallelise_pandas
from tqdm import tqdm
from typing import Dict, List, Optional, Union
import numpy as np
import pandas
import re

tqdm.pandas()

# Define a default set of tag columns
COLS_TAGS = ["this_response_relates_to_", "coronavirus_theme"]

# Define an order of tags, where a lower value is less important
ORDER_TAGS = {np.nan: -5, "duplicate": -4, "INTERNAL": -3, "none": -2, "ok": -1}

# Get a list of stopwords
STOPWORDS = list(stopwords.words('english'))


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


def rank_multiple_tags(df: pandas.DataFrame, col_tags: List[str], s_ranked: pandas.Series,
                       set_tag_ranks: Dict[Union[float, str], int]) -> List[pandas.Series]:
    """Rank multiple tag columns using a predefined hierarchy, back-filled with calculated ranks.

    The predefined hierarchy of ranks is a dictionary is used to set ranks based on the value of the `col_tags` columns
    in `df`. This ensures that desirable/undesirable ranks are forced to a set rank. Any other values not listed in
    the hierarchy are back-filled using already calculated ranks from `s_ranked`.

    The outputted ranking is always a positive, and non-zero integer between 1 and n, where n is the largest possible
    rank.

    :param df: A pandas DataFrame for ranking.
    :param col_tags: A list of column names from `df` that contain tags.
    :param s_ranked: A pandas Series of calculated ranks that ignores the hierarchy.
    :param set_tag_ranks: A dictionary of a predefined hierarchy or ranks for values in columns of `col_tags`,
        which can be different to `s_ranked`. The dictionary keys are values from `col_tags`, and the dictionary values
        are their predefined rankings. Not all values of `col_tags` need to be represented here - those that are
        missing will be replaced by their corresponding value from `s_ranked`.
    :return: A list of pandas Series, where each pandas Series corresponds to one column from `col_tags`. Each pandas
        Series will have the ranks of that column in `col_tags` using a combination of predefined ranks from
        `set_tag_ranks`, back-filled with calculated ranks from `s_ranked`.

    """

    # Rank all the tags, either by `s_ranked` or, if `col_tag` is a key in `set_tag_ranks`, the value of `set_tag_ranks`
    return [rank_tags(df, col_tag, s_ranked, set_tag_ranks) for col_tag in col_tags]


def get_rank_statistic(s_ranked_tags: List[pandas.Series]) -> pandas.Series:
    """Calculate the overall rank using the rank statistic for a list of ranks.

    Uses the rank statistic, i.e. the geometric mean of the ranks, to calculate an overall rank.

    :param s_ranked_tags: A list of pandas Series that each contain ranks.
    :return: A pandas Series with the overall rank based on the rank statistic for all pandas Series in `s_ranked_tags`.

    """
    return reduce(lambda x, y: x.multiply(y), s_ranked_tags).pow(1 / float(len(s_ranked_tags)))


def sort_and_drop_duplicates(df: pandas.DataFrame, col_rank: str, col_duplicates: List[str],
                             ascending: bool = False) -> pandas.DataFrame:
    """Get unique values in a pandas DataFrame based on the lowest or highest ranked value for each duplicate.

    This function first sorts `df` along `col_rank` either in ascending (`ascending=True`) or descending
    (`ascending=False`) order. It then drops any duplicates along the `col_duplicates` columns, before re-sorting the
    resultant pandas DataFrame index in ascending order, and returning it.

    :param df: A pandas DataFrame, potentially containing duplicate data, for sorting and extracting the lowest or
        highest ranked duplicate.
    :param col_rank: A column in `df` that contains the ranks of each row
    :param col_duplicates: A list of column names in `df` where there could be potentially duplicate rows of data.
    :param ascending: Default: False. If True, returns the lowest rank duplicate. If False, returns the highest
        ranked duplicate.
    :return: A pandas DataFrame containing only unique data.

    """
    return df.sort_values(by=col_rank, ascending=ascending).drop_duplicates(subset=col_duplicates).sort_index()


def concat_identical_columns(df1: pandas.DataFrame, df2: pandas.DataFrame) -> pandas.DataFrame:
    """Concatenate two pandas DataFrames along their common columns.

    :param df1: A pandas DataFrame with some columns common to `df2`.
    :param df2: A pandas DataFrame with some columns common to `df1`.
    :return: A pandas DataFrame with both `df1`, and `df2` concatenated together using only their common columns,
        with the returned index sorted in ascending order.

    """

    # Determine the identical columns across both `df1`, and `df2`, keeping the order of columns as seen in `df1`
    cols_identical = sorted(set(df1.columns) & set(df2.columns), key=list(df1.columns).index)

    # Assert that there must be some identical columns
    assert cols_identical, "Must have common columns between `df1` and `df2` - no common columns found!"

    # Return the concatenation of `df1`, and `df2` across common columns, and sort the index in ascending order
    return pandas.concat([df1[cols_identical], df2[cols_identical]]).sort_index()


def extract_unique_tags(df: pandas.DataFrame, col_key: str = "text_date", col_tags: Optional[List[str]] = None,
                        set_tag_ranks: Optional[Dict[Union[float, str], int]] = None,
                        out_col_rank_label: str = "rank") -> pandas.DataFrame:
    """Extract unique tags for all rows in a pandas DataFrame, which might contain different tags for duplicated data.

    The function operates as follows:

    1. Find all the duplicate rows of `df` that are not in `col_key` or `col_tags`;
    2. Sort `df` in descending order along `col_key`;
    3. For each column in `col_tags`, rank each row of `df`. The ranking is determined by either Step 2 or,
       if the value of the column already exists as a key in `set_tag_ranks`, use the value in `set_tag_ranks`. This
       ensures certain tags have a set rank, whilst ranking all others by Step 2;
    4. Calculate the rank statistic (geometric mean of ranks) for all `col_tags` ranks;
    5. Drop the duplicates according to the rank from Step 4; and
    6. Return a pandas DataFrame containing all the unique values from `df`, and the selected values from duplicate
       rows of `df` (Step 5).

    :param df: A pandas DataFrame, potentially containing duplicate data, where unique values are required.
    :param col_key: A column in `df` for sorting the data in descending order.
    :param col_tags: Default: None. A list of columns in `df` containing tags for each row. For duplicate rows, these
        tags may be different. If None, will use a predefined list of columns - see the `COLS_TAGS` variable from
        `src.make_feedback_tagging.tagging_preprocessing` for further information.
    :param set_tag_ranks: Default: None. A dictionary of a predefined hierarchy or ranks for values in columns of
        `col_tags`, which can be different to ranking order from `col_key`. The dictionary keys are values from
        `col_tags`, and the dictionary values are their predefined rankings. Not all values of `col_tags` need to be
        represented here - those that are missing will be replaced by their corresponding value from the ranking of
        `col_key`. The values should all be less than 0, and in ascending order of priority. If None, will use a
        predefined list of columns - see the `ORDER_TAGS` variable from
        `src.make_feedback_tagging.tagging_preprocessing` for further information.
    :param out_col_rank_label: Default: "rank". A column name used to store the rankings - this is not returned,
        and is an internal variable; ensure you do not have a column named this in the `df`
    :return: A pandas DataFrame of unique values, where the duplicated values in `df` are selected using a rank based
        on the values columns in `col_tags`, and the sorting of `col_key`.

    """

    # Raise an AssertionError if `out_col_rank_label` is a column in `df`, as this is an internal variable
    assert out_col_rank_label not in df.columns, "`out_col_rank_label` cannot be a column in `df`; please change " \
                                                 f"this input argument: {out_col_rank_label}"

    # Set `col_tags` to `COLS_TAGS` if it is None
    if not col_tags:
        col_tags = COLS_TAGS

    # Set `set_tag_ranks` to `ORDER_TAGS` if it is None
    if not set_tag_ranks:
        set_tag_ranks = ORDER_TAGS

    # Find all other columns in `df`
    cols_others = [c for c in df.columns if c not in [col_key, *col_tags]]

    # Get all the rows in `df` that have duplicated data in the `col_others` columns
    df_duplicated = find_duplicated_rows(df, cols_others)

    # Rank `df_duplicated` along its `col_key`
    s_duplicated_rank = rank_rows(df_duplicated, col_key)

    # Get the ranks for multiple tag columns, pre-populating pre-defined ranks for some entries, and using
    # `s_duplicated_rank` for others
    s_duplicated_ranked_tags = rank_multiple_tags(df_duplicated, col_tags, s_duplicated_rank, set_tag_ranks)

    # Calculate the rank statistic (geometric mean of ranks) for the tag ranks, and assign the output to a column in
    # `df_duplicated` called `out_col_rank_label`
    df_duplicated = df_duplicated.assign(**{out_col_rank_label: get_rank_statistic(s_duplicated_ranked_tags)})

    # Get a selected value from the duplicated data, based on the rankings in `out_col_rank_label`
    df_selected = sort_and_drop_duplicates(df_duplicated, out_col_rank_label, cols_others)

    # Get the combined unique values in `df`, alongside the selected values from the duplicated values in `df`
    df_out = concat_identical_columns(df[~df.index.isin(df_duplicated.index)], df_selected)

    # Assert that there are no duplicate rows of data in `cols_others`, before returning `df_out`
    assert not df_out.duplicated(subset=cols_others, keep=False).any(), "Duplicate values remain after processing!"
    return df_out


def remove_pii(s: pandas.Series) -> pandas.Series:
    """Strip personally identifiable information (PII) from a pandas Series.

    Uses the src.PreProcess.replace_pii_regex method to clean PII.

    :param s: A pandas Series potentially containing PII.
    :return: `s` with any PII removed from it.

    """
    return s.replace(np.nan, "", regex=True).map(PreProcess.replace_pii_regex).str.lower()


def compile_free_text(df: pandas.DataFrame, cols_free_text: List[str], sep: str = "\n\n") -> pandas.Series:
    """Compile strings in different columns of a pandas DataFrame together.

    :param df: A pandas DataFrame
    :param cols_free_text: A list of columns in `df` that contain free text.
    :param sep: Default: "\n\n". A delimiter used to separate the aggregated free text.
    :return: A pandas Series which is a compilation of all `cols_free_text` text, delimited by `sep`.

    """
    return df[cols_free_text].agg(sep.join, axis=1)


def extract_lemma(s: pandas.Series) -> pandas.Series:
    """Extract lemma from a pandas Series of text.

    Leverages parallel processing and the `src.PreProcess.part_of_speech_tag` method to extract lemma.

    :param s: A pandas Series of text.
    :return: A pandas Series of the lemmas in `s`.

    """
    return s.progress_map(lambda t: " ".join([p[2] for tags in PreProcess.part_of_speech_tag(t) for p in tags]))


def clean_text(df: pandas.DataFrame, cols_free_text: List[str], out_col_free_text: str = "free_text",
               out_col_clean_text: str = "clean_text", out_col_lemma: str = "lemma",
               stop_words: Optional[List[str]] = None, n_cores: Optional[int] = None) -> pandas.DataFrame:
    """Clean free text columns in a pandas DataFrame.

    Cleaning process involves:

    1. Removal of personally identifiable information (PII), and setting text to lowercase;
    2. Compiling all free text columns together into a single column; and
    3. Removing stop words, and certain symbols; and
    4. Extracting the lemma of the resultant text from Step 3.

    :param df: A pandas DataFrame
    :param cols_free_text: A list of columns in `df` that contain free text, which requires cleaning.
    :param out_col_free_text: Default: "free_text". An outputted column with the compiled text form `cols_free_text`
        separated by `\n\n` with PII removed, and all in lowercase.
    :param out_col_clean_text: Default: "clean_text". Like `out_col_free_text`, but with all stopwords removed,
        and any of (, ), [, ], +, and *.
    :param out_col_lemma: Default: "lemma", the resultant, cleaned column of text that will be outputted in `df`.
    :param stop_words: Default: None. A list of stop words. If None, will use nltk.corpus.stopwords.
    :param n_cores: Default: None. Number of processors to parallelise operations over. If None, will use the maximum
        number of available processors.
    :return: A copy of `df` with an additional columns `out_col_free_text`, `out_col_clean_text`, and `out_col_lemma`
        containing the compiled free text, the cleaned free text, and their lemmas, respectively.

    """

    # If `stop_words` is None, use `STOPWORDS`
    if not stop_words:
        stop_words = STOPWORDS

    # Remove PII from the free text columns, and set them to lowercase
    df_out = df.assign(**{c: remove_pii(df[c]) for c in cols_free_text})

    # Compile the free text column
    s_free_text = compile_free_text(df_out, cols_free_text)

    # Strip out some symbols, split the text into words, and then recompile without stop words
    s_rm_stop_words = s_free_text.apply(
        lambda x: " ".join(t for t in re.sub(r"[()\[\]+*]", "", x).split() if t not in stop_words)
    )

    # Extract the lemma from `s_rm_stop_words` - do this in parallel
    s_lemma = parallelise_pandas(s_rm_stop_words, extract_lemma, n_cores)

    # Return `df_out` with `s_lemma`
    return df_out.assign(**{out_col_free_text: s_free_text, out_col_clean_text: s_rm_stop_words,
                            out_col_lemma: s_lemma})


def tagging_preprocessing(df: pandas.DataFrame, cols_free_text: List[str], col_key: str = "text_date",
                          col_tags: Optional[List[str]] = None,
                          set_tag_ranks: Optional[Dict[Union[float, str], int]] = None,
                          out_col_free_text: str = "free_text", out_col_clean_text: str = "clean_text",
                          out_col_lemma: str = "lemma", col_rank_label: str = "rank") -> pandas.DataFrame:
    """Preprocess the manually tagged data.

    The function operates as follows:

    1. Replace all punctuation in the column headers of `df` with an underscore; adjacent punctuation will all be
       replaced by a single underscore;
    2. Convert the datetime-like string column `col_key` of `df` into a datetime pandas Series;
    3. Unique values from `df` across the columns not in `col_key` or `col_tags`, selecting a specific row for
       duplicate data - see the `src.extract_unique_tags` function for more information; and
    4. Removal of all personally identifiable information (PII) in the `cols_free_text` columns, all text in these
       columns to lowercase, all text compiled into a single string as a new column `out_col_free_text`. Stopwords
       and symbols are also removed from `out_col_free_text`, and returned in `out_col_clean_text`. Lemmas from
       `out_col_clean_text` are returned in `out_col_lemma`.

    :param df: A pandas DataFrame potentially containing  duplicate data.
    :param cols_free_text: A list of columns in `df` that contain free text, which requires cleaning.
    :param col_key: Default: "text_date". A unique key column in `df` containing strings that start with the datetime
        format "YYYY-mm-dd HH:MM:SS".
    :param col_tags: Default: None. A list of columns in `df` containing tags for each row. For duplicate rows, these
        tags may be different. If None, will use a predefined list of columns - see the `COLS_TAGS` variable from
        `src.make_feedback_tagging.tagging_preprocessing` for further information.
    :param set_tag_ranks: Default: None. A dictionary of a predefined hierarchy or ranks for values in columns of
        `col_tags`, which can be different to ranking order from `col_key`. The dictionary keys are values from
        `col_tags`, and the dictionary values are their predefined rankings. Not all values of `col_tags` need to be
        represented here - those that are missing will be replaced by their corresponding value from the ranking of
        `col_key`. The values should all be less than 0, and in ascending order of priority. If None, will use a
        predefined list of columns - see the `ORDER_TAGS` variable from
        `src.make_feedback_tagging.tagging_preprocessing` for further information.
    :param out_col_free_text: Default: "free_text". An outputted column with the compiled text form `cols_free_text`
        separated by `\n\n` with PII removed, and all in lowercase.
    :param out_col_clean_text: Default: "clean_text". Like `out_col_free_text`, but with all stopwords removed,
        and any of (, ), [, ], +, and *.
    :param out_col_lemma: Default: "lemma". An extra column outputted in `df` that contains the compiled, lemmatised
        free text.
    :param col_rank_label: Default: "rank". A column name used to store the rankings - this is not returned, and is an
        internal variable; ensure you do not have a column named this in the `df`
    :return: A pandas DataFrame of unique values, where the duplicated values in `df` are selected using a rank based
        on the values columns in `col_tags`, and the sorting of `col_key`, where `col_key` is now a datetime object.
        All column headers will also be in lowercase, with punctuation stripped and replaced with underscores. All
        `cols_free_text` will have PII removed, and will be in lowercase. Additional columns `out_col_free_text`,
        `out_col_clean_text`, and `out_col_lemma` containing the compiled `cols_free_text` text,
        the same text with stopwords and certain synbols removed, and their lemmas, respectively.

    """

    # Standardise the column headers of `df`
    df_process = standardise_columns(df)

    # Convert `col_key` to a datetime
    df_process = convert_object_to_datetime(df_process, col_key)

    # Process the data to remove duplicate data outside of `col_key` and `col_tags`
    df_process = extract_unique_tags(df_process, col_key, col_tags, set_tag_ranks, col_rank_label)

    # Clean `cols_free_text`, and return the compiled free text, and its cleaned and its lemmatised versions
    return clean_text(df_process, cols_free_text, out_col_free_text, out_col_clean_text, out_col_lemma)
