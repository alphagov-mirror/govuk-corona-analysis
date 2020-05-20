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
