from src.make_feedback_tool_data.preprocess import PreProcess
from src.make_feedback_tool_data.regex_categorisation import regex_for_theme, regex_group_verbs
from src.make_feedback_tool_data.text_chunking import ChunkParser
from tqdm import tqdm
from typing import Optional
import logging.config
import os
import nltk
import numpy as np
import pandas as pd
import re

nltk.download("punkt")

# Set up a logger
log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logging.conf")
logging.config.fileConfig(log_file_path)
logger = logging.getLogger(__name__)


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Dropped duplicated rows, based on the primary_key column, which is a unique session identifier.

    :param df: A pandas DataFrame with a column called `primary_key`, which (potentially) contains duplicate data
    :return: A pandas DataFrame identical to `df`, except duplicates along the `primary_key` column are dropped.

    """

    # Calculate the number of rows in `df`, and log this information, alongside other statistics
    loaded_number_rows = df.shape[0]
    logger.info(f"Number of rows: {loaded_number_rows}")
    logger.info(f"Unique clientIds: {df.intents_clientID.nunique()}")
    logger.info(f"Unique primary key: {df.primary_key.nunique()}")
    logger.info(f"Unique session_ids: {df.session_id.nunique()}")
    logger.info("Dropping duplicates...")

    # Drop the duplicates using the `primary_key` column, and reset the index
    df_out = df.drop_duplicates("primary_key").reset_index(drop=True)
    logger.info(f"Dropped {loaded_number_rows - df.shape[0]} rows.")

    # Return the de-duplicated pandas DataFrame
    return df_out


def preprocess_filter_comment_text(df: pd.DataFrame, length_threshold: int = 4000) -> pd.DataFrame:
    """Filter down text to only English text and comments below a character length threshold.

    Also removes personally identifiable information (PII) from the text, according to the
    `PreProcess.replace_pii_regex` method.

    :param df: A pandas DataFrame with a text column `Q3_x` for filtering.
    :param length_threshold: Default: 4000. The maximum number of characters any text within `Q3_x` of `df` can have
        - only rows less than this character limit are retained. All others are filtered out.
    :return: A pandas DataFrame with PII removed, and only English text below the character length threshold.

    """
    logger.info("Removing non-English and lengthy comments...")

    # Remove any personally identifiable information (PII), and filter out any rows were the text length is greater than
    # or equal to `length_threshold`
    out_df = df.assign(Q3_pii_removed=df["Q3_x"].progress_map(PreProcess.replace_pii_regex)) \
        .query(f"Q3_pii_removed.str.len() < {length_threshold}")

    # Detect the language of each row of PII-removed text
    out_df = out_df.assign(language=out_df["Q3_pii_removed"].progress_map(PreProcess.detect_language))

    # Get the count of distinct languages
    lang_dist = out_df["language"].value_counts().to_dict()

    # Log the counts from `lang_dist`
    logger.debug(f"Number of unique languages: {len(lang_dist)}")
    for k, v in lang_dist.items():
        logger.debug(f"{k}: {v / sum(lang_dist.values()):.2%}")

    # Add a flag if the language is English, and filter for English only; return this output
    return out_df.assign(is_en=out_df["language"].isin(["en", "un", "-", "sco"])).query("is_en")


def extract_phrase_mentions(df: pd.DataFrame, grammar_filename: Optional[str] = None) -> pd.DataFrame:
    """Extract phrase mentions from the text.

    For each POS-tagged sentence from comments in the survey data:

    1. Detect and extract chunks as defined by grammar, merge adjacent chunks
    2. Compute pair-wise combinations of chunks
    3. If a combination type is in predefined list, append it to phrase_mentions list

    :param df: A filtered, preprocessed survey pandas DataFrame.
    :param grammar_filename: Default: None. A path string to file containing regular expression grammar patterns
        usable by the `grammar` argument of the nltk.chunk.regexp.RegexpParser class. For each grammar type,
        each pattern should be listed on a separate line, and in descending order of priority (highest first). If
        None, it will use the default regular expression file - see
        src.make_feedback_tool_data.text_chunking.ChunkParser for further details.
    :return: `df` with an additional column containing applicable phrase mentions.

    """

    logger.info("Detecting and extracting phrase-level mentions...")

    # Initialise a storing variable for the phrase mentions
    phrase_mentions = []

    # Initialise a `ChunkParser` class
    parser = ChunkParser(grammar_filename)

    # Iterate through the comments and the POS tagged text
    for comment, vals in tqdm(df[["Q3_x_edit", "pos_tag"]].values):

        # Extract phrase mentions, and combine similar phrases together
        sents = parser.extract_phrase(vals, merge_inplace=True)

        # Add additional storing elements to `phrase_mentions`
        phrase_mentions.append([])

        # Examine sequential pairwise combinations of `sents`
        for combo in PreProcess.compute_combinations(sents, 2):

            # Extract label and text for each pairwise combination
            key = (combo[0].label, combo[1].label)
            arg1 = combo[0].text.lower()
            arg2 = combo[1].text.lower()

            # If the labels for each combination match any of these, get the phrase mention
            if key in [("verb", "noun"), ("verb", "prep_noun"), ("verb", "noun_verb"), ("noun", "prep_noun"),
                       ("prep_noun", "noun"), ("prep_noun", "prep_noun")]:

                # Define a generic phrase for the text in the combination using regular expressions
                generic_phrase = (regex_group_verbs(arg1), regex_for_theme(arg2))

                # Remove certain characters from `arg1`, and `arg2` using regular expressions, and combine together
                # in a tuple
                arg1, arg2 = [re.sub(r"[?()\[\]+*]", "", a) for a in (arg1, arg2)]
                phrase = (arg1, arg2)

                # Get a phrase that matches `comment`
                exact_phrase = list(PreProcess.find_needle(" ".join(phrase), comment.lower()).values())[0]

                # Get the verb that matches `exact_phrase`
                if exact_phrase is not None:
                    exact_verb = list(PreProcess.find_needle(arg1, exact_phrase).values())[0]

                    # If `exact_verb` exists, then remove it out from `exact_phrase`, trim any white space,
                    # and append all the information to `phrase_mentions`
                    if exact_verb is not None:
                        exact_phrase = (exact_verb, re.sub(exact_verb, "", exact_phrase).strip())
                        phrase_mentions[-1].append({"chunked_phrase": phrase, "exact_phrase": exact_phrase,
                                                    "generic_phrase": generic_phrase, "key": key})

    # Return `df` with a new column for `phrase_mentions`
    return df.assign(themed_phrase_mentions=phrase_mentions)


def save_intermediate_df(df: pd.DataFrame, cache_pos_filename: str) -> None:
    """Save intermediate data processing once lemmas and words have been extracted from parts-of-speech tagging.

    :param df: A partially processed pandas DataFrame for caching.
    :param cache_pos_filename: A file path for the cached `df`.
    :return: None. Saves a cached version of `df` in the location specified by `cache_pos_filename`.

    """
    logger.info(f"Saving preprocessed survey data at: {cache_pos_filename}...")

    # Extract lemmas and words from parts-of-speech tagging
    out_df = df.assign(
        lemmas=df["pos_tag"].progress_map(lambda x: [token[2] for sent in x for token in sent]),
        words=df["pos_tag"].progress_map(lambda x: [token[0] for sent in x for token in sent])
    )

    # Save the intermediate pandas DataFrame
    out_df.to_csv(cache_pos_filename, index=False)


def create_phrase_level_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Create exact and generic phrase mention columns, based on verb-like phrase mentions only.

    :param df: A pandas DataFrame containing at least a column `themed_phrase_mentions`. See
        src.make_feedback_tool_data.extract_phrase_mentions.
    :return: `df` with additional columns for exact and generic phrase mentions.

    """

    # Compile the extract phrases column - only uses phrase mentions where the first item is a verb
    logger.info("Assigning exact_phrases column...")
    out_df = df.assign(exact_phrases=df["themed_phrase_mentions"].progress_map(
        lambda x: "\n".join([", ".join(item["exact_phrase"]) for item in x if item["key"][0] == "verb"])
    ))

    # Compile the generic phrases column - only uses phrase mentions where the first item is a verb
    logger.info("Assigning generic_phrases column...")
    out_df = out_df.assign(generic_phrases=out_df["themed_phrase_mentions"].progress_map(
        lambda x: "\n".join([", ".join(item["generic_phrase"]) for item in x if item["key"][0] == "verb"])
    ))

    # Return the amended pandas DataFrame
    return out_df


def create_dataset(survey_filename: str, grammar_filename: str, cache_pos_filename: str, output_filename: str) -> None:
    """Process the survey data, and generate outputs.

    :param survey_filename: A file path where the survey data is located.
    :param grammar_filename: A file path where the regular expressions grammar file is located. See
        src.make_feedback_tool_data.text_chunking.ChunkParser for further details
    :param cache_pos_filename: A file path where partially-processed data will be cached for review.
    :param output_filename: A file path where the processed data will be cached.
    :return: None in Python. Will create two CSV files in the file paths defined by `cache_pos_filename` and
        `output_filename`; the first CSV is the partially-processed data, whilst the second is the final output.

    """

    # Read in the survey data
    logger.info(f"Reading survey file: {survey_filename}")
    df = pd.read_csv(survey_filename)

    # Drop any duplicate rows along the `primary_key` column of `survey_data_df`
    survey_data_df = drop_duplicate_rows(df)

    # Remove personally identifiable information (PII), and keep only rows with English comments less than
    # 4,000 characters long
    survey_data_df = preprocess_filter_comment_text(survey_data_df)

    # Extract the part-of-speech (POS) tags for the comments
    logger.info("Part of speech tagging comments...")
    survey_data_df = survey_data_df.assign(
        pos_tag=survey_data_df[["Q3_pii_removed", "is_en"]].progress_apply(
            lambda x: PreProcess.part_of_speech_tag(x[0]) if x[1] else [],
            axis=1
        )
    )

    # Replace NaN values, and pre-process the feedback text
    logger.info("Pre-processing feedback text for matching...")
    survey_data_df = survey_data_df.assign(Q3_x_edit=survey_data_df["Q3_x"].replace(np.nan, "", regex=True))
    survey_data_df = survey_data_df.assign(
        Q3_x_edit=survey_data_df["Q3_x_edit"].progress_map(lambda x: " ".join(re.sub(r"[()\[\]+*]", "", x).split()))
    )

    # Extract the phrase mentions
    survey_data_df = extract_phrase_mentions(survey_data_df, grammar_filename)

    # Save the partially-processed `survey_data_df`
    save_intermediate_df(survey_data_df, cache_pos_filename)

    # Create phrase-level columns
    survey_data_df = create_phrase_level_columns(survey_data_df)

    # Overwrite the `Q3_x` column with `Q3_x_edit`
    survey_data_df = survey_data_df.assign(Q3_x=survey_data_df["Q3_x_edit"])

    # Define the columns to keep - all the original columns in `df`, but also the `exact_phrases`,
    # and `generic_phrases` columns
    columns_to_keep = [*df.columns, "exact_phrases", "generic_phrases"]

    # Output the file to a CSV; only output the same columns as defined in `columns_to_keep`
    logger.info(f"Saving survey data at: {output_filename}...")
    survey_data_df[columns_to_keep].to_csv(output_filename, index=False)


if __name__ == "__main__":
    # Get environment variables
    DATA_DIR = os.getenv("DIR_DATA")

    # Define paths to various files
    survey_data_filename = os.path.join(DATA_DIR, "uis_20200401_20200409.csv")
    chunk_grammar_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grammar.txt")
    cache_pos_data_filename = survey_data_filename.replace(".csv", "_cache.csv")
    output_data_filename = survey_data_filename.replace(".csv", "_exact_generic_phrases.csv")

    # Execute the `create_dataset` function
    create_dataset(survey_data_filename, chunk_grammar_filename, cache_pos_data_filename, output_data_filename)
    # parser = ChunkParser(chunk_grammar_filename)
    # comment = "This is an example sentence. This is another."
    # tagged = PreProcess.part_of_speech_tag(comment)
    # for sent in parser.extract_phrase(tagged, merge_inplace=True):
    #     for chunk in sent:
    #         print(chunk.text, chunk.label)
