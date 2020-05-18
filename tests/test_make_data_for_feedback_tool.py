from ast import literal_eval
from datetime import timedelta
from faker import Faker
from src.make_feedback_tool_data.make_data_for_feedback_tool import (
    create_dataset,
    create_phrase_level_columns,
    drop_duplicate_rows,
    extract_phrase_mentions,
    preprocess_filter_comment_text,
    save_intermediate_df
)
from src.make_feedback_tool_data.preprocess import PreProcess
from pandas.testing import assert_frame_equal
import numpy as np
import pandas as pd
import pytest
import random
import re

# Set the random seed
random.seed(42)

# Create an example pandas DataFrame of data
DF_EXAMPLE_RAW = pd.DataFrame.from_dict({
    "primary_key": [*range(7)],
    "Q3": [
        "", "-", "These words are in English.", "Cet mots sont en français.",
        "This is in English, but there is a word here in اَلْعَرَبِيَّةُ",
        "Η Βικιπαίδεια είναι διεθνής, παγκόσμια, ψηφιακή, διαδικτυακή, ελεύθερου περιεχομένου, εγκυκλοπαίδεια, που "
        "βασίζεται σε ένα μοντέλο ανοικτό στη σύνταξη του περιεχομένου της. It is the largest and most popular general "
        "reference work on the World Wide Web, and is one of the 20 most popular websites ranked by Alexa, as of March "
        "2020.",
        "維基百科 是维基媒体基金会运营的一个多语言的線上百科全書，并以创建和维护作为开放式协同合作项目，特点是自由內容、自由编辑、自由版权"
    ],
})

# Define example personally identifiable information for `EXAMPLE_PARAGRAPHS`
EXAMPLE_PII_REGEX = r"(English)|(World Wide Web)|(mots)"

# Create a pre-processed version of `DF_EXAMPLE`
DF_EXAMPLE_PRE_PROCESSED = DF_EXAMPLE_RAW \
    .assign(Q3_pii_removed=DF_EXAMPLE_RAW["Q3"].str.replace(EXAMPLE_PII_REGEX, ""),
            language=["un", "-", "en", "fr", "en", "el", "zh"],
            is_en=[True, True, True, False, True, False, False]) \
    .query("is_en")


@pytest.fixture
def patch_preprocess_pii_regex(mocker):
    """Patch the replace_pii_regex method of the PreProcess class with EXAMPLE_PII_REGEX."""
    return mocker.patch.object(PreProcess, "replace_pii_regex", side_effect=lambda s: re.sub(EXAMPLE_PII_REGEX, "", s))


@pytest.fixture
def patch_preprocess_detect_language(mocker):
    """Patch the detect_language method of the PreProcess class"""
    return mocker.patch.object(PreProcess, "detect_language")


@pytest.mark.parametrize("test_input_threshold", [*range(60, 110, 10)])
class TestPreProcessFilterCommentText:

    def test_returns_correctly(self, patch_preprocess_pii_regex, test_input_threshold):
        """Test that the preprocess_filter_comment_text function returns the correct output."""

        # Define the expected output
        test_expected = DF_EXAMPLE_PRE_PROCESSED.query(f"Q3_pii_removed.str.len() < {test_input_threshold}")

        # Call the `preprocess_filter_comment_text` function
        test_output = preprocess_filter_comment_text(DF_EXAMPLE_RAW, test_input_threshold)

        # Assert the same columns exist in both
        assert set(test_output.columns) == set(test_expected.columns)

        # Assert the output is as expected
        assert_frame_equal(test_output, test_expected)

    def test_preprocess_replace_pii_regex_call_count(self, patch_preprocess_pii_regex, test_input_threshold):
        """Test that preprocess_filter_comment_text calls PreProcess.replace_pii_regex the correct number of times."""

        # Call the `preprocess_filter_comment_text` function
        _ = preprocess_filter_comment_text(DF_EXAMPLE_RAW, test_input_threshold)

        # Assert that `PreProcess.replace_pii_regex` is called the correct number of times
        assert patch_preprocess_pii_regex.call_count == len(DF_EXAMPLE_RAW)

    def test_preprocess_replace_pii_regex_called_correctly(self, mocker, patch_preprocess_pii_regex,
                                                           test_input_threshold):
        """Test that preprocess_filter_comment_text calls PreProcess.replace_pii_regex with the correct arguments."""

        # Call the `preprocess_filter_comment_text` function
        _ = preprocess_filter_comment_text(DF_EXAMPLE_RAW, test_input_threshold)

        # Assert that `PreProcess.replace_pii_regex` is called with the correct arguments
        assert patch_preprocess_pii_regex.call_args_list == [mocker.call(v) for v in DF_EXAMPLE_RAW["Q3"]]

    def test_preprocess_detect_language_call_count(self, patch_preprocess_pii_regex, patch_preprocess_detect_language,
                                                   test_input_threshold):
        """Test that preprocess_filter_comment_text calls PreProcess.detect_language the correct number of times."""

        # Call the `preprocess_filter_comment_text` function
        _ = preprocess_filter_comment_text(DF_EXAMPLE_RAW, test_input_threshold)

        # Get the expected call count
        test_expected = DF_EXAMPLE_RAW \
            .assign(Q3_pii_removed=DF_EXAMPLE_RAW["Q3"].str.replace(EXAMPLE_PII_REGEX, "")) \
            .query(f"Q3_pii_removed.str.len() < {test_input_threshold}") \
            .shape[0]

        # Assert that `PreProcess.replace_pii_regex` is called the correct number of times
        assert patch_preprocess_detect_language.call_count == test_expected

    def test_preprocess_detect_language_called_correctly(self, mocker, patch_preprocess_pii_regex,
                                                         patch_preprocess_detect_language, test_input_threshold):
        """Test that preprocess_filter_comment_text calls PreProcess.detect_language with the correct arguments."""

        # Call the `preprocess_filter_comment_text` function
        _ = preprocess_filter_comment_text(DF_EXAMPLE_RAW, test_input_threshold)

        # Define the expected values of the call arguments
        text_expected_values = DF_EXAMPLE_RAW \
            .assign(Q3_pii_removed=DF_EXAMPLE_RAW["Q3"].str.replace(EXAMPLE_PII_REGEX, "")) \
            .query(f"Q3_pii_removed.str.len() < {test_input_threshold}") \
            .Q3_pii_removed \
            .to_list()

        # Assert that `PreProcess.detect_language` is called with the correct arguments
        assert patch_preprocess_detect_language.call_args_list == [mocker.call(v) for v in text_expected_values]


# Define input arguments for the `TestSaveIntermediateDf` test class; the first text is 'I am going to go and test to
# see if this example is correct.', the second text is 'If this test passes, we should be able to extract lemma and
# words.', the third text is a combination of the first and second text into a two-sentence text, and the fourth text
# is 'I tried to signed up for advice due to the ongoing COVID 19 outbreak with specific concern about vulnerable
# people. I could not!'
args_save_intermediate_df_inputs = [
    {"pos_tag": [[[("I", "PRP", "-PRON-"), ("am", "VBP", "be"), ("going", "VBG", "go"), ("to", "TO", "to"),
                  ("go", "VB", "go"), ("and", "CC", "and"), ("test", "VB", "test"), ("to", "TO", "to"),
                  ("see", "VB", "see"), ("if", "IN", "if"), ("this", "DT", "this"), ("example", "NN", "example"),
                  ("is", "VBZ", "be"), ("correct", "JJ", "correct"), (".", ".", ".")]]]},
    {"pos_tag": [[[("If", "IN", "if"), ("this", "DT", "this"), ("test", "NN", "test"), ("passes", "VBZ", "pass"),
                  (",", ",", ","), ("we", "PRP", "-PRON-"), ("should", "MD", "should"), ("be", "VB", "be"),
                  ("able", "JJ", "able"), ("to", "TO", "to"), ("extract", "VB", "extract"), ("lemma", "NN", "lemma"),
                  ("and", "CC", "and"), ("words", "NNS", "word"), (".", ".", ".")]]]},
    {"pos_tag": [[[("I", "PRP", "-PRON-"), ("am", "VBP", "be"), ("going", "VBG", "go"), ("to", "TO", "to"),
                  ("go", "VB", "go"), ("and", "CC", "and"), ("test", "VB", "test"), ("to", "TO", "to"),
                  ("see", "VB", "see"), ("if", "IN", "if"), ("this", "DT", "this"), ("example", "NN", "example"),
                  ("is", "VBZ", "be"), ("correct", "JJ", "correct"), (".", ".", ".")],
                 [("If", "IN", "if"), ("this", "DT", "this"), ("test", "NN", "test"), ("passes", "VBZ", "pass"),
                  (",", ",", ","), ("we", "PRP", "-PRON-"), ("should", "MD", "should"), ("be", "VB", "be"),
                  ("able", "JJ", "able"), ("to", "TO", "to"), ("extract", "VB", "extract"), ("lemma", "NN", "lemma"),
                  ("and", "CC", "and"), ("words", "NNS", "word"), (".", ".", ".")]]]},
    {"pos_tag": [[[("I", "PRP", "-PRON-"), ("tried", "VBD", "try"), ("to", "TO", "to"), ("signed", "VBN", "sign"),
                  ("up", "RP", "up"), ("for", "IN", "for"), ("advice", "NN", "advice"), ("due", "IN", "due"),
                  ("to", "IN", "to"), ("the", "DT", "the"), ("ongoing", "JJ", "ongoing"), ("COVID", "NNP", "COVID"),
                  ("19", "CD", "19"), ("outbreak", "NN", "outbreak"), ("with", "IN", "with"),
                  ("specific", "JJ", "specific"), ("concern", "NN", "concern"), ("about", "IN", "about"),
                  ("vulnerable", "JJ", "vulnerable"), ("people", "NNS", "people")],
                 [("I", "PRP", "-PRON-"), ("could", "MD", "could"), ("not", "RB", "not"), ("!", ".", "!")]]]}
]

# Define the additional expected columns in the outputted CSV by the `save_intermediate_df` function - this will be
# in addition to the columns in `args_save_intermediate_df_inputs`
args_save_intermediate_df_expected = [
    {"lemmas": [["-PRON-", "be", "go", "to", "go", "and", "test", "to", "see", "if", "this", "example", "be",
                 "correct", "."]],
     "words": [["I", "am", "going", "to", "go", "and", "test", "to", "see", "if", "this", "example", "is", "correct",
                "."]]},
    {"lemmas": [["if", "this", "test", "pass", ",", "-PRON-", "should", "be", "able", "to", "extract", "lemma",
                 "and", "word", "."]],
     "words": [["If", "this", "test", "passes", ",", "we", "should", "be", "able", "to", "extract", "lemma", "and",
                "words", "."]]},
    {"lemmas": [["-PRON-", "be", "go", "to", "go", "and", "test", "to", "see", "if", "this", "example", "be",
                 "correct", ".", "if", "this", "test", "pass", ",", "-PRON-", "should", "be", "able", "to",
                 "extract", "lemma", "and", "word", "."]],
     "words": [["I", "am", "going", "to", "go", "and", "test", "to", "see", "if", "this", "example", "is", "correct",
                ".", "If", "this", "test", "passes", ",", "we", "should", "be", "able", "to", "extract", "lemma", "and",
                "words", "."]]},
    {"lemmas": [["-PRON-", "try", "to", "sign", "up", "for", "advice", "due", "to", "the", "ongoing", "COVID", "19",
                 "outbreak", "with", "specific", "concern", "about", "vulnerable", "people", "-PRON-", "could", "not",
                 "!"]],
     "words": [["I", "tried", "to", "signed", "up", "for", "advice", "due", "to", "the", "ongoing", "COVID", "19",
                "outbreak", "with", "specific", "concern", "about", "vulnerable", "people", "I", "could", "not", "!"]]}
]

# Create the test cases for the `TestSaveIntermediateDf` test class, where each tuple in the list consists of two
# elements; the first is each element of `args_save_intermediate_df_inputs` as a pandas DataFrame, and the second is
# a pandas DataFrame of the corresponding elements from `args_save_intermediate_df_inputs` and
# `args_save_intermediate_df_expected` as the expected outputs
args_save_intermediate_df = [
    (pd.DataFrame(i), pd.DataFrame({**i, **e})) for i, e in zip(args_save_intermediate_df_inputs,
                                                                args_save_intermediate_df_expected)
]


@pytest.fixture
def patch_pandas_dataframe_to_csv(mocker):
    """Patch the pandas.DataFrame.to_csv method."""
    return mocker.patch("pandas.DataFrame.to_csv")


@pytest.fixture
def temp_folder(tmpdir_factory):
    """Create a temporary directory to store the output from save_intermediate_df."""
    return tmpdir_factory.mktemp("temp")


@pytest.mark.parametrize("test_input_cache_pos_filename", ["foo.csv", "bar.csv"])
class TestSaveIntermediateDf:

    @pytest.mark.parametrize("test_input_df", [a[0] for a in args_save_intermediate_df])
    def test_calls_to_csv_correctly(self, patch_pandas_dataframe_to_csv, test_input_df, test_input_cache_pos_filename):
        """Test save_intermediate_df calls pandas.DataFrame.to_csv correctly."""

        # Call the `save_intermediate_df` function
        save_intermediate_df(test_input_df, test_input_cache_pos_filename)

        # Assert `pandas.DataFrame.to_csv` is called with the correct arguments
        patch_pandas_dataframe_to_csv.assert_called_once_with(test_input_cache_pos_filename, index=False)

    @pytest.mark.parametrize("test_input_df, test_expected_df", args_save_intermediate_df)
    def test_returns_correctly(self, temp_folder, test_input_df, test_input_cache_pos_filename, test_expected_df):
        """Test the outputted CSV from save_intermediate_df is correct."""

        # Define the file path for the CSV
        test_input_file_path = temp_folder.join(test_input_cache_pos_filename)

        # Call the `save_intermediate_df` function
        save_intermediate_df(test_input_df, test_input_file_path)

        # Assert the CSV output is correct; need to apply `ast.literal_eval` element-wise, as the CSV will contain
        # strings of the lists, rather than the lists themselves
        assert_frame_equal(pd.read_csv(test_input_file_path).applymap(literal_eval), test_expected_df)


# Define the example feedback that would result in `args_save_intermediate_df_inputs`
args_extract_phrase_mentions_inputs_q3_edit = [
    "I am going to go and test to see if this example is correct.",
    "If this test passes, we should be able to extract lemma and words.",
    "I am going to go and test to see if this example is correct. If this test passes, we should be able to extract "
    "lemma and words.",
    "I tried to signed up for advice due to the ongoing COVID 19 outbreak with specific concern about vulnerable "
    "people. I could not!"
]


# Define the inputs for the `extract_phrase_mentions` tests, where each tuple is a pandas DataFrame with columns
# 'Q3_edit' and 'pos_tag'
args_extract_phrase_mentions_integration = [
    pd.DataFrame({"Q3_edit": t, **i}) for t, i in zip(args_extract_phrase_mentions_inputs_q3_edit,
                                                      args_save_intermediate_df_inputs)
]


@pytest.fixture
def patch_chunkparser_extract_phrase(mocker):
    """Patch both the ChunkParser class, and its extract_phrase method, but only return the latter."""
    patch_chunkparser = mocker.patch("src.make_feedback_tool_data.make_data_for_feedback_tool.ChunkParser")
    return patch_chunkparser.return_value.extract_phrase


@pytest.fixture
def patch_preprocess(mocker):
    """Patch the PreProcess class."""
    return mocker.patch("src.make_feedback_tool_data.make_data_for_feedback_tool.PreProcess")


@pytest.mark.parametrize("test_input_df", args_extract_phrase_mentions_integration)
@pytest.mark.parametrize("test_input_grammar_filename", [None, "hello.txt", "world.txt"])
class TestExtractPhraseMentionsIntegration:

    def test_calls_correctly(self, mocker, test_input_df, test_input_grammar_filename):
        """Test extract_phrase_mentions calls ChunkParser correctly."""

        # Patch the `ChunkParser` class
        patch_chunkparser = mocker.patch("src.make_feedback_tool_data.make_data_for_feedback_tool.ChunkParser")

        # Call the `extract_phrase_mentions` function
        _ = extract_phrase_mentions(test_input_df, test_input_grammar_filename)

        # Assert `ChunkParser` is called once with the correct arguments
        patch_chunkparser.assert_called_once_with(test_input_grammar_filename)

    def test_calls_extract_phrase(self, mocker, patch_chunkparser_extract_phrase, test_input_df,
                                  test_input_grammar_filename):
        """Test extract_phrase_mentions calls ChunkParser.extract_phrase correctly."""

        # Call the `extract_phrase_mentions` function
        _ = extract_phrase_mentions(test_input_df, test_input_grammar_filename)

        # Assert `ChunkParser.extract_phrase` is called the correct number of times
        assert patch_chunkparser_extract_phrase.call_count == len(test_input_df)

        # Assert `ChunkParser.extract_phrase` is called with the correct arguments
        for v in test_input_df["pos_tag"].values:
            assert patch_chunkparser_extract_phrase.call_args_list == [mocker.call(v, merge_inplace=True)]

    def test_calls_preprocess_compute_combinations_correctly(self, mocker, patch_chunkparser_extract_phrase,
                                                             patch_preprocess, test_input_df,
                                                             test_input_grammar_filename):
        """Test extract_phrase_mentions calls PreProcess.compute_combinations correctly."""

        # Call the `extract_phrase_mentions` function
        _ = extract_phrase_mentions(test_input_df, test_input_grammar_filename)

        # Assert `PreProcess.compute_combinations` is called the correct number of times
        assert patch_preprocess.compute_combinations.call_count == len(test_input_df)

        # Define the expected call argument for each iteration - this will be the return value from calling
        # `ChunkParser.extract_phrase`
        test_expected = [mocker.call(patch_chunkparser_extract_phrase.return_value, 2)]

        # Assert `ChunkParser.extract_phrase` is called with the correct arguments
        assert patch_preprocess.compute_combinations.call_args_list == test_expected * len(test_input_df)


# Define the expected call arguments for `regex_group_verbs`
args_regex_group_verbs_call_args_expected = [
    ["test to see if"],
    ["to extract"],
    ["test to see if", "to extract"],
    ["tried to signed up for", "advice", "due to the ongoing covid 19 outbreak", "with specific concern"]
]

# Define the expected call arguments for `regex_for_theme`
args_regex_for_theme_call_args_expected = [
    ["this example"],
    ["lemma"],
    ["this example", "lemma"],
    ["advice", "due to the ongoing covid 19 outbreak", "with specific concern", "about vulnerable people"]
]

# Define the test cases for the `test_calls_regex_group_verbs_correctly` test in the
# `TestExtractPhraseMentionsIntegrationComboSection` test class
args_calls_regex_group_verbs_correctly = [
    (i.copy(deep=True), e) for i, e in zip(args_extract_phrase_mentions_integration,
                                           args_regex_group_verbs_call_args_expected)
]

# Define the test cases for the `test_calls_regex_for_theme_correctly` test in the
# `TestExtractPhraseMentionsIntegrationComboSection` test class
args_calls_regex_for_theme_correctly = [
    (i.copy(deep=True), e) for i, e in zip(args_extract_phrase_mentions_integration,
                                           args_regex_for_theme_call_args_expected)
]

# Define the expected call arguments for the `PreProcess.find_needle` method in `extract_phrase_mentions`
args_find_needle_called_correctly_expected = [
    ([("test to see if this example", "i am going to go and test to see if this example is correct."),
      ("test to see if", "test to see if this example")]),
    ([("to extract lemma", "if this test passes, we should be able to extract lemma and words."),
      ("to extract", "to extract lemma")]),
    ([("test to see if this example", "i am going to go and test to see if this example is correct. if this test "
                                      "passes, we should be able to extract lemma and words."),
      ("test to see if", "test to see if this example"),
      ("to extract lemma", "i am going to go and test to see if this example is correct. if this test passes, "
                           "we should be able to extract lemma and words."),
      ("to extract", "to extract lemma")]),
    ([("tried to signed up for advice", "i tried to signed up for advice due to the ongoing covid 19 outbreak with "
                                        "specific concern about vulnerable people. i could not!"),
      ("tried to signed up for", "tried to signed up for advice"),
      ("advice due to the ongoing covid 19 outbreak", "i tried to signed up for advice due to the ongoing covid 19 "
                                                      "outbreak with specific concern about vulnerable people. i could "
                                                      "not!"),
      ("advice", "advice due to the ongoing covid 19 outbreak"),
      ("due to the ongoing covid 19 outbreak with specific concern", "i tried to signed up for advice due to the "
                                                                     "ongoing covid 19 outbreak with specific concern "
                                                                     "about vulnerable people. i could not!"),
      ("due to the ongoing covid 19 outbreak", "due to the ongoing covid 19 outbreak with specific concern"),
      ("with specific concern about vulnerable people", "i tried to signed up for advice due to the ongoing covid 19 "
                                                        "outbreak with specific concern about vulnerable people. i "
                                                        "could not!"),
      ("with specific concern", "with specific concern about vulnerable people")])
]


# Define the test cases for the `test_find_needle_called_correctly` test in the
# `TestExtractPhraseMentionsIntegrationComboSection` test class
args_find_needle_called_correctly = [
    (i.copy(deep=True), e) for i, e in zip(args_extract_phrase_mentions_integration,
                                           args_find_needle_called_correctly_expected)
]


class TestExtractPhraseMentionsIntegrationComboSection:

    @pytest.mark.parametrize("test_input, test_expected", args_calls_regex_group_verbs_correctly)
    def test_calls_regex_group_verbs_correctly(self, mocker, test_input, test_expected):
        """Test extract_phrase_mentions calls regex_group_verbs correctly."""

        # Patch the `regex_group_verbs` function
        patch_regex_group_verbs = mocker.patch(
            "src.make_feedback_tool_data.make_data_for_feedback_tool.regex_group_verbs"
        )

        # Call the `extract_phrase_mentions` function; assumes the default grammar file is unchanged
        _ = extract_phrase_mentions(test_input, None)

        # Assert `regex_group_verbs` is called the expected number of times
        assert patch_regex_group_verbs.call_count == len(test_expected)

        # Assert the call arguments to `regex_group_verbs` are as expected
        assert patch_regex_group_verbs.call_args_list == [mocker.call(a) for a in test_expected]

    @pytest.mark.parametrize("test_input, test_expected", args_calls_regex_for_theme_correctly)
    def test_calls_regex_for_theme_correctly(self, mocker, test_input, test_expected):
        """Test extract_phrase_mentions calls regex_for_theme correctly."""

        # Patch the `regex_for_theme` function
        patch_regex_for_theme = mocker.patch(
            "src.make_feedback_tool_data.make_data_for_feedback_tool.regex_for_theme"
        )

        # Call the `extract_phrase_mentions` function; assumes the default grammar file is unchanged
        _ = extract_phrase_mentions(test_input, None)

        # Assert `regex_for_theme` is called the expected number of times
        assert patch_regex_for_theme.call_count == len(test_expected)

        # Assert the call arguments to `regex_for_theme` are as expected
        assert patch_regex_for_theme.call_args_list == [mocker.call(a) for a in test_expected]

    @pytest.mark.parametrize("test_input, test_expected", args_find_needle_called_correctly)
    def test_find_needle_called_correctly(self, mocker, test_input, test_expected):
        """Test extract_phrase_mentions calls the PreProcess.find_needle method corrrectly."""

        # Patch the `PreProcess.find_needle` method
        patch_find_needle = mocker.patch(
            "src.make_feedback_tool_data.make_data_for_feedback_tool.PreProcess.find_needle",
            wraps=PreProcess.find_needle
        )

        # Call the `extract_phrase_mentions` function; assumes the default grammar file is unchanged
        _ = extract_phrase_mentions(test_input, None)

        # Assert that the `PreProcess.find_needle` method is called the correct number of times
        assert patch_find_needle.call_count == len(test_expected)

        # Assert the call arguments for the `PreProcess.find_needle` method are correct
        assert patch_find_needle.call_args_list == [mocker.call(*e) for e in test_expected]


# Define the expected values of the `test_extract_phrase_mentions_returns_correctly` test
args_extract_phrase_mentions_returns_correctly_expected = [
    ([[{"chunked_phrase": ("test to see if", "this example"),
        "exact_phrase": ("test to see if", "this example"),
        "generic_phrase": ("find-smthg", "unknown"),
        "key": ("verb", "noun")}]]),
    ([[{"chunked_phrase": ("to extract", "lemma"),
        "exact_phrase": ("to extract", "lemma"),
        "generic_phrase": ("unknown", "unknown"),
        "key": ("verb", "noun")}]]),
    ([[{"chunked_phrase": ("test to see if", "this example"),
        "exact_phrase": ("test to see if", "this example"),
        "generic_phrase": ("find-smthg", "unknown"),
        "key": ("verb", "noun")},
       {"chunked_phrase": ("to extract", "lemma"),
        "exact_phrase": ("to extract", "lemma"),
        "generic_phrase": ("unknown", "unknown"),
        "key": ("verb", "noun")}]]),
    ([[{"chunked_phrase": ("tried to signed up for", "advice"),
        "exact_phrase": ("tried to signed up for", "advice"),
        "generic_phrase": ("apply-smthg", "information"),
        "key": ("verb", "noun")},
       {"chunked_phrase": ("advice", "due to the ongoing covid 19 outbreak"),
        "exact_phrase": ("advice", "due to the ongoing covid 19 outbreak"),
        "generic_phrase": ("unknown", "covid-mention"),
        "key": ("noun", "prep_noun")},
       {"chunked_phrase": ("due to the ongoing covid 19 outbreak", "with specific concern"),
        "exact_phrase": ("due to the ongoing covid 19 outbreak", "with specific concern"),
        "generic_phrase": ("unknown", "unknown"), "key": ("prep_noun", "prep_noun")},
       {"chunked_phrase": ("with specific concern", "about vulnerable people"),
        "exact_phrase": ("with specific concern", "about vulnerable people"),
        "generic_phrase": ("unknown", "vulnerable"),
        "key": ("prep_noun", "prep_noun")}]])
]

# Define the test cases for the `test_extract_phrase_mentions_returns_correctly` test
args_extract_phrase_mentions_returns_correctly = [
    (i.copy(deep=True), i.copy(deep=True).assign(themed_phrase_mentions=e)) for i, e in zip(
        args_extract_phrase_mentions_integration, args_extract_phrase_mentions_returns_correctly_expected
    )
]

# Define expected outputs for the `test_create_phrase_level_columns_returns_correctly` test
args_create_phrase_level_columns_returns_correctly_expected = [
    ("test to see if, this example", "find-smthg, unknown"),
    ("to extract, lemma", "unknown, unknown"),
    ("test to see if, this example\nto extract, lemma", "find-smthg, unknown\nunknown, unknown"),
    ("tried to signed up for, advice", "apply-smthg, information")
]

# Initialise a storing variable for the `test_create_phrase_level_columns_returns_correctly` test
args_create_phrase_level_columns_returns_correctly = []

# Define the test cases for the `test_create_phrase_level_columns_returns_correctly` test
for i, e in zip(args_extract_phrase_mentions_returns_correctly_expected,
                args_create_phrase_level_columns_returns_correctly_expected):
    args_create_phrase_level_columns_returns_correctly.append((
        pd.DataFrame([i], columns=["themed_phrase_mentions"]),
        pd.DataFrame([i], columns=["themed_phrase_mentions"]).assign(exact_phrases=e[0], generic_phrases=e[1])
    ))

# Define the test cases for the `test_drop_duplicate_rows_returns_correctly` function as dictionaries,
# and then coerce into pandas DataFrames. The first two test cases should be unchanged, the third test case should
# drop one duplicate row, the fourth test case should drop one duplicate row and reset in the index, and the fifth
# test case should drop two rows
args_drop_duplicate_rows_returns_correctly = [
    ({"primary_key": [0, 1, 2], "intents_clientID": ["a", "b", "c"], "session_id": ["A", "B", "C"]},
     {"primary_key": [0, 1, 2], "intents_clientID": ["a", "b", "c"], "session_id": ["A", "B", "C"]}),
    ({"primary_key": [0, 1, 2], "intents_clientID": ["a", "b", "c"], "session_id": ["A", "B", "B"]},
     {"primary_key": [0, 1, 2], "intents_clientID": ["a", "b", "c"], "session_id": ["A", "B", "B"]}),
    ({"primary_key": [0, 1, 1], "intents_clientID": ["a", "b", "c"], "session_id": ["A", "B", "C"]},
     {"primary_key": [0, 1], "intents_clientID": ["a", "b"], "session_id": ["A", "B"]}),
    ({"primary_key": [1, 1, 0], "intents_clientID": ["a", "b", "c"], "session_id": ["A", "B", "C"]},
     {"primary_key": [1, 0], "intents_clientID": ["a", "c"], "session_id": ["A", "C"]}),
    ({"primary_key": [0, 0, 0], "intents_clientID": ["a", "b", "c"], "session_id": ["A", "B", "C"]},
     {"primary_key": [0], "intents_clientID": ["a"], "session_id": ["A"]}),
]
args_drop_duplicate_rows_returns_correctly = [
    tuple(map(pd.DataFrame, a)) for a in args_drop_duplicate_rows_returns_correctly
]

# Define test cases for the `test_function_returns_correctly` test to run similar functions simultaneously
args_function_returns_correctly = [
    *[(drop_duplicate_rows, *a) for a in args_drop_duplicate_rows_returns_correctly],
    *[(extract_phrase_mentions, *a) for a in args_extract_phrase_mentions_returns_correctly],
    *[(create_phrase_level_columns, *a) for a in args_create_phrase_level_columns_returns_correctly]
]


@pytest.mark.parametrize("test_func, test_input, test_expected", args_function_returns_correctly)
def test_function_returns_correctly(test_func, test_input, test_expected):
    """Test a function returns correctly using the default grammar file."""
    assert_frame_equal(test_func(test_input), test_expected)


# TODO: amend test cases for `create_dataset` to also test the regular expressions processing in the function

# Set the Faker seed, and instantiate a Faker class sent to GB domain
Faker.seed(42)
fake = Faker("en_GB")

# Define the number of rows of data to create in the example survey data
example_survey_len = len(args_extract_phrase_mentions_inputs_q3_edit) + 1

# Create some example data for a few columns that will be manipulated by the `create_dataset` function; note all
# entries here are randomly generated, and do not represent real data except in formatting
EXAMPLE_SURVEY_DICT = {
    "primary_key": list(range(example_survey_len)),
    "intents_clientID": random.sample(range(1000000000), example_survey_len),
    "visitId": random.sample(range(1000000000), example_survey_len),
    "fullVisitorId": random.sample(range(1000000000000000000), example_survey_len),
    "hits_pagePath": [f"/{fake.slug()}" for _ in range(example_survey_len)],
    "Started": [fake.date_time_this_month() for _ in range(example_survey_len)],
    "Ended": None,
    "Q1": random.choices(["Personal", "Professional", "-"], k=example_survey_len),
    "Q2": [fake.job() for _ in range(example_survey_len)],
    "Q3": random.sample([*args_extract_phrase_mentions_inputs_q3_edit, "-"], example_survey_len),
    "Q4": random.choices(["Yes", "Not sure / Not yet", "No", "-"], k=example_survey_len),
    "Q5": random.choices(["Very satisfied", "Satisfied", "Neither satisfied nor dissatisfied", "Dissatisfied",
                          "Not at all satisfied", "-"], k=example_survey_len),
    "Q6": random.choices(["Yes", "No", "-"], k=example_survey_len),
    "Q7": random.choices(["-"] + [fake.sentence() for _ in range(9)], weights=[55] + [5] * 9, k=example_survey_len),
    "Q8": random.choices(["-"] + [fake.paragraph() for _ in range(9)], weights=[55] + [5] * 9, k=example_survey_len),
    "session_id": None,
    "dayofweek": None,
    "isWeekend": None,
    "hour": None,
    "country": random.choices([fake.local_latlng("GB") for _ in range(80)] + [fake.local_latlng() for _ in range(20)],
                              k=example_survey_len),
    "country_grouping": None,
    "UK_region": None,
    "UK_metro_area": None,
    "channelGrouping": random.choices(["(Other)", "Direct", "Display", "Email", "Organic Search", "Paid Search",
                                       "Referral", "Social", None], k=example_survey_len),
    "deviceCategory": random.choices(["desktop", "mobile", None, "tablet"], k=example_survey_len),
    "total_seconds_in_session_across_days": random.choices([None, *range(1000)], k=example_survey_len),
    "total_pageviews_in_session_across_days": random.choices([None, *range(1000)], k=example_survey_len),
    "finding_count": random.choices(range(100), k=example_survey_len),
    "updates_and_alerts_count": random.choices(range(100), k=example_survey_len),
    "news_count": random.choices(range(100), k=example_survey_len),
    "decisions_count": random.choices(range(100), k=example_survey_len),
    "speeches_and_statements_count": random.choices(range(100), k=example_survey_len),
    "transactions_count": random.choices(range(100), k=example_survey_len),
    "regulation_count": random.choices(range(100), k=example_survey_len),
    "guidance_count": random.choices(range(100), k=example_survey_len),
    "business_support_count": random.choices(range(100), k=example_survey_len),
    "policy_count": random.choices(range(100), k=example_survey_len),
    "consultations_count": random.choices(range(100), k=example_survey_len),
    "research_count": random.choices(range(100), k=example_survey_len),
    "statistics_count": random.choices(range(100), k=example_survey_len),
    "transparency_data_count": random.choices(range(100), k=example_survey_len),
    "freedom_of_information_releases_count": random.choices(range(100), k=example_survey_len),
    "incidents_count": random.choices(range(100), k=example_survey_len),
    "done_page_flag": random.choices([0, 1], k=example_survey_len),
    "count_client_error": random.choices(range(100), k=example_survey_len),
    "count_server_error": random.choices(range(100), k=example_survey_len),
    "ga_visit_start_timestamp": None,
    "ga_visit_end_timestamp": None,
    "intents_started_date": None,
    "events_sequence": random.choices(["-"] + [fake.sentence() for _ in range(9)], weights=[55] + [5] * 9,
                                      k=example_survey_len),
    "search_terms_sequence": None,
    "cleaned_search_terms_sequence": random.choices(["-"] + [fake.sentence() for _ in range(9)], weights=[55] + [5] * 9,
                                                    k=example_survey_len),
    "top_level_taxons_sequence": [">>".join(random.choices([""] + [fake.sentence() for _ in range(9)],
                                                           weights=[55] + [5] * 9, k=random.choice(range(1, 20))))
                                  for _ in range(example_survey_len)],
    "page_format_sequence": [">>".join(random.choices([""] + [fake.sentence() for _ in range(9)],
                                                      weights=[55] + [5] * 9, k=random.choice(range(1, 20))))
                             for _ in range(example_survey_len)],
    "Sequence": random.choices(["-"] + [fake.sentence() for _ in range(9)], weights=[55] + [5] * 9,
                               k=example_survey_len),
    "PageSequence": [">>".join(random.choices([""] + [fake.sentence() for _ in range(9)],
                                              weights=[55] + [5] * 9, k=random.choice(range(1, 20))))
                     for _ in range(example_survey_len)],
    "flag_for_criteria": random.choices([None, 0, 1], k=example_survey_len),
    "full_url_in_session_flag": random.choices([0, 1], k=example_survey_len),
    "UserID": random.sample(range(1000000000), example_survey_len),
    "UserNo": random.sample(range(10000), example_survey_len),
    "Name": [fake.name() for _ in range(example_survey_len)],
    "Email": [fake.ascii_email() for _ in range(example_survey_len)],
    "IP Address": [fake.ipv4() for _ in range(example_survey_len)],
    "Unique ID": [np.nan for _ in range(example_survey_len)],
    "Tracking Link": random.choices(["Default Web Link", "GOV UK footer - email", "GOV UK footer - no email"],
                                    k=example_survey_len),
    "clientID": random.sample(range(1000000000), example_survey_len),
    "Page Path": [f"/{fake.slug()}" for _ in range(example_survey_len)],
    "Q1_y": None,
    "Q2_y": None,
    "Q3_y": None,
    "Q4_y": None,
    "Q5_y": None,
    "Q6_y": None,
    "Q7_y": None,
    "Q8_y": None,
    "Started_Date": None,
    "Ended_Date": None,
    "Started_Date_sub_12h": None
}

# Update various None entries in `EXAMPLE_SURVEY_DICT`
EXAMPLE_SURVEY_DICT.update({
    "Ended": [dt + timedelta(minutes=random.randint(10, 120)) for dt in EXAMPLE_SURVEY_DICT["Started"]],
    "session_id": [f"{fv}-{v}" for fv, v in zip(EXAMPLE_SURVEY_DICT["fullVisitorId"], EXAMPLE_SURVEY_DICT["visitId"])],
    "dayofweek": [dt.weekday() for dt in EXAMPLE_SURVEY_DICT["Started"]],
    "isWeekend": [int(dt.weekday() is None or dt.weekday() > 5) for dt in EXAMPLE_SURVEY_DICT["Started"]],
    "hour": [dt.hour for dt in EXAMPLE_SURVEY_DICT["Started"]],
    "country_grouping": ["UK" if c[3] == "GB" else random.choice(["EU_EEA_Swiss", "Other"]) for c in
                         EXAMPLE_SURVEY_DICT["country"]],
    "UK_region": [random.choice(["England", "Scotland", "Wales", "Northern Ireland"]) if c[3] == "GB" else
                  random.choice(["not UK", "(not set)"]) for c in EXAMPLE_SURVEY_DICT["country"]],
    "UK_metro_area": [c[2] if c[3] == "GB" else random.choice(["not UK", "(not set)"]) for c in
                      EXAMPLE_SURVEY_DICT["country"]],
    "ga_visit_start_timestamp": [dt - timedelta(minutes=random.randint(60, 120)) for dt in
                                 EXAMPLE_SURVEY_DICT["Started"]],
    "ga_visit_end_timestamp": [dt - timedelta(minutes=random.randint(5, 59)) for dt in EXAMPLE_SURVEY_DICT["Started"]],
    "intents_started_date": [int(dt.strftime("%Y%m%d")) for dt in EXAMPLE_SURVEY_DICT["Started"]],
    "search_terms_sequence": ["+".join(s) for s in EXAMPLE_SURVEY_DICT["cleaned_search_terms_sequence"]],
    "Q1_y": EXAMPLE_SURVEY_DICT["Q1"],
    "Q2_y": EXAMPLE_SURVEY_DICT["Q2"],
    "Q3_y": EXAMPLE_SURVEY_DICT["Q3"],
    "Q4_y": EXAMPLE_SURVEY_DICT["Q4"],
    "Q5_y": EXAMPLE_SURVEY_DICT["Q5"],
    "Q6_y": EXAMPLE_SURVEY_DICT["Q6"],
    "Q7_y": EXAMPLE_SURVEY_DICT["Q7"],
    "Q8_y": EXAMPLE_SURVEY_DICT["Q8"],
    "Started_Date": [int(dt.strftime("%Y%m%d")) for dt in EXAMPLE_SURVEY_DICT["Started"]],
    "Started_Date_sub_12h": [int((dt - timedelta(hours=12)).strftime("%Y%m%d"))
                             for dt in EXAMPLE_SURVEY_DICT["Started"]]
})
EXAMPLE_SURVEY_DICT.update({
    "country": [c[3] for c in EXAMPLE_SURVEY_DICT["country"]],
    "Ended_Date": [int(dt.strftime("%Y%m%d")) for dt in EXAMPLE_SURVEY_DICT["Ended"]]
})

# Define some additional dummy columns that will be dropped by the `create_dataset` function
COLS_DROPPED = ["test_col_a", "test_col_b", "test_col_b"]

# Add some dummy data that will be dropped to `EXAMPLE_SURVEY_DICT`
for cols_dummy in COLS_DROPPED:
    EXAMPLE_SURVEY_DICT[cols_dummy] = random.sample(range(1000000), example_survey_len)

# Create a pandas DataFrame from `EXAMPLE_SURVEY_DICT`, and change all datetime columns to objects
EXAMPLE_SURVEY_DF = pd.DataFrame.from_dict(EXAMPLE_SURVEY_DICT)
EXAMPLE_SURVEY_DF = EXAMPLE_SURVEY_DF.assign(
    **{c: EXAMPLE_SURVEY_DF[c].replace(r"^\s*$", np.nan, regex=True) for c in EXAMPLE_SURVEY_DF.select_dtypes(
        include="object").columns},
    **{c: EXAMPLE_SURVEY_DF[c].dt.strftime("%Y-%m-%d %H:%M:%S") for c in EXAMPLE_SURVEY_DF.select_dtypes(
        include="datetime").columns}
)

# Add a random duplicate row into `EXAMPLE_SURVEY_DF`, then sort the index, and reset it
EXAMPLE_SURVEY_DF = EXAMPLE_SURVEY_DF.append(EXAMPLE_SURVEY_DF.iloc[random.randrange(0, example_survey_len)]) \
    .sort_index() \
    .reset_index(drop=True)

# Define the changes made to `EXAMPLE_SURVEY_DF` post-execution of the `preprocess_filter_comment_text` function in
# `create_dataset`
EXAMPLE_SURVEY_POST_PREPROCESS_DF = EXAMPLE_SURVEY_DF \
    .assign(Q3_pii_removed=EXAMPLE_SURVEY_DF["Q3"],
            language=["en" if v != "-" else "-" for v in EXAMPLE_SURVEY_DF["Q3"].values],
            is_en=True) \
    .drop_duplicates(subset=["primary_key"]) \
    .reset_index(drop=True)


# Define the expected output pandas DataFrame from `create_dataset` function
EXAMPLE_SURVEY_DF_OUTPUT = EXAMPLE_SURVEY_DF.assign(
    exact_phrases=EXAMPLE_SURVEY_DF["Q3"].map(dict(zip(
        args_extract_phrase_mentions_inputs_q3_edit,
        ["test to see if, this example", "to extract, lemma", "test to see if, this example\nto extract, lemma",
         "tried to signed up for, advice"]
    ))),
    generic_phrases=EXAMPLE_SURVEY_DF["Q3"].map(dict(zip(
        args_extract_phrase_mentions_inputs_q3_edit,
        ["find-smthg, unknown", "unknown, unknown", "find-smthg, unknown\nunknown, unknown", "apply-smthg, information"]
    )))
).drop_duplicates(subset=["primary_key"]) \
    .reset_index(drop=True)


@pytest.fixture
def temp_survey_file(temp_folder, temp_survey_filename):
    """Create a test survey file within a temporary folder."""

    # Create a filename called `temp_survey_filename`
    temp_filepath = temp_folder.join(temp_survey_filename)

    # Write a copy `EXAMPLE_SURVEY_DF` to this file
    EXAMPLE_SURVEY_DF.copy(deep=True).to_csv(temp_filepath, index=False)

    # Return the file path to the temporary survey file
    return temp_filepath


@pytest.fixture
def temp_cache_pos_file(temp_folder, temp_cache_pos_filename):
    """Create a file path to for the cached part-of-speech (POS) file."""
    return temp_folder.join(temp_cache_pos_filename)


@pytest.fixture
def temp_output_file(temp_folder, temp_output_filename):
    """Create a file path to for the output file."""
    return temp_folder.join(temp_output_filename)


@pytest.fixture
def patch_save_intermediate_df(mocker):
    """Patch the save_intermediate_df function."""
    return mocker.patch("src.make_feedback_tool_data.make_data_for_feedback_tool.save_intermediate_df")


@pytest.fixture
def resource_create_dataset_integration(mocker, temp_survey_file, temp_cache_pos_file, temp_output_file,
                                        patch_save_intermediate_df, patch_pandas_dataframe_to_csv):
    """Resource for the `TestCreateDataset` test for the entire create_dataset function."""

    # Patch the `pandas.read_csv` method, and wrap `pandas.read_csv` as well
    patch_pandas_read_csv = mocker.patch("pandas.read_csv", wraps=pd.read_csv)

    # Define a list of function names from `src.make_feedback_tool_data.make_data_for_feedback_tool` that need to be
    # patched
    patch_function_names = ["drop_duplicate_rows", "preprocess_filter_comment_text", "PreProcess.part_of_speech_tag",
                            "extract_phrase_mentions", "create_phrase_level_columns"]

    # Initialise a storing variable
    patch_dict = {}

    # Patch the functions listed in `patch_function_names`; the keys will be lowercase entries of
    # `patch_function_names` with any periods replaced with underscores
    for n in patch_function_names:
        patch_function = mocker.patch(f"src.make_feedback_tool_data.make_data_for_feedback_tool.{n}")
        patch_dict[f"patch_{n.replace('.', '_').lower()}"] = patch_function

    # Return a dictionary containing all necessary patches for tests in the `TestCreateDataset` test class
    return {"temp_survey_file": temp_survey_file, "temp_cache_pos_file": temp_cache_pos_file,
            "temp_output_file": temp_output_file, "patch_pandas_read_csv": patch_pandas_read_csv,
            "patch_save_intermediate_df": patch_save_intermediate_df,
            "patch_pandas_dataframe_to_csv": patch_pandas_dataframe_to_csv, **patch_dict}


# Define test file names for the `TestCreateDataset` test class
args_create_dataset_integration_filenames = [
    ("hello.csv", "world.csv", "hello_world.csv"),
    ("foo.csv", "bar.csv", "foobar.csv")
]


@pytest.mark.parametrize("temp_survey_filename, temp_cache_pos_filename, temp_output_filename",
                         args_create_dataset_integration_filenames)
class TestCreateDataset:

    def test_pandas_read_csv_called_once_correctly(self, resource_create_dataset_integration):
        """Test pandas.read_csv method is called once by create_dataset correctly."""

        # Call the `create_dataset` function using the default grammar file
        create_dataset(resource_create_dataset_integration["temp_survey_file"], None,
                       resource_create_dataset_integration["temp_cache_pos_file"],
                       resource_create_dataset_integration["temp_output_file"])

        # Assert `pandas.read_csv` is called once with the correct arguments
        resource_create_dataset_integration["patch_pandas_read_csv"].assert_called_once_with(
            resource_create_dataset_integration["temp_survey_file"]
        )

    def test_drop_duplicate_rows_called_once_correctly(self, resource_create_dataset_integration):
        """Test drop_duplicate_rows is called once by create_dataset correctly."""

        # Call the `create_dataset` function using the default grammar file
        create_dataset(resource_create_dataset_integration["temp_survey_file"], None,
                       resource_create_dataset_integration["temp_cache_pos_file"],
                       resource_create_dataset_integration["temp_output_file"])

        # Define the function patch under test
        test_function_patch = resource_create_dataset_integration["patch_drop_duplicate_rows"]

        # Assert `test_function_patch` is called only once
        test_function_patch.assert_called_once()

        # Get the call arguments from the first and only call
        test_output_args, test_output_kwargs = test_function_patch.call_args_list[0]

        # Assert there is only one argument, and no keyword arguments
        assert len(test_output_args) == 1
        assert not test_output_kwargs

        # Assert the argument is as expected
        assert_frame_equal(test_output_args[0], EXAMPLE_SURVEY_DF)

    def test_preprocess_filter_comment_text_called_once_correctly(self, resource_create_dataset_integration):
        """Test preprocess_filter_comment_text is called once by create_dataset correctly."""

        # Call the `create_dataset` function using the default grammar file
        create_dataset(resource_create_dataset_integration["temp_survey_file"], None,
                       resource_create_dataset_integration["temp_cache_pos_file"],
                       resource_create_dataset_integration["temp_output_file"])

        # Assert `preprocess_filter_comment_text` is called once correctly
        resource_create_dataset_integration["patch_preprocess_filter_comment_text"].assert_called_once_with(
            resource_create_dataset_integration["patch_drop_duplicate_rows"].return_value
        )

    def test_preprocess_part_of_speech_tag_called_correctly(self, mocker, resource_create_dataset_integration):
        """Test PreProcess.part_of_speech_tag method is called by create_dataset correctly."""

        # Set the return value of the `preprocess_filter_comment_text` patch
        resource_create_dataset_integration["patch_preprocess_filter_comment_text"].return_value = \
            preprocess_filter_comment_text(drop_duplicate_rows(EXAMPLE_SURVEY_DF.copy(deep=True)))

        # Call the `create_dataset` function using the default grammar file
        create_dataset(resource_create_dataset_integration["temp_survey_file"], None,
                       resource_create_dataset_integration["temp_cache_pos_file"],
                       resource_create_dataset_integration["temp_output_file"])

        # Define the function patch under test
        test_function_patch = resource_create_dataset_integration["patch_preprocess_part_of_speech_tag"]

        # Assert that the `PreProcess.part_of_speech_tag` method was called the correct number of times
        assert test_function_patch.call_count == len(EXAMPLE_SURVEY_POST_PREPROCESS_DF.query("is_en"))

        # Define the expected call arguments of the `PreProcess.part_of_speech_tag` method
        test_expected_call_args_list = [
            mocker.call(v) for v in EXAMPLE_SURVEY_POST_PREPROCESS_DF.query("is_en")["Q3_pii_removed"].values
        ]

        # Assert that the `PreProcess.part_of_speech_tag` method was called correctly
        assert test_function_patch.call_args_list == test_expected_call_args_list

    def test_extract_phrase_mentions_called_once_correctly(self, resource_create_dataset_integration):
        """Test extract_phrase_mentions is called once by create_dataset correctly."""

        # Define how the pandas DataFrame should look after execution of the `preprocess_filter_comment_text`
        # function, and set this as the return value of the `preprocess_filter_comment_text` patch
        test_partial_output = preprocess_filter_comment_text(drop_duplicate_rows(EXAMPLE_SURVEY_DF.copy(deep=True)))
        resource_create_dataset_integration["patch_preprocess_filter_comment_text"].return_value = test_partial_output

        # Set a side effect of the `PreProcess.part_of_speech_tag` method
        resource_create_dataset_integration["patch_preprocess_part_of_speech_tag"].side_effect = lambda x: x

        # Call the `create_dataset` function using the default grammar file
        create_dataset(resource_create_dataset_integration["temp_survey_file"], None,
                       resource_create_dataset_integration["temp_cache_pos_file"],
                       resource_create_dataset_integration["temp_output_file"])

        # Define the function patch under test
        test_function_patch = resource_create_dataset_integration["patch_extract_phrase_mentions"]

        # Assert that the `extract_phrase_mentions` was called once
        test_function_patch.assert_called_once()

        # Get the actual call arguments of the first, and only call to `extract_phrase_mentions`
        test_output_args, test_output_kwargs = test_function_patch.call_args_list[0]

        # Assert that there is only two arguments, and no keyword arguments
        assert len(test_output_args) == 2
        assert not test_output_kwargs

        # Define the expected column `Q3_edit` of the first call argument of the `extract_phrase_mentions` function
        test_expected_q3_edit = test_partial_output["Q3"].replace(np.nan, "") \
            .map(lambda x: " ".join(re.sub(r"[()\[\]+*]", "", x).split()))

        # Define the expected first call argument of the `extract_phrase_mentions` function
        test_expected_arg1 = test_partial_output.assign(
            pos_tag=test_partial_output["Q3_pii_removed"].where(test_partial_output["is_en"], []),
            Q3_edit=test_expected_q3_edit
        )

        # Assert the first argument is as expected, and the second argument is None, as we are using the default
        # grammar file
        assert_frame_equal(test_output_args[0], test_expected_arg1)
        assert test_output_args[1] is None

    def test_save_intermediate_df_called_once_correctly(self, resource_create_dataset_integration):
        """Test save_intermediate_df is called once by create_dataset correctly."""

        # Call the `create_dataset` function using the default grammar file
        create_dataset(resource_create_dataset_integration["temp_survey_file"], None,
                       resource_create_dataset_integration["temp_cache_pos_file"],
                       resource_create_dataset_integration["temp_output_file"])

        # Assert `save_intermediate_df` is called once with the correct arguments
        resource_create_dataset_integration["patch_save_intermediate_df"].assert_called_once_with(
            resource_create_dataset_integration["patch_extract_phrase_mentions"].return_value,
            resource_create_dataset_integration["temp_cache_pos_file"]
        )

    def test_create_phrase_level_columns_called_once_correctly(self, resource_create_dataset_integration):
        """Test create_phrase_level_columns is called once by create_dataset correctly."""

        # Call the `create_dataset` function using the default grammar file
        create_dataset(resource_create_dataset_integration["temp_survey_file"], None,
                       resource_create_dataset_integration["temp_cache_pos_file"],
                       resource_create_dataset_integration["temp_output_file"])

        # Assert `create_phrase_level_columns` is called once with the correct arguments
        resource_create_dataset_integration["patch_create_phrase_level_columns"].assert_called_once_with(
            resource_create_dataset_integration["patch_extract_phrase_mentions"].return_value
        )

    def test_pandas_dataframe_to_csv_called_once_correctly(self, temp_survey_file, temp_cache_pos_file,
                                                           temp_output_file, patch_save_intermediate_df,
                                                           patch_pandas_dataframe_to_csv):
        """Test pandas.DataFrame.to_csv method is called once by create_dataset correctly."""

        # Call the `create_dataset` function using the default grammar file
        create_dataset(temp_survey_file, None, temp_cache_pos_file, temp_output_file)

        # Assert that the `pandas.DataFrame.to_csv` method is called once with the correct arguments
        patch_pandas_dataframe_to_csv.assert_called_once_with(temp_output_file, index=False)

    def test_create_dataset_returns_correctly(self, temp_survey_file, temp_cache_pos_file, temp_output_file):
        """Test that the create_dataset function returns the correct output."""

        # Call the `create_dataset` function using the default grammar file
        create_dataset(temp_survey_file, None, temp_cache_pos_file, temp_output_file)

        # Get the actual CSV output from `create_dataset`, and assert it is as expected
        assert_frame_equal(pd.read_csv(temp_output_file), EXAMPLE_SURVEY_DF_OUTPUT)
