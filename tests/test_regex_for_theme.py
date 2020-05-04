from src.make_feedback_tool_data.regex_for_theme import regex_for_theme
import os
import pytest
import yaml

# Get the folder path to the `data` folder, and the name of the expected YAML file
DIR_DATA = os.environ.get("DIR_DATA")
FILE_REGEX_THEME = "regex_for_theme.yaml"

# Load the theme regular expression file
with open(os.path.join(DIR_DATA, FILE_REGEX_THEME), "r") as rf:
    DICT_REGEX = yaml.safe_load(rf)

# Define an example regular expression dictionary, with overlapping patterns
example_dict_regex = {
    "foo": r"(f)[o]{2,}bar",
    "world": r"(hell)o?",
    "bar": r"(f)[o]{2,}"
}

args_test_theme_identification = [
    ("stats", None, "data"),
    ("stats", {}, "data"),
    ("stats", DICT_REGEX, "data"),
    ("foobar", example_dict_regex, "foo"),
    ("hello", example_dict_regex, "world"),
    ("foo", example_dict_regex, "bar")
]


@pytest.mark.parametrize("test_input_text, test_input_dict_regex, test_expected_theme", args_test_theme_identification)
def test_theme_identification(test_input_text, test_input_dict_regex, test_expected_theme):

    # Get the theme for `test_input_text`, based on `test_input_dict_regex`
    test_output_theme = regex_for_theme(test_input_text, test_input_dict_regex)

    # Check `test_output_theme` is as expected
    assert test_output_theme == test_expected_theme
