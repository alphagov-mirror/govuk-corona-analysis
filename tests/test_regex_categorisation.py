from src.make_feedback_tool_data.regex_categorisation import (
    regex_category_identification,
    regex_group_verbs,
    regex_for_theme
)
import os
import pytest
import yaml

# Get the folder path to the `data` folder, and the name of the expected YAML files
DIR_DATA = os.environ.get("DIR_DATA")
FILE_REGEX_THEMES = "regex_for_theme.yaml"
FILE_REGEX_GROUP_VERBS = "regex_group_verbs.yaml"

# Load the theme regular expression file
with open(os.path.join(DIR_DATA, FILE_REGEX_THEMES), "r") as tf:
    DICT_THEMES = yaml.safe_load(tf)

# Load the group verbs regular expression file
with open(os.path.join(DIR_DATA, FILE_REGEX_GROUP_VERBS), "r") as gvf:
    DICT_GROUP_VERBS = yaml.safe_load(gvf)

# Define an example regular expression dictionary, with overlapping patterns
example_dict_regex = {
    "foo": r"(f)[o]{2,}bar",
    "world": r"(hell)o?",
    "bar": r"(f)[o]{2,}"
}

# Define test cases for the `test_category_identification` unit test
args_test_category_identification = [
    ("foobar", example_dict_regex, "foo"),
    ("hello", example_dict_regex, "world"),
    ("foo", example_dict_regex, "bar"),
    ("random", example_dict_regex, "unknown"),
    ("stats", DICT_THEMES, "data"),
    ("I need to pay bills", DICT_GROUP_VERBS, "pay-smthg")
]


# Define test cases for the `TestThemeIdentification` test class
args_test_theme_identification = [
    *args_test_category_identification,
    ("stats", None, "data"),
    ("stats", {}, "data"),
    ("random", None, "unknown")
]

# Define test cases for the `TestGroupVerbsIdentification` test class
args_test_group_verbs_identification = [
    *args_test_category_identification,
    ("I need to pay bills", None, "pay-smthg"),
    ("I need to pay bills", {}, "pay-smthg"),
    ("random", None, "unknown")
]


@pytest.mark.parametrize("test_input_text, test_input_dict_category, test_expected_category",
                         args_test_category_identification)
def test_category_identification(test_input_text, test_input_dict_category, test_expected_category):
    """Test regex_category_identification returns the expected category."""

    # Get the category for `test_input_text`, based on `test_input_dict_category`
    test_output_category = regex_category_identification(test_input_text, test_input_dict_category)

    # Check `test_output_category` is as expected
    assert test_output_category == test_expected_category


@pytest.mark.parametrize("test_input_text, test_input_dict_theme, test_expected_theme",
                         args_test_theme_identification)
class TestThemeIdentification:

    def test_returns_correctly(self, test_input_text, test_input_dict_theme, test_expected_theme):
        """Test regex_for_theme returns the expected theme, even if test_input_dict_regex is None."""

        # Get the theme for `test_input_text`, based on `test_input_dict_theme`
        test_output_theme = regex_for_theme(test_input_text, test_input_dict_theme)

        # Check `test_output_theme` is as expected
        assert test_output_theme == test_expected_theme

    def test_calls_regex_category_identification_correctly(self, mocker, test_input_text, test_input_dict_theme,
                                                           test_expected_theme):
        """Test regex_for_theme calls regex_category_identification correctly."""

        # Patch the `regex_category_identification` function
        patch_regex_category_identification = mocker.patch(
            "src.make_feedback_tool_data.regex_categorisation.regex_category_identification"
        )

        # Call `regex_for_theme`
        _ = regex_for_theme(test_input_text, test_input_dict_theme)

        # Assert the `regex_category_identification` is called once with the correct arguments
        patch_regex_category_identification.assert_called_once_with(
            test_input_text, test_input_dict_theme if test_input_dict_theme else DICT_THEMES
        )


@pytest.mark.parametrize("test_input_text, test_input_dict_group_verb, test_expected_group_verb",
                         args_test_group_verbs_identification)
class TestGroupVerbsIdentification:

    def test_returns_correctly(self, test_input_text, test_input_dict_group_verb, test_expected_group_verb):
        """Test regex_group_verbs returns the expected verb grouping, even if test_input_dict_regex is None."""

        # Get the verb groupings for `test_input_text`, based on `test_input_dict_group_verb`
        test_output_group_verb = regex_group_verbs(test_input_text, test_input_dict_group_verb)

        # Check `test_output_group_verb` is as expected
        assert test_output_group_verb == test_expected_group_verb

    def test_calls_regex_category_identification_correctly(self, mocker, test_input_text, test_input_dict_group_verb,
                                                           test_expected_group_verb):
        """Test regex_group_verbs calls regex_category_identification correctly."""

        # Patch the `regex_category_identification` function
        patch_regex_category_identification = mocker.patch(
            "src.make_feedback_tool_data.regex_categorisation.regex_category_identification"
        )

        # Call `regex_group_verbs`
        _ = regex_group_verbs(test_input_text, test_input_dict_group_verb)

        # Assert the `regex_category_identification` is called once with the correct arguments
        patch_regex_category_identification.assert_called_once_with(
            test_input_text, test_input_dict_group_verb if test_input_dict_group_verb else DICT_GROUP_VERBS
        )
