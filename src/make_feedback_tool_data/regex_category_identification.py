from typing import Dict, Optional
import os
import re
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


def regex_category_identification(text: str, dict_category: Dict[str, str]) -> str:
    """Use regular expressions to determine a category of a text string.

    Args:
        text:           A text string for category identification; the text case is ignored.
        dict_category:  A dictionary where the keys are possible categories, and the values are regular expression
            patterns that match said category.

    Returns:
        A string of the category from the keys of `dict_category`, or 'unknown' if no matching categories exist.

    """

    # Iterate through `dict_category` in key order, and search for the regular expression pattern in `text`; return the
    # key, i.e. the category, if there is a match
    for k, v in dict_category.items():
        if re.search(v, text, re.IGNORECASE):
            return k

    # If there are no matches with any theme keys in `dict_category`, return 'unknown'
    return "unknown"


def regex_for_theme(text: str, dict_themes: Optional[Dict[str, str]] = None) -> str:
    """Use regular expressions to determine the theme of a text string.

    Args:
        text:           A text string for theme identification; the text case is ignored.
        dict_themes:    A dictionary where the keys are possible themes, and the values are regular expression patterns
            that match said theme.

    Returns:
        A string of the theme from the keys of `dict_theme`, or 'unknown' if no matching themes exist.

    """
    return regex_category_identification(text, dict_themes if dict_themes else DICT_THEMES)


def regex_group_verbs(text: str, dict_group_verbs: Optional[Dict[str, str]] = None) -> str:
    """Use regular expressions to determine the verb grouping of a text string.

    Args:
        text:               A text string for theme identification; the text case is ignored.
        dict_group_verbs:   A dictionary where the keys are possible verb groupings, and the values are regular
            expression patterns that match said verb grouping.

    Returns:
        A string of the verb grouping from the keys of `dict_group_verbs`, or 'unknown' if no matching verb groupings
        exist.

    """
    return regex_category_identification(text, dict_group_verbs if dict_group_verbs else DICT_GROUP_VERBS)
