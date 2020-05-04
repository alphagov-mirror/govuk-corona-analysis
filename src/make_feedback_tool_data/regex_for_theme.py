from typing import Dict, Optional
import os
import re
import yaml

# Get the folder path to the `data` folder, and the name of the expected YAML file
DIR_DATA = os.environ.get("DIR_DATA")
FILE_REGEX_THEME = "regex_for_theme.yaml"

# Load the theme regular expression file
with open(os.path.join(DIR_DATA, FILE_REGEX_THEME), "r") as rf:
    DICT_REGEX = yaml.safe_load(rf)


def regex_for_theme(text: str, dict_regex: Optional[Dict[str, str]] = None) -> str:
    """Use regular expressions to determine the theme of a text string.

    Args:
        text:       A text string for theme identification.
        dict_regex: Default: None. A dictionary where the keys are possible themes, and the values are regular
            expression patterns that match said theme. If set to None, the function will use `regex_for_theme.yaml`
            from the `data` folder.

    Returns:
        A string of the theme from the keys of `dict_regex`, or 'unknown' if no matching themes exist.

    """

    # If `dict_regex` is None or empty, use the predefined file
    if not dict_regex:
        dict_regex = DICT_REGEX

    # Iterate through `dict_regex` in key order, and search for the regular expression pattern in `text`; return the
    # key, i.e. the theme, if there is a match
    for k, v in dict_regex.items():
        if re.search(v, text, re.IGNORECASE):
            return k

    # If there are no matches with any theme keys in `dict_regex`, return 'unknown'
    return "unknown"
