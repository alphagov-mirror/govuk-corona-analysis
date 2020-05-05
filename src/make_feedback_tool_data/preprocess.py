from difflib import SequenceMatcher as SM
from nltk import sent_tokenize
from nltk.util import ngrams
from tqdm import tqdm
from typing import Any, Dict, List, Optional, Tuple
import re
import regex
import spacy
import sys

# Try/except block to flag instructions for installing PyICU if the module has not been installed
try:
    from polyglot.detect import Detector
except ImportError as e:
    if str(e) == "No module named 'icu'":
        raise ImportError(f"{str(e)}. Follow Python package instructions in `README.md` to install PyICU!")
    else:
        raise e

tqdm.pandas()

# Load spaCy's pre-trained statistical models for English 'en_core_web_sm'
NLP = spacy.load("en_core_web_sm")

# Define the regular expressions used for stripping out personally identifiable information (PII)
PII_FILTERED = ["DATE_OF_BIRTH", "EMAIL_ADDRESS", "PASSPORT", "PERSON_NAME", "PHONE_NUMBER", "STREET_ADDRESS",
                "UK_NATIONAL_INSURANCE_NUMBER", "UK_PASSPORT"]
PII_REGEX = "|".join([rf"\[{p}\]" for p in PII_FILTERED])


class PreProcess:
    """A class to hold static and class methods to pre-process text data."""

    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """Split a multi-sentence text string into list of sentences.

        :param text: Text string for splitting.
        :return: List of individual sentences from `text`.

        """
        return sent_tokenize(text)

    @staticmethod
    def replace_pii_regex(text: str) -> str:
        """Remove Personally Identifiable Information (PII) from a text string using regular expressions.

        :param text: Text string potentially containing PII.
        :return: `text` with PII removed.

        """
        return re.sub(PII_REGEX, "", text)

    @classmethod
    def part_of_speech_tag(cls, text: str) -> List[List[Tuple[str, str, str]]]:
        """Perform part-of-speech (POS) tagging on a text string.

        Leverages spaCy's pre-trained statistical models for English 'en_core_web_sm'.

        :param text: A text string for POS tagging.
        :return: A nested list of lists, where each nested list represents a sentence of `text`, and contains the POS
            tags of each token in this sentence. Each POS tag is represented as a three-element tuple of the token,
            its POS tag, and its lemma (base word of the token).

        """

        # Split `text` into a list of its sentences
        sentences = cls.split_sentences(text)

        # Return the POS tags for each token in the sentence
        return [[(token.text, token.tag_, token.lemma_) for token in NLP(sentence)] for sentence in sentences]

    @staticmethod
    def detect_language(text: str) -> str:
        """Identify the language of a text string.

        Uses the `polyglot` package. If multiple languages are identified, returns only the most confident/prevalent
        language in `text`. Text strings of '-' are returns as is, with no language identification performed.

        :param text: A text string of one or more languages for identification.
        :return: A text string of the most confident/prevalent language detected in `text`, a '-' (if `text` is '-'),
            or an error string if a language could not be identified.

        """

        # Check if `text` is '-'; if not, try and identify language, otherwise return '-'
        if text != "-":

            # Detect the language of `text`, and return the most confident/prevalent language, if `text` contains
            # multiple languages. If language detection fails, return an error string
            try:
                langs = {language.confidence: language.code for language in Detector(text, quiet=True).languages}
                return langs[max(langs.keys())]
            except Exception:
                return f"[ERROR] {text} {sys.exc_info()}"
        else:
            return "-"

    @staticmethod
    def compute_combinations(items: List[List[Any]], n: int) -> List[List[Any]]:
        """Create list chunks from a nested list of items using a moving window of a set size n.

        See Example for further details.

        :param items: A nested list of items for chunking.
        :param n: The size of each resultant chunk.
        :return: A list of chunks, where each chunk is a list of sentences.

        :example:

        >>> from src.make_feedback_tool_data.preprocess import PreProcess
        >>> PreProcess.compute_combinations([["A", "B", "C", "D"]], 1)
        [['A'], ['B'], ['C'], ['D']]

        >>> PreProcess.compute_combinations([("A", "B", "C", "D")], 2)
        [('A', 'B'), ('B', 'C'), ('C', 'D')]

        >>> PreProcess.compute_combinations([("A", "B", "C", "D")], 3)
        [('A', 'B', 'C'), ('B', 'C', 'D')]

        """
        return [chunks[i:i + n] for chunks in items for i in range(len(chunks) - (n - 1))]

    @staticmethod
    def get_user_group(verb: str, text: str) -> str:
        """Extract user group based on text, and its verb usage using regular expressions.

        Verb must satisfy this regular expression: ((('|’|^(a)?)m)|(have been)|(feel))$

        :param verb: Verb component in `text`
        :param text: Text string containing `verb` and, potentially, a user group
        :return: The user group from `text` if `verb` satisfies the regular expression, otherwise an empty string.

        """
        if re.search(r"((('|’|^(a)?)m)|(have been)|(feel))$", verb):
            return re.sub(r"^((the)|a)\s", "", text)
        else:
            return ""

    @classmethod
    def resolve_function(cls, phrase_mention: List[Tuple[Tuple[str, str], str, str, Tuple[str, str]]]) -> List[str]:
        """Extract the user group for a given text string within a phrase mention.

        Uses the `PreProcess.get_user_group` static method to return the user group.

        :param phrase_mention: A nested list of tuples that comprise components of a text string. Each tuple is four
            elements long; a two-element tuple of the part-of-speech (POS) tags of text component, the text component
            string, its theme as string, and a two-element tuple split of the text component according to the POS
            tags. Note only the first and last (fourth) element of `phrase_mention` is used in this code.
        :return: A list containing a potential user group, if the first POS tag is a verb, and the second text
            component split contains a user group.

        """

        # Iterate over each component in `phrase_mention`, and extract possible user groups if the first POS tag is a
        # verb
        res = [cls.get_user_group(*str_split) for pos, _, _, str_split in phrase_mention if "verb" in pos[0]]

        # Return non-empty string user groups
        return [r for r in res if r != ""]

    @staticmethod
    def find_needle(needle: str, hay: str) -> Dict[str, Optional[str]]:
        """For a pattern identical or similar to a phrase `needle` that can be found in a text string `hay`.

        :param needle: A phrase to find in `hay`.
        :param hay: A text string that may contain `needle`, or a variant of it.
        :return: A dictionary, where the key is `needle`, and the value is a pattern similar or identical to `needle`
            that can be found in `hay`. If no pattern can be found, the value is None.

        """

        # Initialise some storage variables
        needle_length = len(needle.split())
        max_sim_val = 0
        max_sim_string = u""

        # Split `hay` into n-grams, and search for a pattern that will match the n-grams
        for ngram in ngrams(hay.split(), needle_length + int(.65 * needle_length)):

            # Concatenate the n-gram, and determine its similarity ratio to the overall phrase `needle`
            hay_ngram = u" ".join(ngram)
            similarity = SM(None, hay_ngram, needle).ratio()

            # Store the similarity ratio, and the concatenated n-gram if the ratio is above `max_sim_val`
            if similarity > max_sim_val:
                max_sim_val = similarity
                max_sim_string = hay_ngram

        # If no string is found, set it to `hay`
        if max_sim_string == "":
            max_sim_string = hay

        # Split `needle` into individual tokens, and extract a regular expression that best matches `needle` to
        # `hay`
        tokens = needle.split(" ")
        if len(tokens) == 1:
            expression = tokens[0]
        else:
            expression = f"({tokens[0]}).*({tokens[-1]})"

        # Find the full pattern in `max_sim_string` that contains `expression`
        result = regex.search(expression, max_sim_string)

        # If `result` is not None, return the found pattern, otherwise return None
        return {needle: result.group() if result else None}
