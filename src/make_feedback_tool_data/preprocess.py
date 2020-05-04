import re
import sys

import spacy
from nltk import sent_tokenize
# https://markhneedham.com/blog/2017/11/28/python-polyglot-modulenotfounderror-no-module-named-icu/
from polyglot.detect import Detector
from tqdm import tqdm
import regex
from difflib import SequenceMatcher as SM
from nltk.util import ngrams
import codecs
tqdm.pandas()

nlp = spacy.load("en_core_web_sm")

pii_filtered = ["DATE_OF_BIRTH", "EMAIL_ADDRESS", "PASSPORT", "PERSON_NAME",
                "PHONE_NUMBER", "STREET_ADDRESS", "UK_NATIONAL_INSURANCE_NUMBER", "UK_PASSPORT"]
pii_regex = "|".join([f"\\[{p}\\]" for p in pii_filtered])


def split_sentences(comment):
    """
    Split multi-sentence comments into list of sentences.
    :param comment: comment text string
    :return: list of strings
    """
    return sent_tokenize(comment)


def replace_pii_regex(text):
    """

    :param text:
    :return:
    """
    return re.sub(pii_regex, "", text)


def part_of_speech_tag(comment):
    """
    Part of speech tag comments.
    :param comment: a PII-tag removed text comment
    :return: nested list of lists
    """
    sentences = split_sentences(comment)
    return [[(token.text, token.tag_, token.lemma_) for token in nlp(sentence)] for sentence in sentences]


def detect_language(text):
    """

    :param text:
    :return:
    """
    if text != "-":
        try:
            langs = {language.confidence: language.code for language in Detector(text, quiet=True).languages}
            return langs[max(langs.keys())]
        except Exception:
            return f"[ERROR] {text} {sys.exc_info()}"
    return "-"


def compute_combinations(sentences, n):
    """

    :param sentences:
    :param n:
    :return:
    """
    return [chunks[i:i + n] for chunks in sentences for i in range(len(chunks) - (n - 1))]


def get_user_group(arg1, arg2):
    if re.search(r"((('|’|^(a)?)m)|(have been)|(feel))$", arg1):
        return re.sub(r"^((the)|a)\s","", arg2)
    return ""

def resolve_function(x):
    res = [get_user_group(*args) for theme,_,_,args in x if "verb" in theme[0]]
    return [r for r in res if r != ""]


def find_needle(needle, hay):
    needle_length = len(needle.split())
    max_sim_val = 0
    max_sim_string = u""
    #     print(needle)
    for ngram in ngrams(hay.split(), needle_length + int(.65 * needle_length)):
        hay_ngram = u" ".join(ngram)
        similarity = SM(None, hay_ngram, needle).ratio()
        if similarity > max_sim_val:
            max_sim_val = similarity
            max_sim_string = hay_ngram

    if max_sim_string == "":
        max_sim_string = hay

    tokens = needle.split(" ")
    if len(tokens) == 1:
        expression = tokens[0]
    else:
        expression = f"({tokens[0]}).*({tokens[-1]})"
    result = regex.search(expression, max_sim_string)

    if result is not None:
        pattern = result.group()

        return {needle: pattern}
    return {needle: None}

