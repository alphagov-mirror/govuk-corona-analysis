import os
import re
import sys

import pandas as pd
import spacy
from nltk import sent_tokenize
# https://markhneedham.com/blog/2017/11/28/python-polyglot-modulenotfounderror-no-module-named-icu/
from polyglot.detect import Detector
from tqdm import tqdm

tqdm.pandas()

nlp = spacy.load("en_core_web_sm")

pii_filtered = ["DATE_OF_BIRTH", "EMAIL_ADDRESS", "PASSPORT", "PERSON_NAME",
                "PHONE_NUMBER", "STREET_ADDRESS", "UK_NATIONAL_INSURANCE_NUMBER", "UK_PASSPORT"]
pii_regex = "|".join([f"\\[{p}\\]" for p in pii_filtered])


def split_sentences(comment):
    return sent_tokenize(comment)


def replace_pii_regex(text):
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
    if text != "-":
        try:
            langs = {language.confidence: language.code for language in Detector(text, quiet=True).languages}
            return langs[max(langs.keys())]
        except Exception:
            return f"[ERROR] {text} {sys.exc_info()}"
    return "-"


def keep_english_comments(full_df):
    """
    
    :param full_df: 
    :return: 
    """
    full_df['Q3_pii_removed'] = full_df['Q3_x'].progress_map(replace_pii_regex)
    full_df = full_df[(full_df.Q3_pii_removed.str.len() < 4000)]
    full_df = full_df.assign(language=full_df['Q3_pii_removed'].progress_map(detect_language))

    lang_dist = full_df['language'].value_counts().to_dict()
    print(f"Number of unique languages: {len(lang_dist)}")
    print(f"English: {lang_dist['en'] / sum(lang_dist.values()):.2%}")
    print(f"-: {lang_dist['-'] / sum(lang_dist.values()):.2%}")
    # list(lang_dist.items())[0:10]

    full_df['is_en'] = full_df['language'].isin(["en", "un", "-", "sco"])

    return full_df[full_df['is_en']]


def save_intermediate_df(processed_df, cache_pos_filename):
    """

    :param processed_df:
    :param cache_pos_filename:
    :return:
    """
    processed_df['pos_tag'] = processed_df[['Q3_pii_removed', 'is_en']].progress_apply(
        lambda x: part_of_speech_tag(x[0]) if x[1] else [],
        axis=1)
    processed_df['lemmas'] = processed_df['pos_tag'].progress_map(lambda x: [token[2] for sent in x for token in sent])

    processed_df['words'] = processed_df['pos_tag'].progress_map(lambda x: [token[0] for sent in x for token in sent])

    processed_df.to_csv(cache_pos_filename, index=False)


if __name__ == "__main__":
    DATA_DIR = os.getenv("DATA_DIR")
    filename = os.path.join(DATA_DIR, "")
    df = pd.read_csv(filename)
