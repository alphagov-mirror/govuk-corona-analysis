import os
import pandas as pd
import spacy
from nltk import sent_tokenize
import sys
from tqdm import tqdm
import re

from ast import literal_eval

# https://markhneedham.com/blog/2017/11/28/python-polyglot-modulenotfounderror-no-module-named-icu/
from polyglot.detect import Detector

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


def keep_english_comments(df):
    df['Q3_pii_removed'] = df['Q3_x'].progress_map(replace_pii_regex)
    df = df[(df.Q3_pii_removed.str.len() < 4000)]
    df = df.assign(language=df['Q3_pii_removed'].progress_map(detect_language))

    lang_dist = df['language'].value_counts().to_dict()
    print(f"Number of unique languages: {len(lang_dist)}")
    print(f"English: {lang_dist['en'] / sum(lang_dist.values()):.2%}")
    print(f"-: {lang_dist['-'] / sum(lang_dist.values()):.2%}")
    # list(lang_dist.items())[0:10]

    df['is_en'] = df['language'].isin(["en", "un", "-", "sco"])

    return df[df['is_en']]


def save_intermediate_df(df):
    cache_pos_filename = os.path.join(DATA_DIR, "uis_20200401_20200409_lang_pos.csv")

    df['pos_tag'] = df[['Q3_pii_removed', 'is_en']].progress_apply(lambda x: part_of_speech_tag(x[0]) if x[1] else [],
                                                                   axis=1)
    df['lemmas'] = df['pos_tag'].progress_map(lambda x: [token[2] for sent in x for token in sent])

    df['words'] = df['pos_tag'].progress_map(lambda x: [token[0] for sent in x for token in sent])

    df.to_csv(cache_pos_filename, index=False)


if __name__ == "__main__":
    DATA_DIR = os.getenv("DATA_DIR")
