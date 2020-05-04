import os
import re
import sys

import pandas as pd
import spacy
from nltk import sent_tokenize
# https://markhneedham.com/blog/2017/11/28/python-polyglot-modulenotfounderror-no-module-named-icu/
from polyglot.detect import Detector
from tqdm import tqdm
from src.make_feedback_tool_data.text_chunking import ChunkParser
from src.make_feedback_tool_data.regex_category_identification import regex_for_theme, regex_group_verbs

tqdm.pandas()

nlp = spacy.load("en_core_web_sm")

pii_filtered = ["DATE_OF_BIRTH", "EMAIL_ADDRESS", "PASSPORT", "PERSON_NAME",
                "PHONE_NUMBER", "STREET_ADDRESS", "UK_NATIONAL_INSURANCE_NUMBER", "UK_PASSPORT"]
pii_regex = "|".join([f"\\[{p}\\]" for p in pii_filtered])


def split_sentences(comment):
    """

    :param comment:
    :return:
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


def extract_phrase_mentions(df, grammar_filename):
    phrase_mentions = []

    parser = ChunkParser(grammar_filename)

    for vals in tqdm(df.pos_tag.values):
        sents = parser.extract_phrase(vals, True)
        phrase_mentions.append([])
        for combo in compute_combinations(sents, 2):
            key = (combo[0].label, combo[1].label)
            arg1 = combo[0].text.lower()
            arg2 = combo[1].text.lower()

            if key in [('verb', 'noun'), ('verb', 'prep_noun'),
                       ('verb', 'noun_verb'), ('noun', 'prep_noun'),
                       ('prep_noun', 'noun'), ('prep_noun', 'prep_noun')]:
                mention_theme = f"{regex_group_verbs(arg1)} - {regex_for_theme(arg2)}"

                arg1 = re.sub(r"\(|\)|\[|\]|\+", "", arg1)
                arg2 = re.sub(r"\(|\)|\[|\]|\+", "", arg2)
                phrase = f"{arg1} {arg2}"
                phrase_mentions[-1].append((key, phrase, mention_theme, (arg1, arg2)))

    df['theme_mentions'] = phrase_mentions


if __name__ == "__main__":
    DATA_DIR = os.getenv("DATA_DIR")
    filename = os.path.join(DATA_DIR, "")
    df = pd.read_csv(filename)
