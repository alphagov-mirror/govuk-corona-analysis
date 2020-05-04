import os
import re

import pandas as pd
from tqdm import tqdm

from src.make_feedback_tool_data.regex_category_identification import regex_for_theme, regex_group_verbs
from src.make_feedback_tool_data.text_chunking import ChunkParser
from src.make_feedback_tool_data.preprocess import replace_pii_regex, detect_language, part_of_speech_tag, \
    compute_combinations, find_needle, resolve_function

import numpy as np


def preproccess_filter_comment_text(full_df):
    """
    Filter down survey feedback to only english and len < 4K char comments.
    :param full_df:
    :return:
    """

    full_df['Q3_pii_removed'] = full_df['Q3_x'].progress_map(replace_pii_regex)
    full_df = full_df[(full_df.Q3_pii_removed.str.len() < 4000)]

    full_df = full_df.assign(language=full_df['Q3_pii_removed'].progress_map(detect_language))

    # lang_dist = full_df['language'].value_counts().to_dict()
    # print(f"Number of unique languages: {len(lang_dist)}")
    # print(f"English: {lang_dist['en'] / sum(lang_dist.values()):.2%}")
    # print(f"-: {lang_dist['-'] / sum(lang_dist.values()):.2%}")
    # list(lang_dist.items())[0:10]

    full_df['is_en'] = full_df['language'].isin(["en", "un", "-", "sco"])

    return full_df[full_df['is_en']]


def save_intermediate_df(processed_df, cache_pos_filename):
    """

    :param processed_df:
    :param cache_pos_filename:
    :return:
    """

    processed_df['lemmas'] = processed_df['pos_tag'].progress_map(lambda x: [token[2] for sent in x for token in sent])
    processed_df['words'] = processed_df['pos_tag'].progress_map(lambda x: [token[0] for sent in x for token in sent])

    processed_df.to_csv(cache_pos_filename, index=False)


def extract_phrase_mentions(df, grammar_filename, cache_pos_filename):
    """

    :param df:
    :param grammar_filename:
    :param cache_pos_filename:
    :return:
    """
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
    df['theme_mentions_user'] = df['theme_mentions'].map(resolve_function)


def create_phrase_level_data(df, theme_col, phrase_type):
    """
    Create
    :param df:
    :param theme_col:
    :param phrase_type:
    :return:
    """
    df[f'{phrase_type}_dict'] = df[[theme_col, "Q3_x_edit"]][:]. \
        progress_apply(lambda x: [find_needle(phrase, x[1].lower()) for _, phrase, _, _ in x[0]], axis=1)
    df[f'{phrase_type}_list'] = df[f'{phrase_type}_dict'].progress_map(lambda x: [value for phrase_dict in x for
                                                                                  value in phrase_dict.values() if
                                                                                  value is not None] if not isinstance(
        x, float) else [])
    df[f"{phrase_type}"] = df[f'{phrase_type}_list'].progress_map(lambda x: ", ".join(x))


def create_phrase_level_columns(df):
    """

    :param df:
    :return:
    """
    df["Q3_x_edit"] = df["Q3_x"].replace(np.nan, '', regex=True)
    df["Q3_x_edit"] = df["Q3_x_edit"].progress_map(lambda x: ' '.join(re.sub(r"\(|\)|\[|\]|\+", "", x).split()))

    create_phrase_level_data(df, "theme_mentions", "phrases")
    create_phrase_level_data(df, "theme_mentions_user", "user_phrases")


def create_dataset(df, grammar_filename, cache_pos_filename):
    """

    :param df:
    :param grammar_filename:
    :param cache_pos_filename:
    :return:
    """
    df['pos_tag'] = df[['Q3_pii_removed', 'is_en']].progress_apply(
        lambda x: part_of_speech_tag(x[0]) if x[1] else [],
        axis=1)

    save_intermediate_df(df, cache_pos_filename)

    extract_phrase_mentions(df, grammar_filename, cache_pos_filename)

    create_phrase_level_columns(df)

    df_sub = df[['primary_key', 'intents_clientID', 'visitId', 'fullVisitorId',
                 'hits_pagePath', 'Started', 'Ended', 'Q1_x', 'Q2_x', 'Q3_x_edit', 'Q4_x',
                 'Q5_x', 'Q6_x', 'Q7_x', 'Q8_x', 'session_id', 'dayofweek', 'isWeekend',
                 'hour', 'country', 'country_grouping', 'UK_region', 'UK_metro_area',
                 'channelGrouping', 'deviceCategory',
                 'total_seconds_in_session_across_days',
                 'total_pageviews_in_session_across_days', 'finding_count',
                 'updates_and_alerts_count', 'news_count', 'decisions_count',
                 'speeches_and_statements_count', 'transactions_count',
                 'regulation_count', 'guidance_count', 'business_support_count',
                 'policy_count', 'consultations_count', 'research_count',
                 'statistics_count', 'transparency_data_count',
                 'freedom_of_information_releases_count', 'incidents_count',
                 'done_page_flag', 'count_client_error', 'count_server_error',
                 'ga_visit_start_timestamp', 'ga_visit_end_timestamp',
                 'intents_started_date', 'events_sequence', 'search_terms_sequence',
                 'cleaned_search_terms_sequence', 'top_level_taxons_sequence',
                 'page_format_sequence', 'Sequence', 'PageSequence', 'flag_for_criteria',
                 'full_url_in_session_flag', 'UserID', 'UserNo', 'Name', 'Email',
                 'IP Address', 'Unique ID', 'Tracking Link', 'clientID', 'Page Path',
                 'Q1_y', 'Q2_y', 'Q3_y', 'Q4_y', 'Q5_y', 'Q6_y', 'Q7_y', 'Q8_y',
                 'Started_Date', 'Ended_Date', 'Started_Date_sub_12h', 'phrases', 'user_phrases']]

    df_sub.rename(columns={'Q3_x_edit': 'Q3_x'}, inplace=True)
    df_sub.to_csv(os.path.join(DATA_DIR, 'uis_20200401_20200409_phrases_user_groups.csv'), index=False)


if __name__ == "__main__":
    DATA_DIR = os.getenv("DATA_DIR")
    filename = os.path.join(DATA_DIR, "")

    grammar_filename = ""
    cache_pos_filename = ""

    survey_data = pd.read_csv(filename)

    filtered_df = preproccess_filter_comment_text(survey_data)
    create_dataset(filtered_df, grammar_filename, cache_pos_filename)
