from src.make_feedback_tool_data.preprocess import PreProcess
from src.make_feedback_tool_data.regex_category_identification import regex_for_theme, regex_group_verbs
from src.make_feedback_tool_data.text_chunking import ChunkParser

import logging.config
import os

import nltk
import pandas as pd
import re
from tqdm import tqdm

import numpy as np

nltk.download('punkt')


def preproccess_filter_comment_text(full_df, length_threshold=4000):
    """Filter down survey feedback to only english and len < 4K char comments.
    :param full_df:
    :param length_threshold:
    :return:
    """
    logger.info("Removing non-english and lengthy comments...")
    full_df['Q3_pii_removed'] = full_df['Q3_x'].progress_map(PreProcess.replace_pii_regex)
    full_df = full_df[(full_df.Q3_pii_removed.str.len() < length_threshold)]

    full_df = full_df.assign(language=full_df['Q3_pii_removed'].progress_map(PreProcess.detect_language))

    lang_dist = full_df['language'].value_counts().to_dict()
    logger.debug(f"Number of unique languages: {len(lang_dist)}")
    logger.debug(f"English: {lang_dist['en'] / sum(lang_dist.values()):.2%}")
    logger.debug(f"-: {lang_dist['-'] / sum(lang_dist.values()):.2%}")

    full_df['is_en'] = full_df['language'].isin(["en", "un", "-", "sco"])

    return full_df[full_df['is_en']]


def save_intermediate_df(processed_df, cache_pos_filename):
    """

    :param processed_df:
    :param cache_pos_filename:
    :return:
    """
    logger.info(f"Saving preprocessed survey data at: {cache_pos_filename}...")

    processed_df['lemmas'] = processed_df['pos_tag'].progress_map(lambda x: [token[2] for sent in x for token in sent])
    processed_df['words'] = processed_df['pos_tag'].progress_map(lambda x: [token[0] for sent in x for token in sent])

    processed_df.to_csv(cache_pos_filename, index=False)


def extract_phrase_mentions(df, grammar_filename):
    """For each POS-tagged sentence from comments in the survey data:
    1. Detect and extract chunks as defined by grammar, merge adjacent chunks
    2. Compute pair-wise combinations of chunks
    3. If a combination type is in predefined list, append it to phrase_mentions list

    :param df: filtered, preprocessed survey dataframe
    :param grammar_filename:
    :return: inplace define column containing applicable phrase mentions
    """

    logger.info("Detecting and extracting phrase-level mentions...")
    phrase_mentions = []

    parser = ChunkParser(grammar_filename)

    for comment, vals in tqdm(df[['Q3_x_edit', 'pos_tag']].values):
        sents = parser.extract_phrase(vals, merge_inplace=True)
        phrase_mentions.append([])
        for combo in PreProcess.compute_combinations(sents, 2):
            key = (combo[0].label, combo[1].label)
            arg1 = combo[0].text.lower()
            arg2 = combo[1].text.lower()

            if key in [('verb', 'noun'), ('verb', 'prep_noun'),
                       ('verb', 'noun_verb'), ('noun', 'prep_noun'),
                       ('prep_noun', 'noun'), ('prep_noun', 'prep_noun')]:
                generic_phrase = (regex_group_verbs(arg1), regex_for_theme(arg2))

                arg1 = re.sub(r"[?()\[\]+*]", "", arg1)
                arg2 = re.sub(r"[?()\[\]+*]", "", arg2)
                phrase = (arg1, arg2)
                exact_phrase = list(PreProcess.find_needle(" ".join(phrase), comment.lower()).values())[0]

                if exact_phrase is not None:
                    exact_verb = list(PreProcess.find_needle(phrase[0], exact_phrase).values())[0]
                    if exact_verb is not None:
                        exact_phrase = (exact_verb, re.sub(exact_verb, "", exact_phrase).strip())
                        phrase_mentions[-1].append({"chunked_phrase": phrase,
                                                    "exact_phrase": exact_phrase,
                                                    "generic_phrase": generic_phrase,
                                                    "key": key})

    df['themed_phrase_mentions'] = phrase_mentions


def create_phrase_level_columns(df):
    """

    :param df:
    :return:
    """
    logger.info("Assigning exact_phrases column...")
    df['exact_phrases'] = df['themed_phrase_mentions'].progress_map(
        lambda x: "\n".join([", ".join(item['exact_phrase'])
                             for item in x
                             if item['key'][0] == "verb"]))
    logger.info("Assigning generic_phrases column...")
    df['generic_phrases'] = df['themed_phrase_mentions'].progress_map(
        lambda x: "\n".join([", ".join(item['generic_phrase'])
                             for item in x
                             if item['key'][0] == "verb"]))
    # df['user_groups'] = df['theme_mentions'].progress_map(lambda x: get_user_group(x))


def create_dataset(survey_filename, grammar_filename, cache_pos_filename, output_filename):
    """

    :param survey_filename:
    :param grammar_filename:
    :param cache_pos_filename:
    :param output_filename:
    :return:
    """

    logger.info(f"Reading survey file: {survey_filename}")
    survey_data_df = pd.read_csv(survey_filename)

    drop_duplicate_rows(survey_data_df)

    survey_data_df = preproccess_filter_comment_text(survey_data_df)

    logger.info("Part of speech tagging comments...")
    survey_data_df['pos_tag'] = survey_data_df[['Q3_pii_removed', 'is_en']].progress_apply(
        lambda x: PreProcess.part_of_speech_tag(x[0]) if x[1] else [],
        axis=1)

    logger.info("Pre-processing feedback text for matching...")
    survey_data_df["Q3_x_edit"] = survey_data_df["Q3_x"].replace(np.nan, '', regex=True)
    survey_data_df["Q3_x_edit"] = survey_data_df["Q3_x_edit"].progress_map(lambda x: ' '.join(re.sub(r"[()\[\]+*]", "",
                                                                                                     x).split()))

    extract_phrase_mentions(survey_data_df, grammar_filename)

    save_intermediate_df(survey_data_df, cache_pos_filename)

    create_phrase_level_columns(survey_data_df)

    columns_to_keep = ['primary_key', 'intents_clientID', 'visitId', 'fullVisitorId',
                       'hits_pagePath', 'Started', 'Ended', 'Q1_x', 'Q2_x', 'Q3_x', 'Q4_x',
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
                       'Started_Date', 'Ended_Date', 'Started_Date_sub_12h', 'exact_phrases',
                       'generic_phrases']
    survey_data_df.rename(columns={'Q3_x_edit': 'Q3_x'}, inplace=True)
    logger.info(f"Saving survey data at: {output_filename}...")
    survey_data_df[columns_to_keep].to_csv(output_filename, index=False)


def drop_duplicate_rows(survey_data_df):
    """
    Dropped duplicated rows, based on the primary_key column, which is a unique session identifier.
    :param survey_data_df:
    :return: inplace deduped survey dataframe
    """
    loaded_number_rows = survey_data_df.shape[0]
    logger.info(f"Number of rows: {loaded_number_rows}")
    logger.info(f"Unique clientIds: {survey_data_df.intents_clientID.nunique()}")
    logger.info(f"Unique primary key: {survey_data_df.primary_key.nunique()}")
    logger.info(f"Unique session_ids: {survey_data_df.session_id.nunique()}")
    logger.info("Dropping duplicates...")
    survey_data_df.drop_duplicates("primary_key", inplace=True)
    survey_data_df.reset_index(inplace=True, drop=True)
    logger.info(f"Dropped {loaded_number_rows - survey_data_df.shape[0]} rows.")


if __name__ == "__main__":
    log_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logging.conf')
    logging.config.fileConfig(log_file_path)
    logger = logging.getLogger(__name__)

    DATA_DIR = os.getenv("DIR_DATA")

    survey_data_filename = os.path.join(DATA_DIR, 'uis_20200401_20200409.csv')
    chunk_grammar_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "grammar.txt")
    cache_pos_data_filename = survey_data_filename.replace(".csv", "_cache.csv")
    output_data_filename = survey_data_filename.replace(".csv", "_phrases_user_groups.csv")

    create_dataset(survey_data_filename, chunk_grammar_filename, cache_pos_data_filename, output_data_filename)
    # parser = ChunkParser(chunk_grammar_filename)
    # comment = "This is an example sentence. This is another."
    # tagged = PreProcess.part_of_speech_tag(comment)
    # for sent in parser.extract_phrase(tagged, merge_inplace=True):
    #     for chunk in sent:
    #         print(chunk.text, chunk.label)
