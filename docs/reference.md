# `src` API Reference

This page gives an overview of all public objects, functions, and methods in the `src` package.

## `make_feedback_tool_data`

```eval_rst
.. currentmodule:: src.make_feedback_tool_data

```

### Summary function

```eval_rst
.. autosummary::
    :toctree: api/

    create_dataset

```

#### Composite functions

```eval_rst
.. autosummary::
    :toctree: api/

    create_phrase_level_columns
    drop_duplicate_rows
    extract_phrase_mentions
    preprocess_filter_comment_text
    save_intermediate_df

```

### Text pre-processing

```eval_rst
.. autosummary::
    :toctree: api/

    PreProcess
    PreProcess.split_sentences
    PreProcess.replace_pii_regex
    PreProcess.part_of_speech_tag
    PreProcess.detect_language
    PreProcess.compute_combinations
    PreProcess.get_user_group
    PreProcess.resolve_function
    PreProcess.find_needle

```

### Text chunking based on regular expression grammar rules

```eval_rst
.. autosummary::
    :toctree: api/

    ChunkParser
    ChunkParser.extract_phrase
    regex_category_identification
    regex_for_theme
    regex_group_verbs

```


### Part-of-speech (POS) extraction

```eval_rst
.. autosummary::
    :toctree: api/

    Chunk
    Chunk.text
    Chunk.lemma
    Chunk.tagable_words
    Chunk.important_word
    Chunk.important_lemma

```
