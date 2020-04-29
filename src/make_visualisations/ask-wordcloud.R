library(tidyverse)                     # data munging and plotting
library(furrr)                         # parallel processing
library(lubridate)                     # dates
library(googledrive)                   # locate files on Google Drive
library(googlesheets4)                 # import from Google Sheets
library(tidytext)                      # stop_words
library(udpipe)                        # parts of speech, lemmas
library(topicmodels)                   # LDA topic modelling
library(broom)                         # Get document topics from topicmodels
library(wordcloud)                     # wordcloud
library(ggalt)                         # better than default ggplot2
library(igraph)                        # graphs
library(ggraph)                        # plot graphs
library(svglite)                       # create SVG images
library(beepr)                         # beep when ready
library(fs)                            # filesystem
library(here)                          # consistent paths from project root

# You also need data.table for data.table::rbindlist()

questions_path <- here("data", "ask", "questions.csv")
model_dir <- here("models")
img_dir <- here("img")
annotations_path <- here("data", "ask", "udpipe-annotations.Rds")
noun_phrase_path <- here("data", "ask", "noun-phrases.Rds")
verb_phrase_path <- here("data", "ask", "verb-phrases.Rds")

options(future.globals.maxSize = Inf)  # Parallelise large jobs
plan(multiprocess)

# Copy tidytext::stop_words to potentially use as a data.table
stop_words <- data.table::copy(stop_words)

# Run to refresh:
# udpipe_download_model(language = "english-ewt", model_dir = model_dir)

# Run to refresh. Needs human interaction. Could be configured to use
# environment variables after being run once by a human.
# questions <-
#   drive_ls(path = "https://drive.google.com/drive/folders/1cxHngoOCoi5T8jUw4Vm_InTsLPw470R4",
#            pattern = "^ask-\\d{4}-\\d{2}-\\d{2}$") %>%
#   mutate(data = map(id, read_sheet)) %>%
#   pull(data) %>%
#   bind_rows() %>%
#   mutate(id = row_number()) %>%
#   select(id, submission_time, question)
# write_csv(questions, questions_path)
questions <- read_csv(questions_path, col_types = "dTc")

# Nearly one day of data as of 2020-04-28 at 22:06
diff(range(questions$submission_time, na.rm = TRUE))

# Remove stopwords and do a wordcloud
words <-
  questions %>%
  unnest_tokens(word, question) %>%
  anti_join(stop_words, by = "word")

png(path(img_dir, "wordcloud-all.png"))
# svglite(path(img_dir, "wordcloud-all.svg"))
wordcloud(words$word, max.words = 50)
dev.off()

# Noun phrases and verb phrases separately

# Latest model
udpipe_model_path <-
  dir_ls(model_dir, glob = "*/english-ewt*.udpipe") %>%
  sort(decreasing = TRUE) %>%
  head(1)

ud_model <- udpipe_load_model(udpipe_model_path)

# # Run to refresh (takes a while)
# x <-
#   udpipe(
#     questions$question,
#     ud_model,
#     parallel.cores = 4,
#     parser = "none",
#     trace = 100
#   )
# saveRDS(x, here("data", "ask", "udpipe-annotations.Rds"))

# Otherwise load from last time
annotations <-
  readRDS(annotations_path) %>%
  as.data.frame() %>%
  as_tibble()

svglite(path(img_dir, "wordcloud-VERB.svg"))
annotations %>%
  filter(upos == "VERB") %>%
  pull(lemma) %>%
  wordcloud(max.words = 50)
dev.off()

svglite(path(img_dir, "wordcloud-NOUN.svg"))
annotations %>%
  filter(upos == "NOUN") %>%
  pull(lemma) %>%
  wordcloud(max.words = 50)
dev.off()

## Using RAKE.  Unigrams not very effective.  Bigrams better.
stats <-
  keywords_rake(annotations, term = "lemma", group = "doc_id",
                relevant = annotations$upos %in% c("NOUN", "ADJ")) %>%
  arrange(freq) %>%
  mutate(key = fct_inorder(keyword))
stats %>%
  filter(ngram > 1) %>%
  arrange(desc(freq)) %>%
  head(20) %>%
  ggplot(aes(key, freq)) +
  geom_col() +
  scale_y_continuous(position = "right") +
  coord_flip()

## Using Pointwise Mutual Information Collocations.  Terrible
annotations$word <- tolower(annotations$token)
stats <-
  keywords_collocation(x = annotations, term = "word", group = "doc_id") %>%
  arrange(freq) %>%
  mutate(key = fct_inorder(keyword))
stats %>%
  arrange(desc(freq)) %>%
  head(20) %>%
  ggplot(aes(key, freq)) +
  geom_col() +
  scale_y_continuous(position = "right") +
  coord_flip()

## Using a sequence of POS tags (noun phrases / verb phrases)

# Simplify Universal POS to single letters to make regex easier
annotations$phrase_tag <-
  as_phrasemachine(annotations$upos, type = "upos")

# A: adjective
# C: coordinating conjuction
# D: determiner
# M: modifier of verb
# N: noun or proper noun
# O: other elements
# P: preposition
# V: verb

noun_phrase_simple <- "(A|N)*N(P+D*(A|N)*N)*"

verb_phrase_simple <-
  "((A|N)*N(P+D*(A|N)*N)*P*(M|V)*V(M|V)*|(M|V)*V(M|V)*D*(A|N)*N(P+D*(A|N)*N)*|(M|V)*V(M|V)*(P+D*(A|N)*N)+|(A|N)*N(P+D*(A|N)*N)*P*((M|V)*V(M|V)*D*(A|N)*N(P+D*(A|N)*N)*|(M|V)*V(M|V)*(P+D*(A|N)*N)+))"

noun_phrase_coordinating_conjuntion <-
  "((A(CA)*|N)*N((P(CP)*)+(D(CD)*)*(A(CA)*|N)*N)*(C(D(CD)*)*(A(CA)*|N)*N((P(CP)*)+(D(CD)*)*(A(CA)*|N)*N)*)*)"

verb_phrase_coordinating_conjuntion <-
  "(((A(CA)*|N)*N((P(CP)*)+(D(CD)*)*(A(CA)*|N)*N)*(C(D(CD)*)*(A(CA)*|N)*N((P(CP)*)+(D(CD)*)*(A(CA)*|N)*N)*)*)(P(CP)*)*(M(CM)*|V)*V(M(CM)*|V)*(C(M(CM)*|V)*V(M(CM)*|V)*)*|(M(CM)*|V)*V(M(CM)*|V)*(C(M(CM)*|V)*V(M(CM)*|V)*)*(D(CD)*)*((A(CA)*|N)*N((P(CP)*)+(D(CD)*)*(A(CA)*|N)*N)*(C(D(CD)*)*(A(CA)*|N)*N((P(CP)*)+(D(CD)*)*(A(CA)*|N)*N)*)*)|(M(CM)*|V)*V(M(CM)*|V)*(C(M(CM)*|V)*V(M(CM)*|V)*)*((P(CP)*)+(D(CD)*)*(A(CA)*|N)*N)+|((A(CA)*|N)*N((P(CP)*)+(D(CD)*)*(A(CA)*|N)*N)*(C(D(CD)*)*(A(CA)*|N)*N((P(CP)*)+(D(CD)*)*(A(CA)*|N)*N)*)*)(P(CP)*)*((M(CM)*|V)*V(M(CM)*|V)*(C(M(CM)*|V)*V(M(CM)*|V)*)*(D(CD)*)*((A(CA)*|N)*N((P(CP)*)+(D(CD)*)*(A(CA)*|N)*N)*(C(D(CD)*)*(A(CA)*|N)*N((P(CP)*)+(D(CD)*)*(A(CA)*|N)*N)*)*)|(M(CM)*|V)*V(M(CM)*|V)*(C(M(CM)*|V)*V(M(CM)*|V)*)*((P(CP)*)+(D(CD)*)*(A(CA)*|N)*N)+))"

sentences <-
  annotations %>%
  group_by(doc_id, paragraph_id, sentence_id) %>%
  group_split()

## Noun phrases
# noun_phrases <-
#   sentences %>%
#   future_map(~ phrases(.x$phrase_tag, tolower(.x$token),
#                        pattern = noun_phrase_coordinating_conjuntion ,
#                        is_regex = TRUE),
#              .progress = TRUE)
# saveRDS(noun_phrases, noun_phrase_path)
# beep()
noun_phrases <- readRDS(noun_phrase_path)

## Verb phrases
# verb_phrases <-
#   sentences %>%
#   future_map(~ phrases(.x$phrase_tag, tolower(.x$token),
#                        pattern = verb_phrase_coordinating_conjuntion ,
#                        is_regex = TRUE),
#              .progress = TRUE)
# beep()
# saveRDS(verb_phrases, verb_phrase_path)
verb_phrases <- readRDS(verb_phrase_path)

# List some common noun phrases
noun_phrases %>%
  data.table::rbindlist() %>%
  .[!data.table::setDT(stop_words), on = c(keyword = "word")] %>% # remove stop_words
  count(keyword, sort = TRUE) %>%
  as_tibble() %>%
  print(n = 100)

# Top 20 of all nouns and noun phrases
noun_phrases %>%
  data.table::rbindlist() %>%
  .[!data.table::setDT(stop_words), on = c(keyword = "word")] %>% # remove stop_words
  count(keyword) %>%
  arrange(n) %>%
  mutate(key = fct_inorder(keyword)) %>%
  top_n(20, n) %>%
  ggplot(aes(n, key)) +
  geom_lollipop(point.colour="steelblue", point.size=2, horizontal=TRUE) +
  labs(x = "", y = "",
       title = "Common nouns",
       subtitle = "Mentioned in the gov.uk/ask service") +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        plot.title = element_text(face = "bold"))
ggsave(path(img_dir, "common-nouns.png"))

# Top 20 multi-word noun phrases
noun_phrases %>%
  data.table::rbindlist() %>%
  filter(ngram > 1) %>%
  count(keyword) %>%
  arrange(n) %>%
  mutate(key = fct_inorder(keyword)) %>%
  top_n(20, n) %>%
  ggplot(aes(n, key)) +
  geom_lollipop(point.colour="steelblue", point.size=2, horizontal=TRUE) +
  labs(x = "", y = "",
       title = "Common multi-word nouns",
       subtitle = "Mentioned in the gov.uk/ask service") +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        plot.title = element_text(face = "bold"))
ggsave(path(img_dir, "common-multi-nouns.png"))

# List some common verb phrases
verb_phrases %>%
  data.table::rbindlist() %>%
  as_tibble() %>%
  count(keyword, sort = TRUE) %>%
  anti_join(stop_words, by = c("keyword" = "word")) %>%
  print(n = 100)

# Top 20 of all verbs and verb phrases
verb_phrases %>%
  data.table::rbindlist() %>%
  count(keyword) %>%
  arrange(n) %>%
  mutate(key = fct_inorder(keyword)) %>%
  top_n(20, n) %>%
  ggplot(aes(n, key)) +
  geom_lollipop(point.colour="steelblue", point.size=2, horizontal=TRUE) +
  labs(x = "", y = "",
       title = "Common verbs",
       subtitle = "Mentioned in the gov.uk/ask service") +
  theme_minimal() +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        plot.title = element_text(face = "bold"))
ggsave(path(img_dir, "common-verbs.png"))

# Co-occurrences of nouns with adjectives
cooc <- cooccurrence(x = filter(annotations, upos %in% c("NOUN", "ADJ")),
                     term = "lemma",
                     group = c("doc_id", "paragraph_id", "sentence_id"))
head(cooc)

wordnetwork <- head(cooc, 30)
wordnetwork <- graph_from_data_frame(wordnetwork)
ggraph(wordnetwork, layout = "fr") +
  geom_edge_link(aes(width = cooc, edge_alpha = cooc), edge_colour = "pink") +
  geom_node_text(aes(label = name), col = "darkgreen", size = 4) +
  theme_graph() +
  theme(legend.position = "none") +
  labs(title = "Common pairs of words within a sentence",
       subtitle = "Nouns and adjectives only")
ggsave(path(img_dir, "cooccurrence-graph.png"))

# Topic modelling of noun phrases

# Join the phrases back to the sentence IDs
x <-
  annotations %>%
  distinct(doc_id, paragraph_id, sentence_id) %>%
  arrange(doc_id, paragraph_id, sentence_id) %>%
  # Repeat each sentence row as many times as there are phrases in it
  .[rep(seq_len(nrow(.)), times = map_int(noun_phrases, nrow)), ] %>%
  # Combine the sentence rows with the phrases rows
  bind_cols(keyword = unlist(map(noun_phrases, pluck, "keyword"))) %>%
  as_tibble() %>%
  anti_join(stop_words, by = c("keyword" = "word"))

## Build document/term/matrix
dtm <- document_term_frequencies(x, document = "doc_id", term = "keyword")
dtm <- document_term_matrix(x = dtm)
dtm <- dtm_remove_lowfreq(dtm, minfreq = 5)

## Build Topicic model. Eight topics worked well for Mat.
m <- LDA(dtm, k = 8, method = "Gibbs",
         control = list(seed = 2020-04-29)) #, nstart = 5, burnin = 2000, best = TRUE))
beep()

## Terms associated with each model
predict(m, type = "terms", min_posterior = 0.01)

## Typical questions in each topic
tidy(m, matrix = "gamma") %>%
  group_by(topic) %>%
  top_n(20, wt = gamma) %>%
  ungroup() %>%
  mutate(id = parse_number(document)) %>%
  inner_join(questions, by = "id") %>%
  arrange(topic, desc(gamma)) %>%
  write_tsv(here("data", "ask", "topics.tsv"))
