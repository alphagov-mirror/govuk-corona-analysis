import re


class Chunk:

    def __init__(self, label, tokens, indices):
        """Helper class to extract and combine useful tokens/lemmas based on part-of-speech (POS) tag.

        :param label: A POS tag.
        :param tokens: A three-element tuple or list of a token, its POS tag, and its lemma.
        :param indices: The index of the word in the sentence.

        """
        self.label = label
        self.tokens = tokens
        self.indices = indices
        self.text = self.text()
        self.lemma = self.lemma()
        self.important_lemma = self.important_lemma()
        self.important_word = self.important_word()

    def text(self):
        """Combine tokens into a text string.

        :return: A string of all tokens delimited by a space.

        """
        return " ".join([w for w, _, _ in self.tokens])

    def lemma(self):
        """Combine lemmas of the tokens into a string.

        :return: A string of all lemmas delimited by a space.

        """
        return " ".join([l for _, _, l in self.tokens])

    def tagable_words(self):
        """Get each token and its parts-of-speech (POS) tag, if the POS tag is a noun or verb.

        :return: A list of two-element tuples, where the first element is the token, and the second element is the
        noun or verb POS tag (NN or VB).

        """
        return [(w, pos) for w, pos, _ in self.tokens if re.search(r"(NN)|(VB)", pos)]

    def important_word(self):
        """Get a string of all important tokens, based on their part-of-speech (POS) tag.

        Important tokens are defined as being nouns, verbs, adjectives, or cardinal numbers.

        :return: A string of all important tokens delimited by a space.

        """
        return " ".join([w for w, pos, _ in self.tokens if re.search(r"(NN)|(VB)|(JJ)|(CD)", pos)])

    def important_lemma(self):
        """Get a string of all important lemmas, based on their part-of-speech (POS) tag.

        Important lemmas are defined as being nouns, verbs, adjectives, or cardinal numbers.

        :return: A string of all important lemmas delimited by a space.

        """
        return " ".join([l for _, pos, l in self.tokens if re.search(r"(NN)|(VB)|(JJ)|(CD)", pos)])
