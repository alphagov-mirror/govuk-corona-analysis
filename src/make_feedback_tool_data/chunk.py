import re


class Chunk:

    def __init__(self, label, tokens, indices):
        self.label = label
        self.tokens = tokens
        self.indices = indices
        self.text = self.text()
        self.lemma = self.lemma()
        self.important_lemma = self.important_lemma()
        self.important_word = self.important_word()

    def text(self):
        return " ".join([w for w, _, _ in self.tokens])

    def lemma(self):
        return " ".join([l for _, _, l in self.tokens])

    def tagable_words(self):
        return [(w, pos) for w, pos, _ in self.tokens if re.search(r"(NN)|(VB)", pos)]

    def important_word(self):
        return " ".join([w for w, pos, _ in self.tokens if re.search(r"(NN)|(VB)|(JJ)|(CD)", pos)])

    def important_lemma(self):
        return " ".join([l for _, pos, l in self.tokens if re.search(r"(NN)|(VB)|(JJ)|(CD)", pos)])

