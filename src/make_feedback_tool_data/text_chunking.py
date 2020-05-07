from nltk import RegexpParser, tree
from src.make_feedback_tool_data.chunk import Chunk
import logging.config

class ChunkParser:

    def __init__(self, grammar_filename):
        self.logger = logging.getLogger(__name__)
        self.grammar = self._load_grammar_from_file(grammar_filename)
        self.logger.info("Initializing parser...")
        self.parser = RegexpParser(self.grammar)

    def _load_grammar_from_file(self, grammar_filename):
        """

        :param grammar_filename:
        :return:
        """
        self.logger.info(f"Using grammar file: {grammar_filename}")
        with open(grammar_filename, "r") as f:
            return "".join(f.readlines())

    def _chunk_text(self, tagged):
        """

        :param tagged:
        :return:
        """

        chunks = self.parser.parse(tagged)
        index = 0
        segments = []
        for el in chunks:
            if type(el) == tree.Tree:
                chunk = Chunk(el.label(), el.leaves(), list(range(index, index + len(el.leaves()))))
                segments.append(chunk)
                index += len(el.leaves())
            else:
                index += 1
        return segments

    def _merge_adjacent_chunks(self, chunks):
        """

        :param chunks:
        :return:
        """
        merged = []
        previous_label = ""
        for chunk in chunks:
            if chunk.label == previous_label and chunk.label != "prep_noun":
                merged[-1] = Chunk(chunk.label,
                                   merged[-1].tokens + chunk.tokens,
                                   merged[-1].indices + chunk.indices)
            else:
                merged.append(chunk)
            previous_label = chunk.label
        return merged

    def extract_phrase(self, sentences, merge_inplace=False):
        """

        :param sentences:
        :param merge_inplace:
        :return:
        """
        chunks = []
        for sentence in sentences:
            chunks.append(self._chunk_text(sentence))
        if merge_inplace:
            return [self._merge_adjacent_chunks(chunk) for chunk in chunks]
        return chunks
