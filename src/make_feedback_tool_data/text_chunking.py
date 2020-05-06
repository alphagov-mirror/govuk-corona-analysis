from nltk import RegexpParser
from src.make_feedback_tool_data.chunk import Chunk
from typing import List, Tuple, Union
import nltk.tree


class ChunkParser:

    def __init__(self, grammar_filename: str) -> None:
        """

        :param grammar_filename: A path string to file containing regular expression grammar patterns usable by the
            `grammar` argument of the nltk.chunk.regexp.RegexpParser class. For each grammar type, each pattern
            should be listed on a separate line,  and in descending order of priority (highest first).

        """

        # Load the regular expressions from `grammar_filename`
        self.grammar = self._load_grammar_from_file(grammar_filename)

        # Initialise a nltk.RegexpParser object using `self.grammar`
        self.parser = RegexpParser(self.grammar)

    @staticmethod
    def _load_grammar_from_file(grammar_filename: str) -> str:
        """Load grammar regular expression patterns from a file.

        :param grammar_filename: A path string to file containing regular expression grammar patterns usable by the
            `grammar` argument of the nltk.chunk.regexp.RegexpParser class. For each grammar type, each pattern
            should be listed on a separate line,  and in descending order of priority (highest first).
        :return: A string of the parsed text from `grammar_filename`.

        """
        with open(grammar_filename, "r") as f:
            return "".join(f.readlines())

    def _chunk_text(self, tagged: Union[nltk.tree.Tree, List[Tuple[str, str, str]]]) -> List[Chunk]:
        """Chunk tokens with part-of-speech (POS) tags and lemmas according to grammar regular expressions.

        Uses the `parse` method from nltk.chunk.regexp.RegexpParser class for chunking.

        :param tagged: A list of three-element tuples of a token, its POS tag, and its lemma.
        :return: A list of `src.make_feedback_tool_data.Chunk` objects, where each chunk is a grammar chunk as
            defined by the regular expressions in `grammar_filename` when the `ChunkParser` object was initialised.

        """

        # Chunk `tagged` according to the grammar regular expressions from the `grammar_filename` when the
        # `ChunkParser` object was initialised
        chunks = self.parser.parse(tagged)

        # Initialise a counter, and list to storing future `Chunk` objects
        index = 0
        segments = []

        # Iterate over `chunks`, and check if the iteration is a nltk.tree.Tree object
        for el in chunks:
            if isinstance(el, nltk.tree.Tree):

                # Crate a `Chunk` object from `el`, using the `index` counter to define the position of tokens within
                # the original sentence
                chunk = Chunk(el.label(), el.leaves(), list(range(index, index + len(el.leaves()))))

                # Append `chunk` to `segments`, and increase the `index` counter by the number of tokens in `el`
                segments.append(chunk)
                index += len(el.leaves())

            else:

                # If `el` is not a nltk.tree.Tree object, skip over `el`, i.e. the token, and increase the `index`
                # counter by one
                index += 1

        return segments

    @staticmethod
    def _merge_adjacent_chunks(chunks: List[Chunk]) -> List[Chunk]:
        """Merge adjacent grammar chunks together to reduce the number of chunks returned.

        Only adjacent chunks with the same label, where the label is not 'prep_noun' are merged together. Otherwise,
        the chunk, and its relative positioning are left unchanged.

        :param chunks: A list of `src.make_feedback_tool_data.Chunk` objects.
        :return: A list of `src.make_feedback_tool_data.Chunk` objects, where adjacent chunks with the same grammar
        in `chunks` are merged together, as long as the grammar is not 'prep_noun'. All other chunks are left unchanged.

        """

        # Initialise storage variables
        merged = []
        previous_label = ""

        # Iterate over `chunks`, checking if the `label` attribute of `chunks` is the same as `previous_label`,
        # and also not 'prep_noun'; if not, keep the `chunk` unchanged in `merged`
        for chunk in chunks:
            if chunk.label == previous_label and chunk.label != "prep_noun":

                # Replace the last element of `merged` with a new `Chunk` object that uses the same `label`
                # attribute, but merges the `tokens` and `indices` attributes of the previous and current `chunk`.
                # This helps reduce the number of grammar chunks returned, by collapsing identical `label`s together
                merged[-1] = Chunk(chunk.label,
                                   merged[-1].tokens + chunk.tokens,
                                   merged[-1].indices + chunk.indices)
            else:
                merged.append(chunk)

            # Set `previous_label` to the current `chunk.label` for the next iteration
            previous_label = chunk.label

        return merged

    def extract_phrase(self, sentences: List[List[Tuple[str, str, str]]], merge_inplace: bool = False) \
            -> List[List[Chunk]]:
        """Extract phrases for each grammar chunk of a part-of-speech (POS) tagged sentence.

        :param sentences: A list of list of tuples, where the inner list represents sentences. The
            three-element tuples in each sentence list are a token, its POS tag, and its lemma.
        :param merge_inplace: Default: False. If True, adjacent, identically-labelled grammar `Chunk` objects in each
            sentence list of `sentences` are merged together, unless the are 'prep_noun'. See
            `ChunkParser._merge_adjacent_chunks` for further information. If False, no merging is performed.
        :return: A list of `Chunk` objects, where each object represents a phrase extracted based on grammar rules
            defined in `grammar_filename` when the `ChunkParser` object was initialised.

        """

        # Initialise a storage variable
        chunks = []

        # Iterate over `sentences` and, for each `sentence`, chunk it, and append to `chunks`
        for sentence in sentences:
            chunks.append(self._chunk_text(sentence))

        # If `merge_inplace` is True, merge adjacent, identically-labelled grammar chunks. Otherwise, return the
        # `chunks` unmodified
        if merge_inplace:
            return [self._merge_adjacent_chunks(chunk) for chunk in chunks]
        else:
            return chunks
