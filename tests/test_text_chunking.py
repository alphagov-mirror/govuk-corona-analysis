from src.make_feedback_tool_data.chunk import Chunk
from src.make_feedback_tool_data.text_chunking import ChunkParser, FIlE_GRAMMAR
import inspect
import pytest

# Compile a list of the attribute, and instance and static method names in the `ChunkParser` class
args_chunkparser_member_names = ["_load_grammar_from_file", "_chunk_text", "_merge_adjacent_chunks", "extract_phrase"]

# Define a list of method names in `ChunkParser`
args_chunkparser_method_names = [a for a in args_chunkparser_member_names if a not in ["_load_grammar_from_file"]]


@pytest.mark.parametrize("test_method_name", args_chunkparser_method_names)
def test_methods_in_members(test_method_name):
    """Test that a method exists in args_chunkparser_member_names."""
    assert test_method_name in args_chunkparser_member_names


def test_chunkparser_has_members():
    """Test that ChunkParser has all the member objects in args_chunkparser_member_names."""

    # Get the member objects of the `ChunkParser` class
    test_output = [m[0] for m in inspect.getmembers(ChunkParser, inspect.isroutine) if not m[0].startswith("__")]

    # Assert that all the member objects exist
    assert set(test_output) == set(args_chunkparser_member_names)


# Define test cases for the in the `TestChunkParserInitialisation` test class
args_chunk_parser_initialisation = [None, FIlE_GRAMMAR]

# Define the instance methods of `ChunkParser` that are called when `ChunkParser` is initialised; this is all of
# `args_chunkparser_member_names` excluding those in `args_chunkparser_method_names`
args_init_call_members = [a for a in args_chunkparser_member_names if a not in args_chunkparser_method_names]

# Load the `FIlE_GRAMMAR` file
with open(FIlE_GRAMMAR, "r") as f:
    file_grammar_regex_patterns = "".join(f.readlines())


@pytest.mark.parametrize("test_input_filename", args_chunk_parser_initialisation)
class TestChunkParserInitialisation:

    def test_runs(self, test_input_filename):
        """Test that ChunkParser can be initialised."""

        # Attempt to create a `ChunkParser` object; if this fails for any reason, fail the test
        try:
            assert ChunkParser(test_input_filename)
        except Exception as e:
            pytest.fail(f"Raised exception {type(e)}:\n{str(e)}")

    @pytest.mark.parametrize("test_attribute", args_init_call_members)
    def test_calls_attributes(self, mocker, test_input_filename, test_attribute):
        """Test that ChunkParser, when initialised, calls various attributes."""

        # Patch the attribute of the `ChunkParser` class object
        patch_member = mocker.patch.object(ChunkParser, test_attribute)

        # Invoke the `ChunkParser` class
        _ = ChunkParser(test_input_filename)

        # Assert that `test_attribute` has been called once correctly; if `test_input_filename` is None,
        # then this should be `FILE_GRAMMAR`, otherwise it will be `test_input_filename`
        patch_member.assert_called_once_with(test_input_filename if test_input_filename else FIlE_GRAMMAR)

    def test_grammar_attrbute_correct(self, test_input_filename):
        """Check the grammar attribute of ChunkParser is correct."""

        # Invoke the `ChunkParser` class
        test_object = ChunkParser(test_input_filename)

        # Assert the `grammar` attribute is the same as `file_grammar_regex_patterns`
        assert test_object.grammar == file_grammar_regex_patterns


def test_grammar_attribute_same_for_none_or_file_grammar():
    """Check that the grammar attribute of ChunkParser is the same if using None or FILE_GRAMMAR as the input."""
    assert ChunkParser().grammar == ChunkParser(FIlE_GRAMMAR).grammar


@pytest.fixture
def temp_grammar_file(tmpdir_factory, temp_text):
    """Set up a pytest tmpdir_factory fixture to simulate a grammar regular expressions file."""

    # Create a temporary directory 'temp', and define a file path to a temporary text file 'temp_file.txt'
    temp_file_path = tmpdir_factory.mktemp("temp").join("temp_file.txt")

    # Write `temp_text` to `temp_file_path`
    _ = temp_file_path.write(temp_text)

    # Return the file path
    return temp_file_path


# Define text for the `temp_grammar_file` pytest fixture
args_temp_grammar_file_text = [
    "pronoun:\n{<DT><IN><PRP>}\n{<IN>?<PRP>}\nnoun_verb:\n{<IN>?<JJ.*>*<NN.*>+<HYPH>?<VBD|VBN|VBG><NN.*>*}",
    "rb:\n{<RB>+}\npunct:\n{<-RRB->|<-LRB->|<,>|<.>}"
]


@pytest.mark.parametrize("temp_text", args_temp_grammar_file_text)
class TestChunkParserInitialisationOtherFiles:

    def test_runs_for_other_grammar_files(self, temp_grammar_file):
        """Test that ChunkParser, when initialised, can use other grammar files correctly."""

        # Attempt to create a `ChunkParser` object; if this fails for any reason, fail the test
        try:
            assert ChunkParser(temp_grammar_file)
        except Exception as e:
            pytest.fail(f"Raised exception {type(e)}:\n{str(e)}")

    def test_grammar_attribute_correct(self, temp_grammar_file, temp_text):
        """Test the grammar attribute of ChunkParser, when using other grammar files, is as expected."""

        # Invoke the `ChunkParser` class
        test_object = ChunkParser(temp_grammar_file)

        # Assert the `grammar` attribute of `test_object` is the same as `temp_text`
        assert test_object.grammar == temp_text


@pytest.mark.parametrize("temp_text", args_temp_grammar_file_text)
def test__load_grammar_from_file_returns_correctly(temp_grammar_file, temp_text):
    """Test the _load_grammar_from_file static method of ChunkParser loads files correctly."""
    assert ChunkParser()._load_grammar_from_file(temp_grammar_file) == temp_text


# Define inputs for the `test__chunk_text_returns_correctly` pytest; the original sentences are 'Signed up for advice
# due to the ongoing COVID 19 outbreak', 'This is an example sentence.', and 'Here is text with some more descriptive
# elements; limited to 1 sentence only!'
args_methods_with_list_chunk_outputs_returns_correctly__chunk_text_inputs = [
    [("Signed", "VBN", "sign"), ("up", "RP", "up"), ("for", "IN", "for"), ("advice", "NN", "advice"),
     ("due", "IN", "due"), ("to", "IN", "to"), ("the", "DT", "the"), ("ongoing", "JJ", "ongoing"),
     ("COVID", "NNP", "COVID"), ("19", "CD", "19"), ("outbreak", "NN", "outbreak")],
    [("This", "DT", "this"), ("is", "VBZ", "be"), ("an", "DT", "an"), ("example", "NN", "example"),
     ("sentence", "NN", "sentence"), (".", ".", ".")],
    [("Here", "RB", "here"), ("is", "VBZ", "be"), ("text", "NN", "text"), ("with", "IN", "with"),
     ("some", "DT", "some"), ("more", "RBR", "more"), ("descriptive", "JJ", "descriptive"),
     ("elements", "NNS", "element"), (";", ",", ";"), ("limited", "VBD", "limit"), ("to", "IN", "to"),
     ("1", "CD", "1"), ("sentence", "NN", "sentence"), ("only", "RB", "only"), ("!", ".", "!")]
]

# Define expected outputs for the `test__chunk_text_returns_correctly` pytest
args_methods_with_list_chunk_outputs_returns_correctly__chunk_text_expected = [
    [Chunk("verb", [("Signed", "VBN", "sign"), ("up", "RP", "up"), ("for", "IN", "for")], [0, 1, 2]),
     Chunk("noun", [("advice", "NN", "advice")], [3]),
     Chunk("prep_noun", [("due", "IN", "due"), ("to", "IN", "to"), ("the", "DT", "the"),
                         ("ongoing", "JJ", "ongoing"), ("COVID", "NNP", "COVID"), ("19", "CD", "19"),
                         ("outbreak", "NN", "outbreak")], [4, 5, 6, 7, 8, 9, 10])],
    [Chunk("verb", [("is", "VBZ", "be")], [1]),
     Chunk("noun", [("an", "DT", "an"), ("example", "NN", "example")], [2, 3]),
     Chunk("noun", [("sentence", "NN", "sentence")], [4]),
     Chunk("punct", [(".", ".", ".")], [5])],
    [Chunk("rb", [("Here", "RB", "here")], [0]),
     Chunk("verb", [("is", "VBZ", "be")], [1]),
     Chunk("noun", [("text", "NN", "text")], [2]),
     Chunk("adjective", [("descriptive", "JJ", "descriptive")], [6]),
     Chunk("noun", [("elements", "NNS", "element")], [7]),
     Chunk("punct", [(";", ",", ";")], [8]),
     Chunk("verb", [("limited", "VBD", "limit"), ("to", "IN", "to")], [9, 10]),
     Chunk("noun", [("1", "CD", "1"), ("sentence", "NN", "sentence")], [11, 12]),
     Chunk("rb", [("only", "RB", "only")], [13]),
     Chunk("punct", [("!", ".", "!")], [14])]
]

# Create the test cases for the `__chunk_text` method
args_methods_with_list_chunk_outputs_returns_correctly__chunk_text = zip(
    args_methods_with_list_chunk_outputs_returns_correctly__chunk_text_inputs,
    args_methods_with_list_chunk_outputs_returns_correctly__chunk_text_expected
)

# Define input arguments to test the `_merge_adjacent_chunks` method, where there should be no changes. The test
# cases are for the sentences 'This is an example of a sentence', and 'Signed up for advice due to the ongoing COVID 19
# outbreak with specific concern about vulnerable people'; the second comprises three adjacent 'prep_noun' phrases
# that should not be merged
args_methods_with_list_chunk_outputs_returns_correctly__merge_adjacent_chunks_unchanged = [
    [Chunk("verb", [("is", "VBZ", "be")], [1]),
     Chunk("noun", [("an", "DT", "an"), ("example", "NN", "example")], [2, 3]),
     Chunk("prep_noun", [("of", "IN", "of"), ("a", "DT", "a"), ("sentence", "NN", "sentence")], [4, 5, 6]),
     Chunk("punct", [(".", ".", ".")], [7])],
    [Chunk("verb", [("Signed", "VBN", "sign"), ("up", "RP", "up"), ("for", "IN", "for")], [0, 1, 2]),
     Chunk("noun", [("advice", "NN", "advice")], [3]),
     Chunk("prep_noun", [("due", "IN", "due"), ("to", "IN", "to"), ("the", "DT", "the"),
                         ("ongoing", "JJ", "ongoing"), ("COVID", "NNP", "COVID"), ("19", "CD", "19"),
                         ("outbreak", "NN", "outbreak")], [4, 5, 6, 7, 8, 9, 10]),
     Chunk("prep_noun", [("with", "IN", "with"), ("specific", "JJ", "specific"), ("concern", "NN", "concern")],
           [11, 12, 13]),
     Chunk("prep_noun", [("about", "IN", "about"), ("vulnerable", "JJ", "vulnerable"), ("people", "NNS", "people")],
           [14, 15, 16])]
]

# Define input arguments to test the `_merge_adjacent_chunks` method, where there should be changes. The first test
# case is for the sentence 'This is an example sentence, the second test case is for the sentence 'I am going to go
# and test to see if this example is correct.', and the third test case is for the sentence 'I tried to signed up for
# advice due to the ongoing COVID 19 outbreak with specific concern about vulnerable people'. The first sentence has
# two adjacent noun clauses that should be merged, and the second sentence has two groups of adjacent verbs (three
# verbs in the first group, and two verbs in the second group), which should be merged together within their groups,
# and the third sentence has adjacent verbs that should be merged, but not adjacent `prep_noun`s
args_methods_with_list_chunk_outputs_returns_correctly__merge_adjacent_chunks_changed_inputs = [
    [Chunk("verb", [("is", "VBZ", "be")], [1]),
     Chunk("noun", [("an", "DT", "an"), ("example", "NN", "example")], [2, 3]),
     Chunk("noun", [("sentence", "NN", "sentence")], [4]),
     Chunk("punct", [(".", ".", ".")], [5])],
    [Chunk("pronoun", [("I", "PRP", "-PRON-")], [0]),
     Chunk("verb", [("am", "VBP", "be")], [1]),
     Chunk("verb", [("going", "VBG", "go"), ("to", "TO", "to")], [2, 3]),
     Chunk("verb", [("go", "VB", "go")], [4]),
     Chunk("cc", [("and", "CC", "and")], [5]),
     Chunk("verb", [("test", "VB", "test"), ("to", "TO", "to")], [6, 7]),
     Chunk("verb", [("see", "VB", "see"), ("if", "IN", "if")], [8, 9]),
     Chunk("noun", [("this", "DT", "this"), ("example", "NN", "example")], [10, 11]),
     Chunk("verb", [("is", "VBZ", "be")], [12]),
     Chunk("adjective", [("correct", "JJ", "correct")], [13]),
     Chunk("punct", [(".", ".", ".")], [14])],
    [Chunk("pronoun", [("I", "PRP", "-PRON-")], [0]),
     Chunk("verb", [("tried", "VBD", "try"), ("to", "TO", "to")], [1, 2]),
     Chunk("verb", [("signed", "VBN", "sign"), ("up", "RP", "up"), ("for", "IN", "for")], [3, 4, 5]),
     Chunk("noun", [("advice", "NN", "advice")], [6]),
     Chunk("prep_noun", [("due", "IN", "due"), ("to", "IN", "to"), ("the", "DT", "the"),
                         ("ongoing", "JJ", "ongoing"), ("COVID", "NNP", "COVID"), ("19", "CD", "19"),
                         ("outbreak", "NN", "outbreak")], [7, 8, 9, 10, 11, 12, 13]),
     Chunk("prep_noun", [("with", "IN", "with"), ("specific", "JJ", "specific"), ("concern", "NN", "concern")],
           [14, 15, 16]),
     Chunk("prep_noun", [("about", "IN", "about"), ("vulnerable", "JJ", "vulnerable"), ("people", "NNS", "people")],
           [17, 18, 19])]
]

# The expected output from the `_merge_adjacent_chunks` method run on
# `args_methods_with_list_chunk_outputs_returns_correctly__merge_adjacent_chunks_changed_inputs`. Note the first
# sentence has combined the two adjacent nouns, and the second sentence has combined the first group of three
# adjacent verbs, and the second group of two adjacent verbs
args_methods_with_list_chunk_outputs_returns_correctly__merge_adjacent_chunks_changed_expected = [
    [Chunk("verb", [("is", "VBZ", "be")], [1]),
     Chunk("noun", [("an", "DT", "an"), ("example", "NN", "example"), ("sentence", "NN", "sentence")], [2, 3, 4]),
     Chunk("punct", [(".", ".", ".")], [5])],
    [Chunk("pronoun", [("I", "PRP", "-PRON-")], [0]),
     Chunk("verb", [("am", "VBP", "be"), ("going", "VBG", "go"), ("to", "TO", "to"), ("go", "VB", "go")], [1, 2, 3, 4]),
     Chunk("cc", [("and", "CC", "and")], [5]),
     Chunk("verb", [("test", "VB", "test"), ("to", "TO", "to"), ("see", "VB", "see"), ("if", "IN", "if")],
           [6, 7, 8, 9]),
     Chunk("noun", [("this", "DT", "this"), ("example", "NN", "example")], [10, 11]),
     Chunk("verb", [("is", "VBZ", "be")], [12]),
     Chunk("adjective", [("correct", "JJ", "correct")], [13]),
     Chunk("punct", [(".", ".", ".")], [14])],
    [Chunk("pronoun", [("I", "PRP", "-PRON-")], [0]),
     Chunk("verb", [("tried", "VBD", "try"), ("to", "TO", "to"), ("signed", "VBN", "sign"), ("up", "RP", "up"),
                    ("for", "IN", "for")], [1, 2, 3, 4, 5]),
     Chunk("noun", [("advice", "NN", "advice")], [6]),
     Chunk("prep_noun", [("due", "IN", "due"), ("to", "IN", "to"), ("the", "DT", "the"),
                         ("ongoing", "JJ", "ongoing"), ("COVID", "NNP", "COVID"), ("19", "CD", "19"),
                         ("outbreak", "NN", "outbreak")], [7, 8, 9, 10, 11, 12, 13]),
     Chunk("prep_noun", [("with", "IN", "with"), ("specific", "JJ", "specific"), ("concern", "NN", "concern")],
           [14, 15, 16]),
     Chunk("prep_noun", [("about", "IN", "about"), ("vulnerable", "JJ", "vulnerable"), ("people", "NNS", "people")],
           [17, 18, 19])]
]

# Create the test cases for the `_merge_adjacent_chunks` method where merging should occur
args_methods_with_list_chunk_outputs_returns_correctly__merge_adjacent_chunks_changed = zip(
    args_methods_with_list_chunk_outputs_returns_correctly__merge_adjacent_chunks_changed_inputs,
    args_methods_with_list_chunk_outputs_returns_correctly__merge_adjacent_chunks_changed_expected
)

# Define the test cases for the `test_methods_with_list_chunk_outputs_returns_correctly` pytest
args_methods_with_list_chunk_outputs_returns_correctly = [
    *[("_chunk_text", i, e) for i, e in args_methods_with_list_chunk_outputs_returns_correctly__chunk_text],
    *[("_merge_adjacent_chunks", a, a) for a in
      args_methods_with_list_chunk_outputs_returns_correctly__merge_adjacent_chunks_unchanged],
    *[("_merge_adjacent_chunks", i, e) for i, e in
      args_methods_with_list_chunk_outputs_returns_correctly__merge_adjacent_chunks_changed],
]


@pytest.mark.parametrize("test_method, test_input, test_expected",
                         args_methods_with_list_chunk_outputs_returns_correctly)
def test_methods_with_list_chunk_outputs_returns_correctly(test_method, test_input, test_expected):
    """Test the instance methods of ChunkParser that return lists of Chunk objects are correct."""

    # Invoke the `test_method` method of `ChunkParser`
    test_output = getattr(ChunkParser(), test_method)(test_input)

    # Assert each element in `test_output` is a Chunk object, and has the same `__dict__` as the equivalent element
    # in `test_expected`
    for e, o in zip(test_expected, test_output):
        assert isinstance(o, Chunk)
        assert o.__dict__ == e.__dict__


# Define input arguments for the `TestExtractPhraseMethod` test class; the first text is 'I am going to go and test to
# see if this example is correct.', the second text is 'If merge is True, then there will be some changes with this
# example!', the third text is a combination of the first and second text into a two-sentence text, and the fourth
# text is 'I tried to signed up for advice due to the ongoing COVID 19 outbreak with specific concern about vulnerable
# people. I could not!'
args_extract_phrase_method_inputs = [
    [[("I", "PRP", "-PRON-"), ("am", "VBP", "be"), ("going", "VBG", "go"), ("to", "TO", "to"), ("go", "VB", "go"),
     ("and", "CC", "and"), ("test", "VB", "test"), ("to", "TO", "to"), ("see", "VB", "see"), ("if", "IN", "if"),
     ("this", "DT", "this"), ("example", "NN", "example"), ("is", "VBZ", "be"), ("correct", "JJ", "correct"),
     (".", ".", ".")]],
    [[("If", "IN", "if"), ("merge", "NN", "merge"), ("is", "VBZ", "be"), ("True", "JJ", "true"), (",", ",", ","),
     ("then", "RB", "then"), ("there", "EX", "there"), ("will", "MD", "will"), ("be", "VB", "be"),
     ("some", "DT", "some"), ("changes", "NNS", "change"), ("with", "IN", "with"), ("this", "DT", "this"),
     ("example", "NN", "example"), ("!", ".", "!")]],
    [[("I", "PRP", "-PRON-"), ("am", "VBP", "be"), ("going", "VBG", "go"), ("to", "TO", "to"), ("go", "VB", "go"),
     ("and", "CC", "and"), ("test", "VB", "test"), ("to", "TO", "to"), ("see", "VB", "see"), ("if", "IN", "if"),
     ("this", "DT", "this"), ("example", "NN", "example"), ("is", "VBZ", "be"), ("correct", "JJ", "correct"),
     (".", ".", ".")],
     [("If", "IN", "if"), ("merge", "NN", "merge"), ("is", "VBZ", "be"), ("True", "JJ", "true"), (",", ",", ","),
      ("then", "RB", "then"), ("there", "EX", "there"), ("will", "MD", "will"), ("be", "VB", "be"),
      ("some", "DT", "some"), ("changes", "NNS", "change"), ("with", "IN", "with"), ("this", "DT", "this"),
      ("example", "NN", "example"), ("!", ".", "!")]],
    [[("I", "PRP", "-PRON-"), ("tried", "VBD", "try"), ("to", "TO", "to"), ("signed", "VBN", "sign"),
      ("up", "RP", "up"), ("for", "IN", "for"), ("advice", "NN", "advice"), ("due", "IN", "due"), ("to", "IN", "to"),
      ("the", "DT", "the"), ("ongoing", "JJ", "ongoing"), ("COVID", "NNP", "COVID"), ("19", "CD", "19"),
      ("outbreak", "NN", "outbreak"), ("with", "IN", "with"), ("specific", "JJ", "specific"),
      ("concern", "NN", "concern"), ("about", "IN", "about"), ("vulnerable", "JJ", "vulnerable"),
      ("people", "NNS", "people")],
     [("I", "PRP", "-PRON-"), ("could", "MD", "could"), ("not", "RB", "not"), ("!", ".", "!")]]
]

# Define the expected outputs from the `extract_phrase` method, where `merge_inplace = False`. In this case,
# adjacent `Chunk` objects with the same label (except 'prep_noun') are not combined together
args_extract_phrase_method_expected_unchanged = [
    [[Chunk("pronoun", [("I", "PRP", "-PRON-")], [0]),
      Chunk("verb", [("am", "VBP", "be")], [1]),
      Chunk("verb", [("going", "VBG", "go"), ("to", "TO", "to")], [2, 3]),
      Chunk("verb", [("go", "VB", "go")], [4]),
      Chunk("cc", [("and", "CC", "and")], [5]),
      Chunk("verb", [("test", "VB", "test"), ("to", "TO", "to")], [6, 7]),
      Chunk("verb", [("see", "VB", "see"), ("if", "IN", "if")], [8, 9]),
      Chunk("noun", [("this", "DT", "this"), ("example", "NN", "example")], [10, 11]),
      Chunk("verb", [("is", "VBZ", "be")], [12]),
      Chunk("adjective", [("correct", "JJ", "correct")], [13]),
      Chunk("punct", [(".", ".", ".")], [14])]],
    [[Chunk("prep_noun", [("If", "IN", "if"), ("merge", "NN", "merge")], [0, 1]),
      Chunk("verb", [("is", "VBZ", "be")], [2]),
      Chunk("adjective", [("True", "JJ", "true")], [3]),
      Chunk("punct", [(",", ",", ",")], [4]),
      Chunk("rb", [("then", "RB", "then")], [5]),
      Chunk("verb", [("will", "MD", "will")], [7]),
      Chunk("verb", [("be", "VB", "be")], [8]),
      Chunk("noun", [("some", "DT", "some"), ("changes", "NNS", "change")], [9, 10]),
      Chunk("prep_noun", [("with", "IN", "with"), ("this", "DT", "this"), ("example", "NN", "example")], [11, 12, 13]),
      Chunk("punct", [("!", ".", "!")], [14])]],
    [[Chunk("pronoun", [("I", "PRP", "-PRON-")], [0]),
      Chunk("verb", [("am", "VBP", "be")], [1]),
      Chunk("verb", [("going", "VBG", "go"), ("to", "TO", "to")], [2, 3]),
      Chunk("verb", [("go", "VB", "go")], [4]),
      Chunk("cc", [("and", "CC", "and")], [5]),
      Chunk("verb", [("test", "VB", "test"), ("to", "TO", "to")], [6, 7]),
      Chunk("verb", [("see", "VB", "see"), ("if", "IN", "if")], [8, 9]),
      Chunk("noun", [("this", "DT", "this"), ("example", "NN", "example")], [10, 11]),
      Chunk("verb", [("is", "VBZ", "be")], [12]),
      Chunk("adjective", [("correct", "JJ", "correct")], [13]),
      Chunk("punct", [(".", ".", ".")], [14])],
     [Chunk("prep_noun", [("If", "IN", "if"), ("merge", "NN", "merge")], [0, 1]),
      Chunk("verb", [("is", "VBZ", "be")], [2]),
      Chunk("adjective", [("True", "JJ", "true")], [3]),
      Chunk("punct", [(",", ",", ",")], [4]),
      Chunk("rb", [("then", "RB", "then")], [5]),
      Chunk("verb", [("will", "MD", "will")], [7]),
      Chunk("verb", [("be", "VB", "be")], [8]),
      Chunk("noun", [("some", "DT", "some"), ("changes", "NNS", "change")], [9, 10]),
      Chunk("prep_noun", [("with", "IN", "with"), ("this", "DT", "this"), ("example", "NN", "example")], [11, 12, 13]),
      Chunk("punct", [("!", ".", "!")], [14])]],
    [[Chunk("pronoun", [("I", "PRP", "-PRON-")], [0]),
      Chunk("verb", [("tried", "VBD", "try"), ("to", "TO", "to")], [1, 2]),
      Chunk("verb", [("signed", "VBN", "sign"), ("up", "RP", "up"), ("for", "IN", "for")], [3, 4, 5]),
      Chunk("noun", [("advice", "NN", "advice")], [6]),
      Chunk("prep_noun", [("due", "IN", "due"), ("to", "IN", "to"), ("the", "DT", "the"),
                          ("ongoing", "JJ", "ongoing"), ("COVID", "NNP", "COVID"), ("19", "CD", "19"),
                          ("outbreak", "NN", "outbreak")], [7, 8, 9, 10, 11, 12, 13]),
      Chunk("prep_noun", [("with", "IN", "with"), ("specific", "JJ", "specific"), ("concern", "NN", "concern")],
            [14, 15, 16]),
      Chunk("prep_noun", [("about", "IN", "about"), ("vulnerable", "JJ", "vulnerable"), ("people", "NNS", "people")],
            [17, 18, 19]),
      Chunk("punct", [(".", ".", ".")], [20])],
     [Chunk("pronoun", [("I", "PRP", "-PRON-")], [0]),
      Chunk("verb", [("could", "MD", "could"), ("not", "RB", "not")], [1, 2]),
      Chunk("punct", [("!", ".", "!")], [3])]]
]

# Define the expected outputs from the `extract_phrase` method, where `merge_inplace = False`. In this case,
# adjacent `Chunk` objects with the same label (except 'prep_noun') are combined together. In the first text, two
# groups of verbs (one group of three, and another of two) are combined together. In the second text, a group of two
# verbs are combined. In the third text, three groups of verbs (a group of three, and two groups of two) are
# combined. In the fourth text, two verbs are combined, but the group of three 'prep_noun's are unchanged
args_extract_phrase_method_expected_changed = [
    [[Chunk("pronoun", [("I", "PRP", "-PRON-")], [0]),
      Chunk("verb", [("am", "VBP", "be"), ("going", "VBG", "go"), ("to", "TO", "to"), ("go", "VB", "go")],
            [1, 2, 3, 4]),
      Chunk("cc", [("and", "CC", "and")], [5]),
      Chunk("verb", [("test", "VB", "test"), ("to", "TO", "to"), ("see", "VB", "see"), ("if", "IN", "if")],
            [6, 7, 8, 9]),
      Chunk("noun", [("this", "DT", "this"), ("example", "NN", "example")], [10, 11]),
      Chunk("verb", [("is", "VBZ", "be")], [12]),
      Chunk("adjective", [("correct", "JJ", "correct")], [13]),
      Chunk("punct", [(".", ".", ".")], [14])]],
    [[Chunk("prep_noun", [("If", "IN", "if"), ("merge", "NN", "merge")], [0, 1]),
      Chunk("verb", [("is", "VBZ", "be")], [2]),
      Chunk("adjective", [("True", "JJ", "true")], [3]),
      Chunk("punct", [(",", ",", ",")], [4]),
      Chunk("rb", [("then", "RB", "then")], [5]),
      Chunk("verb", [("will", "MD", "will"), ("be", "VB", "be")], [7, 8]),
      Chunk("noun", [("some", "DT", "some"), ("changes", "NNS", "change")], [9, 10]),
      Chunk("prep_noun", [("with", "IN", "with"), ("this", "DT", "this"), ("example", "NN", "example")], [11, 12, 13]),
      Chunk("punct", [("!", ".", "!")], [14])]],
    [[Chunk("pronoun", [("I", "PRP", "-PRON-")], [0]),
      Chunk("verb", [("am", "VBP", "be"), ("going", "VBG", "go"), ("to", "TO", "to"), ("go", "VB", "go")],
            [1, 2, 3, 4]),
      Chunk("cc", [("and", "CC", "and")], [5]),
      Chunk("verb", [("test", "VB", "test"), ("to", "TO", "to"), ("see", "VB", "see"), ("if", "IN", "if")],
            [6, 7, 8, 9]),
      Chunk("noun", [("this", "DT", "this"), ("example", "NN", "example")], [10, 11]),
      Chunk("verb", [("is", "VBZ", "be")], [12]),
      Chunk("adjective", [("correct", "JJ", "correct")], [13]),
      Chunk("punct", [(".", ".", ".")], [14])],
     [Chunk("prep_noun", [("If", "IN", "if"), ("merge", "NN", "merge")], [0, 1]),
      Chunk("verb", [("is", "VBZ", "be")], [2]),
      Chunk("adjective", [("True", "JJ", "true")], [3]),
      Chunk("punct", [(",", ",", ",")], [4]),
      Chunk("rb", [("then", "RB", "then")], [5]),
      Chunk("verb", [("will", "MD", "will"), ("be", "VB", "be")], [7, 8]),
      Chunk("noun", [("some", "DT", "some"), ("changes", "NNS", "change")], [9, 10]),
      Chunk("prep_noun", [("with", "IN", "with"), ("this", "DT", "this"), ("example", "NN", "example")], [11, 12, 13]),
      Chunk("punct", [("!", ".", "!")], [14])]],
    [[Chunk("pronoun", [("I", "PRP", "-PRON-")], [0]),
      Chunk("verb", [("tried", "VBD", "try"), ("to", "TO", "to"), ("signed", "VBN", "sign"), ("up", "RP", "up"),
                     ("for", "IN", "for")], [1, 2, 3, 4, 5]),
      Chunk("noun", [("advice", "NN", "advice")], [6]),
      Chunk("prep_noun", [("due", "IN", "due"), ("to", "IN", "to"), ("the", "DT", "the"),
                          ("ongoing", "JJ", "ongoing"), ("COVID", "NNP", "COVID"), ("19", "CD", "19"),
                          ("outbreak", "NN", "outbreak")], [7, 8, 9, 10, 11, 12, 13]),
      Chunk("prep_noun", [("with", "IN", "with"), ("specific", "JJ", "specific"), ("concern", "NN", "concern")],
            [14, 15, 16]),
      Chunk("prep_noun", [("about", "IN", "about"), ("vulnerable", "JJ", "vulnerable"), ("people", "NNS", "people")],
            [17, 18, 19]),
      Chunk("punct", [(".", ".", ".")], [20])],
     [Chunk("pronoun", [("I", "PRP", "-PRON-")], [0]),
      Chunk("verb", [("could", "MD", "could"), ("not", "RB", "not")], [1, 2]),
      Chunk("punct", [("!", ".", "!")], [3])]]
]

# Define test cases for the `TestExtractPhraseMethod` test class
args_extract_phrase_method = [
    *[(i, None, e) for i, e in zip(args_extract_phrase_method_inputs, args_extract_phrase_method_expected_unchanged)],
    *[(i, False, e) for i, e in zip(args_extract_phrase_method_inputs, args_extract_phrase_method_expected_unchanged)],
    *[(i, True, e) for i, e in zip(args_extract_phrase_method_inputs, args_extract_phrase_method_expected_changed)]
]


@pytest.mark.parametrize("test_input_sentences, test_input_merge_inplace, test_expected", args_extract_phrase_method)
def test_extract_phrase_returns_correctly(test_input_sentences, test_input_merge_inplace, test_expected):
    """Test that the extract_phrase method returns correctly."""

    # Invoke the `extract_phrase` method of `ChunkParser`
    test_output = ChunkParser().extract_phrase(test_input_sentences, test_input_merge_inplace)

    # Assert each element within the nested `test_output` is a Chunk object, and has the same `__dict__` as
    # the equivalent nested element in `test_expected`
    for e_list, o_list in zip(test_expected, test_output):
        for e, o in zip(e_list, o_list):
            assert isinstance(o, Chunk)
            assert o.__dict__ == e.__dict__


@pytest.fixture
def resource__chunk_text_patch(mocker):
    """Patch the _chunk_text method of ChunkParser."""
    return mocker.patch.object(ChunkParser, "_chunk_text")


@pytest.fixture
def resource__merge_adjacent_chunks_patch(mocker):
    """Patch the _merge_adjacent_chunks method of ChunkParser."""
    return mocker.patch.object(ChunkParser, "_merge_adjacent_chunks")


@pytest.mark.parametrize("test_input_sentences, test_input_merge_inplace", [a[:-1] for a in args_extract_phrase_method])
class TestExtractPhraseCallsMethods:

    def test__chunk_text_call_count(self, resource__chunk_text_patch, test_input_sentences, test_input_merge_inplace):
        """Test that the extract_phrase method calls _chunk_text the correct number of times."""

        # Invoke the `extract_phrase` method of `ChunkParser`
        _ = ChunkParser().extract_phrase(test_input_sentences, test_input_merge_inplace)

        # Assert `_chunk_text` is called the correct number of times
        assert resource__chunk_text_patch.call_count == len(test_input_sentences)

    def test__chunk_text_called_correctly(self, mocker, resource__chunk_text_patch, test_input_sentences,
                                          test_input_merge_inplace):
        """Test that the extract_phrase method calls _chunk_text with the correct input arguments."""

        # Invoke the `extract_phrase` method of `ChunkParser`
        _ = ChunkParser().extract_phrase(test_input_sentences, test_input_merge_inplace)

        # Get the argument calls to the `_chunk_text` method, and assert they are as expected
        assert resource__chunk_text_patch.call_args_list == [mocker.call(s) for s in test_input_sentences]

    def test__merge_adjacent_chunks_call_count(self, resource__merge_adjacent_chunks_patch, test_input_sentences,
                                               test_input_merge_inplace):
        """Test that the extract_phrase method calls _merge_adjacent_chunks the correct number of times."""

        # Invoke the `extract_phrase` method of `ChunkParser`
        _ = ChunkParser().extract_phrase(test_input_sentences, test_input_merge_inplace)

        # If `test_input_merge_inplace` is True, assert `_merge_adjacent_chunks` is called the correct number of
        # times, otherwise assert it is not called at all
        if test_input_merge_inplace:
            assert resource__merge_adjacent_chunks_patch.call_count == len(test_input_sentences)
        else:
            assert not resource__merge_adjacent_chunks_patch.called

    def test__merge_adjacent_chunks_called_correctly(self, mocker, resource__chunk_text_patch,
                                                     resource__merge_adjacent_chunks_patch, test_input_sentences,
                                                     test_input_merge_inplace):
        """Test that the extract_phrase method calls _merge_adjacent_chunks with the correct input arguments."""

        # Invoke the `extract_phrase` method of `ChunkParser`
        _ = ChunkParser().extract_phrase(test_input_sentences, test_input_merge_inplace)

        # If `test_input_merge_inplace` is True, assert `_merge_adjacent_chunks` is called with the correct arguments,
        # otherwise assert it is not called at all. If True, the correct arguments is the return value from the
        # `_chunk_text` patch duplicated by the number of sentences to replicate the `chunks` variable in the
        # `extract_phrase` method
        if test_input_merge_inplace:
            test_expected = [mocker.call(resource__chunk_text_patch.return_value)] * len(test_input_sentences)
            assert resource__merge_adjacent_chunks_patch.call_args_list == test_expected
        else:
            assert not resource__merge_adjacent_chunks_patch.called

    def test_returns_method_output(self, resource__chunk_text_patch, resource__merge_adjacent_chunks_patch,
                                   test_input_sentences, test_input_merge_inplace):
        """Test that the return from the extract_phrase method is from either _chunk_text or _merge_adjacent_chunks."""

        # Invoke the `extract_phrase` method of `ChunkParser`
        test_output = ChunkParser().extract_phrase(test_input_sentences, test_input_merge_inplace)

        # If `test_input_merge_inplace`, assert `test_output` is the return value from the
        # `_merge_adjacent_chunks` method, otherwise assert it is the return value from the `_chunk_text` method
        if test_input_merge_inplace:
            assert test_output == [resource__merge_adjacent_chunks_patch.return_value] * len(test_input_sentences)
        else:
            assert test_output == [resource__chunk_text_patch.return_value] * len(test_input_sentences)
