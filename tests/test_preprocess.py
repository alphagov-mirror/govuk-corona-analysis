from src.make_feedback_tool_data.preprocess import PreProcess
from src.make_feedback_tool_data.preprocess import PII_FILTERED
import pytest


# Test cases for the `test_method_returns_correctly` pytest on the `split_sentences` static method
args_method_returns_correctly_split_sentences = [
    ("", []),
    ("This is a single sentence.", ["This is a single sentence."]),
    ("Hello? World!", ["Hello?", "World!"]),
    ("One. Two. Three. Four.", ["One.", "Two.", "Three.", "Four."])
]

# Test cases for the `test_method_returns_correctly` pytest on the `replace_pii_regex` static method
args_method_returns_correctly_replace_pii_regex = [
    ("", ""),
    ("No PII here!", "No PII here!"),
    *[(f"[{p}]", "") for p in PII_FILTERED],
    ("Some text [{}] with PII".format("][".join(PII_FILTERED)), "Some text  with PII")
]

# Test cases for the `test_method_returns_correctly` pytest on the `part_of_speech_tag` class method
args_method_returns_correctly_part_of_speech_tag = [
    ("", []),
    ("This a single sentence.", [[("This", "DT", "this"), ("a", "DT", "a"), ("single", "JJ", "single"),
                                  ("sentence", "NN", "sentence"), (".", ".", ".")]]),
    ("This is a test with punctuation’. this is another sentence.", [[("This", "DT", "this"), ("is", "VBZ", "be"),
                                                                      ("a", "DT", "a"), ("test", "NN", "test"),
                                                                      ("with", "IN", "with"),
                                                                      ("punctuation", "NN", "punctuation"),
                                                                      ("’", "''", "'"), (".", ".", ".")],
                                                                     [("this", "DT", "this"), ("is", "VBZ", "be"),
                                                                      ("another", "DT", "another"),
                                                                      ("sentence", "NN", "sentence"),
                                                                      (".", ".", ".")]])
]

# Test cases for the `test_method_returns_correctly` pytest on the `detect_language` static method; last two test
# cases are adapted from the Wikipedia page on the Greek, English, and Chinese versions of https://www.wikipedia.org/
args_method_returns_correctly_detect_language = [
    ("", "un"),
    ("-", "-"),
    ("These words are in English.", "en"),
    ("Cet mots sont en français.", "fr"),
    ("This is in English, mais la majorité des mots sont en français dans cette phrase", "fr"),
    ("Η Βικιπαίδεια είναι διεθνής, παγκόσμια, ψηφιακή, διαδικτυακή, ελεύθερου περιεχομένου, εγκυκλοπαίδεια, "
     "που βασίζεται σε ένα μοντέλο ανοικτό στη σύνταξη του περιεχομένου της. It is the largest and most popular "
     "general reference work on the World Wide Web, and is one of the 20 most popular websites ranked by Alexa, "
     "as of March 2020.", "el"),
    ("維基百科 是维基媒体基金会运营的一个多语言的線上百科全書，并以创建和维护作为开放式协同合作项目，特点是自由內容、自由编辑、自由版权", "zh")
]

# Test cases for the `test_method_returns_correctly` pytest on the `compute_combinations` class method
args_method_returns_correctly_compute_combinations = [
    ([["A", "B", "C", "D"]], 1, [["A"], ["B"], ["C"], ["D"]]),
    ([["A", "B", "C", "D"]], 2, [["A", "B"], ["B", "C"], ["C", "D"]]),
    ([["A", "B", "C", "D"]], 3, [["A", "B", "C"], ["B", "C", "D"]]),
    ([[1, 2, 3, 4]], 1, [[1], [2], [3], [4]]),
    ([[1, 2, 3, 4]], 2, [[1, 2], [2, 3], [3, 4]]),
    ([[1, 2, 3, 4]], 3, [[1, 2, 3], [2, 3, 4]]),
]

# Test cases for the `test_method_returns_correctly` pytest on the `get_user_group` static method
args_method_returns_correctly_get_user_group = [
    ("", "", ""),
    ("is", "a random sentence", ""),  # 'This is a random sentence'
    ("am", "a key worker", "key worker"),  # 'I am a key worker'
    ("am", "the key worker", "key worker"),  # 'I am the key worker in our household'
    ("'m", "a key worker", "key worker"),  # 'I'm a key worker'
    ("’m", "a key worker", "key worker"),  # 'I’m a key worker'
    ("feel", "alone", "alone"),  # 'I feel alone'
    ("have been", "tested", "tested")  # 'I have been tested'
]

# Test cases for the `test_method_returns_correctly` pytest on the `resolve_function` class method. The test cases
# leverage `args_method_returns_correctly_get_user_group` to build the required `phrase_mention` input argument of
# the `resolve_function` class method. The first two sets of test cases check single phrase mentions where the first
# POS tag is a verb or otherwise. The second two sets of test cases check multiple phrase mentions where the first POS
# tag is a verb or otherwise. The third two sets of test cases check multiple (two) phrase mentions where the phrases
# have alternate first POS tags (verb or otherwise), and vice versa
args_method_returns_correctly_resolve_function = [
    (*[([(("verb", "other"), " ".join(a[:1]), "theme", a[:-1])], [a[-1]] if a[-1] else []) for a in
       args_method_returns_correctly_get_user_group]),
    (*[([(("other", "other"), " ".join(a[:1]), "theme", a[:-1])], []) for a in
       args_method_returns_correctly_get_user_group]),
    ([(("verb", "other"), " ".join(a[:1]), "theme", a[:-1]) for a in args_method_returns_correctly_get_user_group],
     [a[-1] for a in args_method_returns_correctly_get_user_group if a[-1]]),
    ([(("other", "other"), " ".join(a[:1]), "theme", a[:-1]) for a in args_method_returns_correctly_get_user_group],
     []),
    ([(("verb", "other"), "is a random sentence", "theme", ("is", "a random sentence")),
      (("other", "other"), "am a key worker", "theme", ("am", "a key worker"))], []),
    ([(("other", "other"), "is a random sentence", "theme", ("is", "a random sentence")),
      (("verb", "other"), "am a key worker", "theme", ("am", "a key worker"))], ["key worker"]),
]

# Test cases for the `test_method_returns_correctly` pytest on the `find_needle` static method
args_method_returns_correctly_find_needle = [
    ("", "", {"": ""}),
    ("a random needle", "will not be found in this hay", {"a random needle": None}),
    ("told register as a vulnerable person",
     "told to register as a vulnerable person for delivery service for on line shopping",
     {"told register as a vulnerable person": "told to register as a vulnerable person"})
]

# Compile the test cases for the `test_method_returns_correctly` pytest alongside the methods for testing. For each
# test case, the tuple comprises the `PreProcess` method (first element of the tuple), all the input arguments (at
# least one element), and the expected argument (last element). This lets the `test_method_returns_correctly` pytest
# handle methods with any number of input arguments
args_test_method_returns_correctly = [
    *[(PreProcess.split_sentences, a[:-1], a[-1]) for a in args_method_returns_correctly_split_sentences],
    *[(PreProcess.replace_pii_regex, a[:-1], a[-1]) for a in args_method_returns_correctly_replace_pii_regex],
    *[(PreProcess.part_of_speech_tag, a[:-1], a[-1]) for a in args_method_returns_correctly_part_of_speech_tag],
    *[(PreProcess.detect_language, a[:-1], a[-1]) for a in args_method_returns_correctly_detect_language],
    *[(PreProcess.compute_combinations, a[:-1], a[-1]) for a in args_method_returns_correctly_compute_combinations],
    *[(PreProcess.get_user_group, a[:-1], a[-1]) for a in args_method_returns_correctly_get_user_group],
    *[(PreProcess.resolve_function, a[:-1], a[-1]) for a in args_method_returns_correctly_resolve_function],
    *[(PreProcess.find_needle, a[:-1], a[-1]) for a in args_method_returns_correctly_find_needle]
]


@pytest.mark.parametrize("test_method, test_input_args, test_expected", args_test_method_returns_correctly)
def test_method_returns_correctly(test_method, test_input_args, test_expected):
    """Test a method from the PreProcess class returns correctly."""
    assert test_method(*test_input_args) == test_expected


# Test cases for the `test_part_of_speech_tag_calls_split_sentences` pytest
args_part_of_speech_tag_calls_split_sentences = [a[1] for a in args_method_returns_correctly_part_of_speech_tag]


@pytest.mark.parametrize("test_input", args_part_of_speech_tag_calls_split_sentences)
def test_part_of_speech_tag_calls_split_sentences(mocker, test_input):
    """Test the part_of_speech_tag class method calls the split_sentences static method."""

    # Patch the `split_sentences` static method
    patch_split_sentences = mocker.patch.object(PreProcess, "split_sentences")

    # Call the `part_of_speech_tag` class method
    _ = PreProcess.part_of_speech_tag(test_input)

    # Assert `patch_split_sentences` was called once with the correct arguments
    patch_split_sentences.assert_called_once_with(test_input)


# Test cases for the `test_detect_language_returns_error_string` pytest
args_test_detect_language_returns_error_string = [
    123905234091,
    ["Some string within a sentence."]
]


@pytest.mark.parametrize("test_input", args_test_detect_language_returns_error_string)
def test_detect_language_returns_error_string(test_input):
    """Test the detect_language static method returns an error string if an exception is raised."""

    # Attempt to call the `detect_language` static method, and assert it should return an error string; if an exception
    # is raised, fail this test
    try:
        assert PreProcess.detect_language(test_input).startswith(f"[ERROR] {test_input} ")
    except Exception as e:
        pytest.fail(f"Raised exception {type(e)}:\n{str(e)}")


# Test cases for the `test_resolve_function_calls_get_user_group_correctly` pytest; this is based off the
# `args_method_returns_correctly_resolve_function` test cases
args_resolve_function_calls_get_user_group_correctly = [a[0] for a in args_method_returns_correctly_resolve_function]


@pytest.fixture
def resource_resolve_function_calls_get_user_group(mocker):
    """Patch the get_user_group static method used for the TestResolveFunctionCallsGetUserGroup test class."""
    return mocker.patch.object(PreProcess, "get_user_group")


@pytest.mark.parametrize("test_input", args_resolve_function_calls_get_user_group_correctly)
class TestResolveFunctionCallsGetUserGroup:

    def test_call_count_correct(self, resource_resolve_function_calls_get_user_group, test_input):
        """Test the get_user_group static method is called the correct number of times."""

        # Call the `resolve_function` class method
        _ = PreProcess.resolve_function(test_input)

        # Define the expected call count of the `get_user_group` static method
        test_expected_call_count = sum(1 for a in test_input if a[0][0] == "verb")

        # Assert the `get_user_group` static method is called the correct number of times
        assert resource_resolve_function_calls_get_user_group.call_count == test_expected_call_count

    def test_called_correctly(self, mocker, resource_resolve_function_calls_get_user_group, test_input):
        """Test the the get_user_group static method is called with the correct arguments."""

        # Call the `resolve_function` class method
        _ = PreProcess.resolve_function(test_input)

        # Define the expected input arguments to the `get_user_group` static method
        test_expected_call_args = [mocker.call(*a[-1]) for a in test_input if a[0][0] == "verb"]

        # Assert the `get_user_group` static method is called with the correct arguments
        assert resource_resolve_function_calls_get_user_group.call_args_list == test_expected_call_args
