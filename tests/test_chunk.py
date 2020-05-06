from src.make_feedback_tool_data.chunk import Chunk
import inspect
import pytest


# Compile a list of the attribute, and instance method names in the `Chunk` class
args_chunk_member_names = ["text", "lemma", "tagable_words", "important_word", "important_lemma"]

# Define a list of method names in `Chunk`
args_chunk_method_names = ["tagable_words"]


@pytest.mark.parametrize("test_method_name", args_chunk_method_names)
def test_methods_in_members(test_method_name):
    """Test that a method exists in args_chunk_member_names."""
    assert test_method_name in args_chunk_member_names


def test_chunk_has_members():
    """Test that Chunk has all the member objects in args_chunk_member_names."""

    # Get the member objects of the `Chunk` class
    test_output = [m[0] for m in inspect.getmembers(Chunk, inspect.isroutine) if not m[0].startswith("__")]

    # Assert that all the member objects exist
    assert set(test_output) == set(args_chunk_member_names)


# Define some inputs for the tests in the script; the original sentence is 'Signed up for advice due to the COVID 19
# ongoing outbreak'
args_inputs = [
    ("verb", [("Signed", "VBN", "sign"), ("up", "RP", "up"), ("for", "IN", "for")], [0, 1, 2]),
    ("noun", [("advice", "NN", "advice")], [3]),
    ("prep_noun", [("to", "IN", "to"), ("the", "DT", "the"), ('ongoing', 'JJ', 'ongoing'), ("COVID", "NNP", "COVID"),
                   ("19", "CD", "19"), ("outbreak", "NN", "outbreak")],
     [5, 6, 7, 8, 9])
]

# Define the instance methods of `Chunk` that are called when `Chunk` is initialised; this is all of
# `args_chunk_member_names` excluding those in `args_chunk_method_names`
args_init_call_members = [a for a in args_chunk_member_names if a not in args_chunk_method_names]


@pytest.mark.parametrize("test_input_label, test_input_tokens, test_input_indices", args_inputs)
class TestChunkInitialisation:

    def test_runs(self, test_input_label, test_input_tokens, test_input_indices):
        """Test that Chunk can be initialised."""

        # Attempt to create a `Chunk` object; if this fails for any reason, fail the test
        try:
            assert Chunk(test_input_label, test_input_tokens, test_input_indices)
        except Exception as e:
            pytest.fail(f"Raised exception {type(e)}:\n{str(e)}")

    @pytest.mark.parametrize("test_attribute", args_init_call_members)
    def test_calls_attributes(self, mocker, test_input_label, test_input_tokens, test_input_indices, test_attribute):
        """Test that Chunk, when initialised, calls various object members."""

        # Patch the attribute of the `Chunk` class object
        patch_member = mocker.patch.object(Chunk, test_attribute)

        # Invoke the `Chunk` class
        _ = Chunk(test_input_label, test_input_tokens, test_input_indices)

        # Assert that `test_attribute` has been called once correctly with no arguments
        patch_member.assert_called_once_with()


# Define the expected `text` attribute output for `args_input`
test_expected_text = ["Signed up for", "advice", "to the ongoing COVID 19 outbreak"]

# Define the expected `lemma` attribute output for `args_input`
test_expected_lemma = ["sign up for", "advice", "to the ongoing COVID 19 outbreak"]

# Define the expected `tagable_words` instance method output for `args_input`
test_expected_tagable_words = [[("Signed", "VBN")], [("advice", "NN")], [("COVID", "NNP"), ("outbreak", "NN")]]

# Define the expected `important_word` attribute output for `args_input`
test_expected_important_word = ["Signed", "advice", "ongoing COVID 19 outbreak"]

# Define the expected `important_lemma` attribute output for `args_input`
test_expected_important_lemma = ["sign", "advice", "ongoing COVID 19 outbreak"]

# Zip the member object names with their respective expected outputs
test_member_object_expected = zip(args_chunk_member_names,
                                  [test_expected_text, test_expected_lemma, test_expected_tagable_words,
                                   test_expected_important_word, test_expected_important_lemma])

# Populate the test cases for the `test_member_returns_correctly` pytest - this will be a five-element tuple. The
# first element is the member of the `Chunk` class, the second- to fourth-element will be the input arguments (label,
# tokens, indices), and the last element will be the expected output from this member
args_method_returns_correctly = [(m, *i, e) for m, ev in test_member_object_expected for i, e in zip(args_inputs, ev)]


@pytest.mark.parametrize("test_obj_member, test_input_label, test_input_tokens, test_input_indices, test_expected",
                         args_method_returns_correctly)
def test_member_returns_correctly(test_obj_member, test_input_label, test_input_tokens, test_input_indices,
                                  test_expected):
    """Test that members in the Chunk class return correct values."""

    # Initialise a Chunk object
    test_object = Chunk(test_input_label, test_input_tokens, test_input_indices)

    # Check if `test_obj_member` is a method; if so call the method and check it, otherwise check the attribute
    if test_obj_member in args_chunk_method_names:
        assert getattr(test_object, test_obj_member)() == test_expected
    else:
        assert getattr(test_object, test_obj_member) == test_expected
