from nltk import  RegexpParser, tree
from src.make_feedback_tool_data.chunk import Chunk

parser = RegexpParser(grammar)

def chunk_text(tagged):

    chunks = parser.parse(tagged)
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

def extract_phrase(sentences, merge_inplace=False):
    chunks = []
    for sentence in sentences:
        chunks.append(chunk_text(sentence))
    if merge_inplace:
        return [merge_adjacent_chunks(chunk) for chunk in chunks]
    return chunks

def merge_adjacent_chunks(chunks):
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

def compute_combinations(sentences, n):
    return [chunks[i:i+n] for chunks in sentences for i in range(len(chunks)-(n-1))]
