from src.chunker import SemanticDoclingChunker


class Encoder:
    def encode(self, text):
        return text.split()

    def decode(self, tokens):
        return " ".join(tokens)


def test_oversized_sentence_is_split_into_bounded_overlapping_windows():
    chunker = SemanticDoclingChunker.__new__(SemanticDoclingChunker)
    chunker.encoder = Encoder()
    chunker.child_size = 4
    chunker.child_overlap = 1

    windows = chunker._split_token_windows("one two three four five six seven")
    assert windows == ["one two three four", "four five six seven"]
    assert all(len(chunker.encoder.encode(window)) <= 4 for window in windows)


def test_invalid_overlap_is_rejected():
    chunker = SemanticDoclingChunker.__new__(SemanticDoclingChunker)
    chunker.encoder = Encoder()
    chunker.child_size = 4
    chunker.child_overlap = 4

    try:
        chunker._split_token_windows("one two three four five")
    except ValueError as error:
        assert "CHUNK_OVERLAP" in str(error)
    else:
        raise AssertionError("invalid overlap must fail")
