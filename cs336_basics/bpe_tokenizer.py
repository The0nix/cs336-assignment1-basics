import heapq
import itertools
import logging
import multiprocessing.pool
import os
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from typing import BinaryIO, Self

import regex as re
from tqdm.auto import tqdm

PRETOKENIZE_REGEX = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

type ByteWord = tuple[bytes, ...]
type BytePair = tuple[bytes, bytes]


class MaxBytePair:
    def __init__(self, pair: BytePair):
        self.pair = pair

    def __lt__(self, other):
        # Reverse comparison
        return self.pair > other.pair

    def __eq__(self, other):
        return self.pair == other.pair

    def __repr__(self):
        return self.pair


logger = logging.getLogger(__name__)


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))

def _pretokenize(
    text: bytes, special_tokens: list[str] | None = None, verbose: bool = False, return_special_tokens: bool = False
) -> Iterator[bytes]:
    if special_tokens is not None:
        split_regex = rb"|".join(re.escape(st.encode("utf-8")) for st in sorted(special_tokens, key=len, reverse=True))
        if return_special_tokens:
            split_regex = b"".join((b"(", split_regex, b")"))
        parts = re.split(split_regex, text)
    else:
        parts = [text]
    del text

    parts_generator = iter(tqdm(parts, desc="Special token splits", disable=not verbose))
    while True:
        tokens = list(itertools.islice(parts_generator, 2 if special_tokens and return_special_tokens else 1))
        if not tokens:
            break
        part, *special = tokens
        for match in tqdm(PRETOKENIZE_REGEX.finditer(part), desc="Word splits", leave=False, disable=not verbose):
            match_text = match.group(0)
            yield match_text
        if special:
            yield from special


def _pretokenize_file_chunk(
    input_path: str, start: int, end: int, special_tokens: list[str] | None = None, verbose: bool = False
) -> Iterator[bytes]:
    # Can add partial matches with regex
    with open(input_path, "rb") as f:
        f.seek(start)
        text = f.read(end - start)
    yield from _pretokenize(text, special_tokens, verbose)


def _get_word_counts(words: Iterable[bytes]) -> Counter[ByteWord]:
    word_counts: Counter[ByteWord] = Counter()
    for word in words:
        bytes_tuple = tuple(bytes([b]) for b in word)
        word_counts[bytes_tuple] += 1
    return word_counts


def _get_word_counts_from_file_chunk(
    input_path: str,
    start: int,
    end: int,
    special_tokens: list[str] | None = None,
    verbose: bool = False,
) -> Counter[ByteWord]:
    return _get_word_counts(_pretokenize_file_chunk(input_path, start, end, special_tokens, verbose))


def _word_pairs(word: ByteWord) -> Iterator[tuple[BytePair | None, BytePair, BytePair | None]]:
    """Returns iterator of prev, cur, next"""
    for i in range(len(word) - 1):
        yield (
            (word[i - 1], word[i]) if i > 0 else None,
            (word[i], word[i + 1]),
            (word[i + 1], word[i + 2]) if i < len(word) - 2 else None,
        )


def train_bpe_tokenizer(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    n_jobs: int | None = None,
    split_special_token: bytes = b"<|endoftext|>",
) -> tuple[dict[int, bytes], list[BytePair]]:
    assert vocab_size > 256 + len(special_tokens), "vocab_size is too low to create any new tokens"

    n_jobs = n_jobs if n_jobs is not None else os.process_cpu_count()
    n_jobs = n_jobs if n_jobs is not None else 1

    with open(input_path, "rb") as f:
        chunk_boundaries = find_chunk_boundaries(f, n_jobs, split_special_token)

    with multiprocessing.pool.Pool(processes=n_jobs) as pool:
        chunk_word_counts = pool.starmap(
            _get_word_counts_from_file_chunk,
            (
                (input_path, start, end, special_tokens, i == 0)
                for i, (start, end) in enumerate(zip(chunk_boundaries[:-1], chunk_boundaries[1:]))
            ),
        )
    word_counts: Counter[ByteWord] = Counter()
    for cwc in chunk_word_counts:
        word_counts.update(cwc)

    vocab = {i: bytes([i]) for i in range(256)}
    for i, special_token in enumerate(special_tokens, 256):
        vocab[i] = special_token.encode()
    merges: list[BytePair] = []

    # Initial pairs count
    pairs_counts: Counter[BytePair] = Counter()
    pairs_to_words: defaultdict[BytePair, set[ByteWord]] = defaultdict(set)
    for word, count in tqdm(word_counts.items(), desc="Initial pair counts"):
        for i in range(len(word) - 1):
            pair = word[i], word[i + 1]
            pairs_counts[pair] += count
            pairs_to_words[pair].add(word)
    pairs_counts_heap = [(-count, MaxBytePair(pair)) for pair, count in pairs_counts.items()]
    heapq.heapify(pairs_counts_heap)

    pbar = tqdm(initial=len(vocab), total=vocab_size, desc="Vocab size")
    while len(vocab) < vocab_size:
        # Add most common pair to the vocabulary
        most_common_count, most_common_maxpair = heapq.heappop(pairs_counts_heap)
        while pairs_counts[most_common_maxpair.pair] != -most_common_count:
            most_common_count, most_common_maxpair = heapq.heappop(pairs_counts_heap)
        most_common_count = -most_common_count
        most_common_pair = most_common_maxpair.pair
        if most_common_count <= 1:
            logger.warning(
                "BPE training corpus exhausted at vocab_size %d before getting to the specified vocab_size of %d",
                len(vocab),
                vocab_size,
            )
            break

        merges.append(most_common_pair)
        new_bytes = most_common_pair[0] + most_common_pair[1]
        vocab[len(vocab)] = new_bytes

        # Merge pretokens with this pair
        for word in list(pairs_to_words[most_common_pair]):
            # Construct new word and update word counts
            new_word: list[bytes] = []
            pairs_iterator = iter(_word_pairs(word))
            while True:
                try:
                    prev, cur, next_ = next(pairs_iterator)
                except StopIteration:
                    break
                pairs_to_words[cur].discard(word)
                if cur == most_common_pair:
                    new_word.append(new_bytes)
                    if prev is not None:
                        pairs_counts[prev] -= word_counts[word]
                        pairs_counts[(prev[0], new_bytes)] += word_counts[word]
                        heapq.heappush(pairs_counts_heap, (-pairs_counts[prev], MaxBytePair(prev)))
                        heapq.heappush(
                            pairs_counts_heap, (-pairs_counts[(prev[0], new_bytes)], MaxBytePair((prev[0], new_bytes)))
                        )
                    if next_ is not None:
                        pairs_counts[next_] -= word_counts[word]
                        pairs_counts[(new_bytes, next_[1])] += word_counts[word]
                        heapq.heappush(pairs_counts_heap, (-pairs_counts[next_], MaxBytePair(next_)))
                        heapq.heappush(
                            pairs_counts_heap,
                            (-pairs_counts[(new_bytes, next_[1])], MaxBytePair((new_bytes, next_[1]))),
                        )
                        prev, cur, next_ = next(pairs_iterator)
                else:
                    new_word.append(cur[0])
            if cur != most_common_pair:
                new_word.append(cur[1])

            # Add new word to new pairs
            new_word_tuple = tuple(new_word)
            for _, cur, _ in _word_pairs(new_word_tuple):
                pairs_to_words[cur].add(new_word_tuple)
            word_counts[new_word_tuple] += word_counts[word]
            del word_counts[word]
        del pairs_counts[most_common_pair]
        del pairs_to_words[most_common_pair]
        pbar.update()
    pbar.close()

    return vocab, merges


class BPETokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[BytePair], special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab
        self.reverse_vocab = {token: index for index, token in self.vocab.items()}
        for special_token in special_tokens or []:
            token_bytes = special_token.encode("utf-8")
            if token_bytes not in self.reverse_vocab:
                index = len(vocab)
                vocab[index] = token_bytes
                self.reverse_vocab[token_bytes] = index
        self.merges = merges
        self.special_tokens = special_tokens

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None) -> Self:
        raise NotImplementedError

    def encode_word(self, word: bytes) -> list[int]:
        if word in self.reverse_vocab:
            return [self.reverse_vocab[word]]
        cur_word = tuple(bytes([b]) for b in word)
        for merge in self.merges:
            if len(cur_word) <= 1:
                break
            new_word: list[bytes] = []
            pairs_iterator = iter(_word_pairs(cur_word))
            while True:
                try:
                    prev, cur, next_ = next(pairs_iterator)
                except StopIteration:
                    break
                if merge == cur:
                    new_word.append(cur[0] + cur[1])
                    if next_ is not None:
                        prev, cur, next_ = next(pairs_iterator)
                else:
                    new_word.append(cur[0])
            if len(cur_word) > 1 and cur != merge:
                new_word.append(cur[1])
            cur_word = tuple(new_word)
        return [self.reverse_vocab[token] for token in cur_word]

    def encode(self, text: str) -> list[int]:
        result = []
        for word in _pretokenize(text.encode("utf-8"), self.special_tokens, return_special_tokens=True):
            encoded = self.encode_word(word)
            result.extend(encoded)
        return result
    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self, ids: list[int]) -> str:
        return b"".join([self.vocab[id_] for id_ in ids]).decode("utf-8", errors="replace")


if __name__ == "__main__":
    vocab, merges = train_bpe_tokenizer("../data/TinyStoriesV2-GPT4-valid.txt", 12000, special_tokens=["<|endoftext|>"])
    tokenizer = BPETokenizer(vocab, merges)
    ids = tokenizer.encode("Hi, my name is pew pew pew")
    print(ids)
    print(tokenizer.decode(ids))
