import multiprocessing.pool
import os
import regex as re
import logging
from collections import Counter, defaultdict
from collections.abc import Iterator
from typing import BinaryIO

from tqdm.auto import tqdm

PRETOKENIZE_REGEX = re.compile(rb"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

type ByteWord = tuple[bytes, ...]
type BytePair = tuple[bytes, bytes]

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


class BPEExhaustedError(Exception):
    pass


def _pretokenize_chunk(
    input_path: str, special_tokens: list[str], start: int, end: int, verbose: bool = False
) -> Counter[ByteWord]:
    # Can add partial matches with regex
    with open(input_path, 'rb') as f:
        f.seek(start)
        text = f.read(end - start)

    split_regex = re.compile(rb"|".join(re.escape(st.encode('utf-8')) for st in special_tokens))
    parts = split_regex.split(text)
    del text

    word_counts: Counter[ByteWord] = Counter()
    for part in tqdm(parts, desc='Special token splits', disable=not verbose):
        for match in tqdm(PRETOKENIZE_REGEX.finditer(part), desc='Word splits', leave=False, disable=not verbose):
            match_text = match.group(0)
            if len(match_text) > 1:
                bytes_tuple = tuple(b.to_bytes() for b in match_text)
                word_counts[bytes_tuple] += 1

    return word_counts


def _get_most_common_pair(pair_counts: Counter[BytePair]) -> BytePair:
    max_pair, max_pair_count = next(iter(pair_counts.items()))
    for pair, count in pair_counts.items():
        max_pair_count, max_pair = max((max_pair_count, max_pair), (count, pair))
    if max_pair_count <= 1:
        raise BPEExhaustedError
    return max_pair

def _word_pairs(word: ByteWord) -> Iterator[tuple[BytePair | None, BytePair, BytePair | None]]:
    """Returns iterator of prev, cur, next"""
    for i in range(len(word) - 1):
        yield (
            (word[i - 1], word[i]) if i > 0 else None,
            (word[i],     word[i + 1]),
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
            _pretokenize_chunk,
            (
                (input_path, special_tokens, start, end, i == 0)
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
    for word, count in tqdm(word_counts.items(), desc='Initial pair counts'):
        for i in range(len(word) - 1):
            pair = word[i], word[i + 1]
            pairs_counts[pair] += count
            pairs_to_words[pair].add(word)

    pbar = tqdm(initial=len(vocab), total=vocab_size, desc='Vocab size')
    while len(vocab) < vocab_size:
        # Add most common pair to the vocabulary
        try:
            most_common_pair = _get_most_common_pair(pairs_counts)
        except BPEExhaustedError:
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
                    if next_ is not None:
                        pairs_counts[next_] -= word_counts[word]
                        pairs_counts[(new_bytes, next_[1])] += word_counts[word]
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

if __name__ == '__main__':
    vocab, merges = train_bpe_tokenizer("../data/TinyStoriesV2-GPT4-valid.txt", 12000, special_tokens=["<|endoftext|>"])