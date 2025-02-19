from __future__ import annotations

import collections
import multiprocessing
import os
import regex as re
from itertools import islice
from tqdm import tqdm

TOKEN_PATTERN = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
def read_corpus(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield line.strip()


def tokenize(text):
    tokens = [tuple(token.encode("utf-8")) for token in TOKEN_PATTERN.findall(text)]
    return collections.Counter(tokens)


def process_lines(lines):
    combined_text = "".join(lines)
    token_counts = tokenize(combined_text)
    return token_counts


def chunked_iterator(iterable, size):
    iterator = iter(iterable)
    while chunk := list(islice(iterator, size)):
        yield chunk


def multi_process_tokenization(input_path, num_workers=4, batch_size=100):
    word_freqs = collections.Counter()

    # Create a process pool
    with multiprocessing.Pool(num_workers) as pool:
        with open(input_path, "r", encoding="utf-8") as f:
            results = pool.imap(process_lines, chunked_iterator(f, batch_size), chunksize=4)

            for idx, local_counter in enumerate(results):
                if idx % 100 == 0:
                    print(f"Processed {idx * batch_size}")
                word_freqs.update(local_counter)

    return word_freqs


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
):
    """Given the path to an input corpus, run train a BPE tokenizer and
    output its vocabulary and merges.

    Args:
        input_path: str | os.PathLike
            Path to BPE tokenizer training data.
        vocab_size: int
            Total number of items in the tokenizer's vocabulary (including special tokens).
        special_tokens: list[str]
            A list of string special tokens to be added to the tokenizer vocabulary.
            These strings will never be split into multiple tokens, and will always be
            kept as a single token. If these special tokens occur in the `input_path`,
            they are treated as any other string.

    Returns:
        Tuple of (vocab, merges):
            vocab: dict[int, bytes]
                The trained tokenizer vocabulary, a mapping from int (token ID in the vocabulary)
                to bytes (token bytes)
            merges: list[tuple[bytes, bytes]]
                BPE merges. Each list item is a tuple of bytes (<token1>, <token2>),
                representing that <token1> was merged with <token2>.
                Merges are ordered by order of creation.
    """

    word_freqs = multi_process_tokenization(input_path, num_workers=12, batch_size=5000)
    vocab = {i: bytes([i]) for i in range(256)}
    special_token_ids = {idx + 256: token.encode("utf-8") for idx, token in enumerate(special_tokens)}
    vocab.update(special_token_ids)

    merges = []
    pair_freqs = collections.defaultdict(int)
    for word, freq in word_freqs.items():
        for i in range(len(word) - 1):
            pair_freqs[(word[i], word[i+1])] += freq

    with tqdm(total=vocab_size) as pbar:
        while len(vocab) < vocab_size:
            if len(vocab) % 100 == 0:
                print(f"Vocab size: {len(vocab)}")
            best_pair = max(pair_freqs, key=lambda pair: (pair_freqs[pair], vocab[pair[0]], vocab[pair[1]]), default=None)
            new_token = vocab[best_pair[0]] + vocab[best_pair[1]]
            new_vocab_idx = len(vocab)
            vocab[new_vocab_idx] = new_token
            merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))

            # Update word frequencies
            new_words = collections.defaultdict(int)
            words_to_remove = []
            for word, freq in word_freqs.items():
                new_word = []
                i = 0
                should_update = False
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(new_vocab_idx)
                        should_update = True
                        # Update pair freqs based on what pair got merged
                        if i > 0:
                            pair_freqs[(word[i-1], word[i])] -= freq
                            if pair_freqs[(word[i-1], word[i])] == 0:
                                del pair_freqs[(word[i-1], word[i])]
                            pair_freqs[(word[i-1], new_vocab_idx)] += freq
                        
                        if i < len(word) - 2:
                            pair_freqs[(word[i+1], word[i+2])] -= freq
                            if pair_freqs[(word[i], word[i+2])] == 0:
                                del pair_freqs[(word[i], word[i+2])]
                            pair_freqs[(new_vocab_idx, word[i+2])] += freq
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1

                if should_update:
                    new_words[tuple(new_word)] += freq
                    words_to_remove.append(word)

            for word_to_remove in words_to_remove:
                del word_freqs[word_to_remove]
            word_freqs.update(new_words)
            del pair_freqs[(best_pair[0], best_pair[1])]
            pbar.update(1)
            if len(vocab) >= vocab_size:
                break

    return vocab, merges