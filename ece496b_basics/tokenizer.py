import regex as re
import pickle
from typing import Optional, Dict, List, Tuple
import json
from collections import defaultdict


class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] | None = None):
        self.vocab = vocab
        self.merges = merges  # Merge ranking
        self.special_tokens = special_tokens
        self.encoded_special_tokens = []
        offset = len(vocab)
        if special_tokens:
            self.special_tokens = sorted(special_tokens, key = lambda x: len(x), reverse=True)
            self.encoded_special_tokens = set(s.encode("utf-8") for s in (special_tokens or []))
            for idx, special_token in enumerate(special_tokens):
                if special_token.encode("utf-8") in vocab.values():
                    continue
                vocab[idx + offset] = special_token.encode("utf-8")
        self.reverse_vocab = {v: k for k, v in vocab.items()}
    
    def encode(self, text: str) -> List[int]:
        # Handle special tokens separately
        # Quite possibly some really janky code (it is janky code)
        if self.special_tokens:
            text_list = [text]
            for special_token in self.special_tokens:
                for special_character in [".", "^", "$", "*", "+", "?", "{", "}", "[", "]", "(", ")", "|"]:
                    special_token = special_token.replace(special_character, f"\\{special_character}")
                token_regex = f"({special_token})"
                for idx, split_text in enumerate(text_list):
                    if split_text in self.special_tokens:
                        continue
                    text_list[idx] = re.split(token_regex, split_text)
                new_list = []
                for split_text in text_list:
                    if type(split_text) is list:
                        new_list.extend(split_text)
                    else:
                        new_list.append(split_text)
                text_list = new_list
            for idx, split_text in enumerate(text_list):
                if split_text in self.special_tokens:
                    continue
                text_list[idx] = re.findall(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", split_text)
            words = []
            for split_text in text_list:
                if type(split_text) is list:
                    words.extend(split_text)
                else:
                    words.append(split_text)
        else:
            words = re.findall(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""", text)
        tokens = []
        
        for word in words:
            encoded_word = word.encode("utf-8")

            # Handle special tokens
            if encoded_word in self.encoded_special_tokens:
                tokens.append(encoded_word)
                continue
            
            word_bytes = [bytes([char]) for char in encoded_word]
            pairs = [(word_bytes[i], word_bytes[i + 1]) for i in range(len(word_bytes) - 1)]

            while pairs:
                merge_applied = False
                for merge in self.merges:
                    if merge in pairs:
                        i = pairs.index(merge)
                        merged_token = merge[0] + merge[1] 
                        word_bytes = word_bytes[:i] + [merged_token] + word_bytes[i+2:]
                        pairs = [(word_bytes[j], word_bytes[j + 1]) for j in range(len(word_bytes) - 1)]
                        merge_applied = True
                        break
                
                if not merge_applied:
                    break

            tokens.extend(word_bytes)

        return [self.reverse_vocab.get(token, 0) for token in tokens]
    
    def encode_iterable(self, iterable):
        for text in iterable:
            for token_id in self.encode(text):
                yield token_id

    def decode(self, token_ids: List[int]) -> str:
        return b"".join(self.vocab[token_id] for token_id in token_ids).decode("utf-8", errors="ignore")

def tokenize(text):
        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        for match in re.finditer(PAT, text):
            yield tuple(match.group().encode("utf-8"))

def from_files(vocab_filepath, merges_filepath, special_tokens = None):
    with open(vocab_filepath, "rb") as file:
        vocab = pickle.load(file)
    with open(merges_filepath, "rb") as file:
        merges = pickle.load(file)
    return Tokenizer(vocab, merges, special_tokens)

def get_tokenizer(
    vocab: Dict[int, bytes],
    merges: List[Tuple[bytes, bytes]],
    special_tokens: Optional[List[str]] = None,
) -> Tokenizer:
    """Return a BPE tokenizer instance."""
    return Tokenizer(vocab, merges, special_tokens)
