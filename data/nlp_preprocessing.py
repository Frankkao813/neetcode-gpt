import torch
import torch.nn as nn
from torchtyping import TensorType
from typing import List

class Solution:
    def get_dataset(self, positive: List[str], negative: List[str]) -> TensorType[float]:
        # 1. Build vocabulary: collect all unique words, sort them, assign integer IDs starting at 1
        # 2. Encode each sentence by replacing words with their IDs
        # 3. Combine positive + negative into one list of tensors
        # 4. Pad shorter sequences with 0s using nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        
        # process all sentences, build a unique word list
        sentences = positive + negative
        words = []
        for sentence in sentences:
            words.extend(sentence.split())

        # sorted(): creates and return the new sorted list
        unique_words = sorted(set(words))
        # build a dictionary
        lookup = {}
        for idx, i in enumerate(unique_words):
            lookup[i] = idx + 1
        
        # encode the senteces
        encoded_sentences = []
        for sentence in sentences:
            encoded = []
            for word in sentence.split():
                encoded.append(lookup[word])
            encoded_sentences.append(torch.tensor(encoded, dtype=torch.float32))
            

        # nn.utils.rnn.pad_sequence(): takes a list of tensors whose first dimension can have different lengths, then returns one padded tensor
        padded = nn.utils.rnn.pad_sequence(encoded_sentences, 
                                batch_first = True,
                                padding_value = 0)

        return padded



        
