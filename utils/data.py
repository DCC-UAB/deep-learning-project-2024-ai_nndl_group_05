from __future__ import unicode_literals, print_function, division
from io import open
import unicodedata
import re
import random

import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, random_split

import config

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        # Words
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        # Chars
        self.char2index = {}
        self.char2count = {}
        self.index2char = {0: "SOS", 1: "EOS"}
        self.n_chars = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        if config.model == "words":
            for word in sentence.split(' '):
                self.addWord(word)
        elif config.model == "chars":
            for char in sentence:
                self.addChar(char)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def addChar(self, char):
        if char not in self.char2index:
            self.char2index[char] = self.n_chars
            self.char2count[char] = 1
            self.index2char[self.n_chars] = char
            self.n_chars += 1
        else:
            self.char2count[char] += 1

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


# Read spa-eng file
def readLangs(lang1, lang2):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open(config.data_path, encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]
    pairs = [pair[:2] for pair in pairs]

    # Reverse pairs, make Lang instances
    if config.reverse:
        pairs = [list(reversed(p)) for p in pairs]
    
    input_lang = Lang(lang1)
    output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


def filterPair(p):
    if config.model == "words":
        return len(p[0].split(' ')) < config.max_length and \
                len(p[1].split(' ')) < config.max_length
    elif config.model == "chars":
        return len(p[0]) < config.max_length and len(p[1]) < config.max_length

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]
    

# FUNCTIONS TO GET DATA LOADER
def indexesFromSentence(lang, sentence):
    if config.model == "words":
        return [lang.word2index[word] for word in sentence.split(' ')]
    elif config.model == "chars":
        return [lang.char2index[char] for char in sentence if char in lang.char2index]


def prepareData(lang1, lang2):
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    print(f"Read {len(pairs)} sentence pairs")
    pairs = filterPairs(pairs) #borrar
    print(f"Trimmed to {len(pairs)} sentence pairs") #borrar
    print("Counting words...")
    
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    
    if config.model == "words":
        print("Counted words:")
        print(input_lang.name, input_lang.n_words)
        print(output_lang.name, output_lang.n_words)
    elif config.model == "chars":
        print("Counted chars:")
        print(input_lang.name, input_lang.n_chars)
        print(output_lang.name, output_lang.n_chars)
    
    print("Random pair: ",random.choice(pairs))

    return input_lang, output_lang, pairs

# MAIN

def get_dataloader():
    input_lang, output_lang, pairs = prepareData(config.input_language, 
                                                 config.output_language)

    # Transform to numerical data: one hot vector 
    n = len(pairs)
    input_ids = np.zeros((n, config.max_length), dtype=np.int32) 
    target_ids = np.zeros((n, config.max_length), dtype=np.int32) 

    for idx, (inp, tgt) in enumerate(pairs):
        #inp = "thomas edison invented the light bulb"
        #tgt = "thomas edison invento la bombilla"
        inp_ids = indexesFromSentence(input_lang, inp) 
        tgt_ids = indexesFromSentence(output_lang, tgt) 
        inp_ids.append(EOS_token) 
        tgt_ids.append(EOS_token) 
        input_ids[idx, :len(inp_ids)] = inp_ids 
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    data = TensorDataset(torch.LongTensor(input_ids).to(device),
                        torch.LongTensor(target_ids).to(device))
   
    
    # Create dataloaders
    val_size = int(config.validation_split * len(data))
    test_size = int(config.test_split * len(data))
    train_size = len(data) - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)


    return input_lang, output_lang, train_loader, val_loader, test_loader
