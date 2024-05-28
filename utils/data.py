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
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

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
    return len(p[0].split(' ')) < config.max_length and \
        len(p[1].split(' ')) < config.max_length


def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]


# FUNCTIONS TO GET DATA LOADER
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]


def prepareData(lang1, lang2):
    input_lang, output_lang, pairs = readLangs(lang1, lang2)
    print(f"Read {len(pairs)} sentence pairs")
    pairs = filterPairs(pairs) #borrar
    print(f"Trimmed to {len(pairs)} sentence pairs") #borrar
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    
    print("Random pair: ",random.choice(pairs))

    return input_lang, output_lang, pairs

# MAIN

def get_dataloader():
    input_lang, output_lang, pairs = prepareData(config.input_language, 
                                                 config.output_language)

    # Transform to numerical data: one hot vector 
    n = len(pairs)
    input_ids = np.zeros((n, config.max_length), dtype=np.int32) #inputs_ids[0]=[0 0 0 0 0 0 0 0 0 0], len[0]=max_length, len=119635, type=<class 'numpy.ndarray'>
    target_ids = np.zeros((n, config.max_length), dtype=np.int32) #target_ids[0]=[0 0 0 0 0 0 0 0 0 0], len[0]=max_length, len=119635, type=<class 'numpy.ndarray'>

    for idx, (inp, tgt) in enumerate(pairs):
        #inp = "thomas edison invented the light bulb"
        #tgt = "thomas edison invento la bombilla"
        inp_ids = indexesFromSentence(input_lang, inp) #thomas edison invented the light bulb -> [10615, 8633, 3602, 827, 1422, 4877]
        tgt_ids = indexesFromSentence(output_lang, tgt) #thomas edison invento la bombilla -> [20828, 16967, 7086, 150, 5906]
        inp_ids.append(EOS_token) #[10615, 8633, 3602, 827, 1422, 4877, 1]
        tgt_ids.append(EOS_token) #[20828, 16967, 7086, 150, 5906, 1]
        input_ids[idx, :len(inp_ids)] = inp_ids #input_ids[idx] = [10615  8633  3602   827  1422  4877     1     0     0     0]
        target_ids[idx, :len(tgt_ids)] = tgt_ids #target_ids[idx] = [20828 16967  7086   150  5906     1     0     0     0     0]

    data = TensorDataset(torch.LongTensor(input_ids).to(device),
                        torch.LongTensor(target_ids).to(device))
    print("Train data tensor example: ",data[1000]) 
    # (tensor([ 236,  178, 2813,  663,  475,  224,  157,    1,    0,    0]), tensor([3938, 1636, 5700,   66, 2880,  460,  361,    1,    0,    0]))
    
    # Create dataloaders
    #train_sampler = RandomSampler(train_data) #train_data = data
    #train_loader = DataLoader(train_data, sampler=train_sampler, batch_size=config.batch_size)
    val_size = int(config.validation_split * len(data))
    test_size = int(config.test_split * len(data))
    train_size = len(data) - val_size - test_size

    train_dataset, val_dataset, test_dataset = random_split(data, [train_size, val_size, test_size])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size)

    """for inp, tgt in train_loader:
        print(inp) # tensor([[inp_ids]*batch_size])
        print(tgt) # tensor([[inp_ids]*batch_size])
        
        Example: 
        tensor([[   19,   112,  1914,   282,   779,     1,     0,     0,     0,     0],
        [  164,    18,   123,   621,   256,   282,  1190,    20,     7,     1],
        [   23,   312,   282,   150,    88,  4234,   256,   735,   410,     1],
        [   71,    48,     1,     0,     0,     0,     0,     0,     0,     0],
        [   71,  2030,   886,   174,   827,  6280,     1,     0,     0,     0],
        [   71,  7930,   223,  7931,     1,     0,     0,     0,     0,     0],
        [   71,  9894,   886,   282,    13,   269,   827,  2392,     1,     0],
        [   23,   159,   138,  1948,     6,  1749,    44,   930,  2086,     1],
        [   65,   169,   147,   114,   152,   799,  3875,     1,     0,     0],
        [   71,   906, 11093,   366,   511,   568,     1,     0,     0,     0],
        [  314,  1747,   174,  2189,     1,     0,     0,     0,     0,     0],
        [  320,   123,    11,    46,   364,   152,  1949,     7,     1,     0],
        [   23,   499,   308,   282,    61,   282,   711,     1,     0,     0],
        [   23,    59,   134,   194,     1,     0,     0,     0,     0,     0],
        [  537,   880,  1119,     1,     0,     0,     0,     0,     0,     0],
        [  130,    46,  1306,   880,   676,     1,     0,     0,     0,     0],
        [  379,   411,  1150,  4864, 10488,    44,   152,  2443,     1,     0],
        [  543,  1948,   619,    29,   808,     1,     0,     0,     0,     0],
        [   23,    97,   663,   123,    20,   152,  2823,     1,     0,     0],
        [  123,   323,   200,   328,   146,   147,   123,     7,     1,     0],
        [   23,   105,   301,   123,   138,   918,     1,     0,     0,     0],
        [  236,   448,   228,   257,    46,     1,     0,     0,     0,     0],
        [   23,   687,   138,  5686,   366,  2533,     1,     0,     0,     0],
        [ 3127,   827,  7246, 11192,   286,  2371,     1,     0,     0,     0],
        [   71,  1735,   147,   495,   282,   148,   883,  2093,     1,     0],
        [   42,  5258,   285,  1261,   319,   410,   663,   223,   825,     1],
        [    6,   906,   827,   975,  1321,   282,     7,     1,     0,     0],
        [   65,   730,   370,   560,  4115,   314,   918,   138,  3764,     1],
        [  411,   123,    88,   366,   880,   961,     7,     1,     0,     0],
        [  543,  2605,   411,   549,     1,     0,     0,     0,     0,     0],
        [   23,    59,   156,   282,    14,   560,    90,   999,     1,     0],
        [  236,   480,   159,  1906,  1070,  1307,     1,     0,     0,     0]])
        """
    
    return input_lang, output_lang, train_loader, val_loader, test_loader
