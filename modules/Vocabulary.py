import re
import numpy as np

import torch
from torch import nn

class Vocab(nn.Module):
    
    def __init__(self, emb_dim, word2index = None, embedding = None, 
                 word2count = None, emb2vocab = None):
        
        '''
        This is a class that handles the transformation between 
        embedding space and vocabulary space. The methods of this class update both 
        the number of word embeedings and also the size of the matrix that maps
        the vector output of a neural network module to a vector the length of
        the output vocabulary for selecting the next token to output.
        
        embed_dim (integer): number of dimensions to represent words/tokens with
        word2index (dict): a dictionary mapping a word-string to it's unique integer index 
                    that represents it in the embedding matrix
        embedding (nn.Embedding): module that keeps a matrix where each row is a trainable
                    word vector with the row index corresponding to the token index
        word2count (dict): a dictionary mapping a word-string to the number of times it's been 
                    used, either in the input or the outout 
        emb2vocab (nn.Linear): module that takes an vector of length emb_dim 
                    (note: there is no reason that this has to be the same length as
                    the word vectors) and transforms this to a vector of length
                    vocab_size, aka self.embedding.weight.shape[0] or len(self.word2index)
        '''
        
        super().__init__()
        
        self.emb_dim = emb_dim

        if word2index is not None:
            self.word2index = word2index
            self.index2word = {v: k for k, v in self.word2index.items()}
        else:
            self.word2index = {"<PAD>":0, "<SOS>":1, "<EOS>":2, "<UNK>":3}
            self.index2word = {v: k for k, v in self.word2index.items()}
            
        if word2count is not None:
            self.word2count = word2count
        else:
            self.word2count = {}
            
        self.embedding = nn.Embedding(len(self.word2index), self.emb_dim)
        self.emb2vocab = nn.Linear(self.emb_dim,len(self.word2index),bias=False)
            
    def string2tokens(self, string):
        """ this function is to change the string according to 
        the substitution rules you apply below """
        # puts space between words and ,.!?
        string = re.sub(r"([,.!?])", r" \1", string) 
        # collapse words like you're and don`t, into youre and dont 
        string = re.sub(r"([`'])", r"", string) 
        # turn characters not in ^a-zA-Z0-9,.!? into a space
        string = re.sub(r"[^a-zA-Z0-9,.!?#]+", r" ", string) 
        # make all text lowercase
        string = string.lower()
        # split sentence string into list of word strings
        string = string.rstrip().lstrip().split(" ")
        return string
    
    def tokens2tensor(self, list_o_strings):
        """takes a list of strings, looks each up with word2index
           and returns a torch long tensor of those indices"""
        integer_sequence = []
        for wrdstr in list_o_strings:
            if wrdstr in self.word2index:
                integer_sequence.append(self.word2index[wrdstr])
            else:
                integer_sequence.append(self.word2index["<UNK>"])
        return torch.LongTensor([integer_sequence])
    
    def string2tensor(self, sentence_str):
        """ takes string sentence, returns tensor integer sentence
        without adding new words to the vocabulary"""
        list_o_strings = self.string2tokens(sentence_str)
        tnsr_int_sntnc = self.tokens2tensor(list_o_strings)
        return tnsr_int_sntnc
    
    def string2embedding(self, sentence_str):
        """takes a sentence as a string and increments the
        wordcount of each word in the string, if word has never been
        seen, it is added to the word2index and embedding"""
        list_o_strings = self.string2tokens(sentence_str)
        for wrdstr in list_o_strings:
            if wrdstr in self.word2index:
                if wrdstr in self.word2count:
                    self.word2count[wrdstr] += 1
                else:
                    self.word2count[wrdstr] = 1
            else:
                self.word2count[wrdstr] = 1
                self.word2index[wrdstr] = len(self.word2index)
                self.embedding.weight = nn.Parameter(torch.cat((self.embedding.weight, 
                     torch.randn(1, self.emb_dim)), dim=0), requires_grad=True)
                self.emb2vocab.weight = nn.Parameter(torch.cat((self.emb2vocab.weight,
                     torch.randn(1, self.emb_dim)), dim=0), requires_grad=True)
                
        self.index2word = {v: k for k, v in self.word2index.items()}
        return list_o_strings
    
    def string2embed2tensor(self, sentence_str):
        """ takes string sentence, returns tensor integer sentence
         adding new words to the vocabulary"""
        
        list_o_strings = self.string2embedding(sentence_str)
        tnsr_int_sntnc = self.tokens2tensor(list_o_strings)
        return tnsr_int_sntnc
    
    def prunevocab(self, mincount):
        """ loops through word2count to find words used less than mincount
        if it's count is less than mincount then its index is used to remove
        the row of that index in the embedding matrix. The word2index dictionary
        is also adjusted by decrementing it's integer indices to make restore the
        word-index-vector relationship. lastly the words are removed from word2index
        and word2count and  index2word is recreated from word2index """
        words2del = [] 
        for wrdstr in self.word2count:
            if self.word2count[wrdstr] < mincount:
                words2del.append(wrdstr)
                wrdidx = self.word2index[wrdstr]
                self.embedding.weight=nn.Parameter(torch.cat((self.embedding.weight[:wrdidx], 
                                 self.embedding.weight[wrdidx+1:]), dim=0),requires_grad=True)
                self.emb2vocab.weight=nn.Parameter(torch.cat((self.emb2vocab.weight[:wrdidx], 
                                 self.emb2vocab.weight[wrdidx+1:]),dim=0),requires_grad=True)
                for decrwrd in self.word2count:
                    self.word2index[decrwrd] -= 1

        for wrd in words2del:
            del self.word2index[wrd]
            del self.word2count[wrd]
        
        self.index2word = {v: k for k, v in self.word2index.items()}