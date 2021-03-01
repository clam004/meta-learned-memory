import torch
import math, copy, sys, logging, json, time, random, os, string, pickle, re

def save_model(model,name):
    torch.save(model.state_dict(),name)
    
def load_model(model,name):
    model.load_state_dict(torch.load(name))
    
class Teacher(): 
    
    def __init__(self, vocab):
        
        self.vocab = vocab

        self.vocab.string2embedding("my name is, hi. what is my name? its")
        
        self.name_list = [
                          'vicki', 'carson', 'melissa', 'salvador', 
                          'force', 'sky', 'zen', 'adam'
                         ]

    def random_name(self,):
        """ Generate a random string of fixed length """
        return random.choice(self.name_list)
    
    def repeat(self, batch_size):
        
        self.mynameis = self.vocab.string2tensor("my name is")
        self.hi = self.vocab.string2tensor("hi")
        self.whatmyname = self.vocab.string2tensor("what is my name?")
        self.its = self.vocab.string2tensor("its")

        self.mynameis = self.mynameis.repeat(batch_size,1)
        self.hi = self.hi.repeat(batch_size,1)
        self.whatmyname = self.whatmyname.repeat(batch_size,1)
        self.its = self.its.repeat(batch_size,1)
    
    def get_batch(self, batch_size):
        
        self.repeat(batch_size)
        
        newnames = ""
        for n in range(batch_size):
            newnames += " " + self.random_name()
            
        self.vocab.string2embedding(newnames)
        
        self.names = self.vocab.string2tensor(newnames).T

        self.intro = torch.cat((self.mynameis, self.names),dim=1)
        self.introtarget = torch.cat((self.hi, self.names),dim=1)
        self.yournameis = torch.cat((self.its, self.names),dim=1)
        
        return self.intro, self.introtarget, self.whatmyname, self.yournameis