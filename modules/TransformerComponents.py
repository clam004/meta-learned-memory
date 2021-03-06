import math, random, copy
import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable # Used for positional encoder

class Embedder(nn.Module):
    def __init__(self, vocab_size, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.embed = nn.Embedding(vocab_size, emb_dim)
    def forward(self, x):
        return self.embed(x)

class PositionalEncoder(nn.Module):

    def __init__(self, emb_dim, max_seq_len = 200, dropout = 0.1):
        super().__init__()
        self. emb_dim =  emb_dim
        self.dropout = nn.Dropout(dropout)
        # create constant 'pe' matrix with values dependant on pos and i
        # this matrix of shape (1, input_seq_len, emb_dim) is
        # cut down to shape (1, input_seq_len, emb_dim) int he forward pass
        # to be broadcasted across each sampel in the batch 
        pe = torch.zeros(max_seq_len, emb_dim)
        for pos in range(max_seq_len):
            for i in range(0, emb_dim, 2):
                wavelength = 10000 ** ((2 * i)/ emb_dim)
                pe[pos, i] = math.sin(pos / wavelength)
                pe[pos, i + 1] = math.cos(pos / wavelength)
        pe = pe.unsqueeze(0) # add a batch dimention to your pe matrix 
        self.register_buffer('pe', pe)#block of data(persistent buffer)->temporary memory
 
    def forward(self, x):
        '''
        input: sequence of vectors  
               shape (batch size, input sequence length, vector dimentions)
        output: sequence of vectors of same shape as input with positional
                aka time encoding added to each sample 
                shape (batch size, input sequence length, vector dimentions)
        '''
        x = x * math.sqrt(self. emb_dim) # make embeddings relatively larger
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe #add constant to embedding
        return self.dropout(x)
    
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class Norm(nn.Module):
    def __init__(self, emb_dim, eps = 1e-6):
        super().__init__()
        self.size = emb_dim
        # alpha and bias are learnable parameters that scale and shift
        # the representations respectively, aka stretch and translate 
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps #prevents divide by zero explosions 
    
    def forward(self, x):
        '''
        input/output: x/norm shape (batch size, sequence length, embedding dimensions)
        '''
        norm = (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps)
        norm = self.alpha * norm + self.bias
        return norm

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, emb_dim, dim_k = None, dropout = 0.1):
        super().__init__()
        
        self.emb_dim = emb_dim
        self.dim_k = dim_k if dim_k else emb_dim // num_heads
        self.num_heads = num_heads
        self.q_linear = nn.Linear(emb_dim,self.dim_k*num_heads)
        self.k_linear = nn.Linear(emb_dim,self.dim_k*num_heads)
        self.v_linear = nn.Linear(emb_dim,self.dim_k*num_heads)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.dim_k*num_heads,emb_dim)
    
    def attention(self, q, k, v, dim_k, mask=None, dropout=None):
        k = k.transpose(-2, -1)

        # matrix multiplication is done using the last two dimensions
        # (batch_size,num_heads,q_seq_len,dim_k)X(batch_size,num_heads,dim_k,k_seq_len)
        # (batch_size,num_heads,q_seq_len,k_seq_len)
        scores = torch.matmul(q, k) / math.sqrt(dim_k) 

        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask == 0, -1e9) 
        softscores = F.softmax(scores, dim=-1)
        if dropout is not None: softscores = dropout(softscores)
            
        #(batch_size,num_heads,seq_len,seq_len)X(batch_size,num_heads,seq_len,dim_k)
        output = torch.matmul(softscores, v)
        return output, scores #=(batch_size,num_heads,seq_len,dim_k)
    
    def forward(self, q, k, v, mask=None):
        '''
        inputs:
            q has shape (batch size, q_sequence length, embedding dimensions)
            k,v are shape (batch size, kv_sequence length, embedding dimensions)
            source_mask of shape (batch size, 1, kv_sequence length)
        outputs: sequence of vectors, re-represented using attention
            shape (batch size, q_sequence length, embedding dimensions)
        use:
            The encoder layer places the same source vector sequence into q,k,v 
            and source_mask into mask.
            The decoder layer uses this twice, once with decoder inputs as q,k,v 
            and target mask as mask. then with decoder inputs as q, encoder outputs
            as k, v and source mask as mask
        '''
        # k,q,v are each shape (batch size, sequence length, dim_k * num_heads)
        batch_size = q.size(0)
        q = self.q_linear(q)
        k = self.k_linear(k)
        v = self.v_linear(v)

        # k,q,v are each shape (batch size, sequence length, num_heads, dim_k)
        k = k.view(batch_size,-1,self.num_heads,self.dim_k)
        q = q.view(batch_size,-1,self.num_heads,self.dim_k)
        v = v.view(batch_size,-1,self.num_heads,self.dim_k)
        # transpose to shape (batch_size, num_heads, sequence length, dim_k)
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we will define next
        attn, scores = self.attention(q, k, v, self.dim_k, mask, self.dropout)

        # concatenate heads and 
        concat=attn.transpose(1,2).contiguous().view(batch_size,-1,self.dim_k*self.num_heads)

        # put through final linear layer
        output = self.out(concat)

        return output, scores

class FeedForward(nn.Module):
    def __init__(self, emb_dim, ff_dim=2048, dropout = 0.1):
        super().__init__() 
    
        # We set d_ff as a default to 2048
        self.linear_1 = nn.Linear(emb_dim, ff_dim)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(ff_dim, emb_dim)
    
    def forward(self, x):
        x = self.dropout(F.leaky_relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x

class DecoderLayer(nn.Module):

    def __init__(self, emb_dim, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(emb_dim)
        self.norm_2 = Norm(emb_dim)
        self.norm_3 = Norm(emb_dim)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)
        
        self.attn_1 = MultiHeadAttention(heads, emb_dim, dropout=dropout)
        self.attn_2 = MultiHeadAttention(heads, emb_dim, dropout=dropout)
        self.ff = FeedForward(emb_dim, dropout=dropout)

    def forward(self, de_out, de_mask, en_out, en_mask):
        '''
        inputs:
            de_out-decoder ouput so far (batch size, output sequence length, embedding dimensions)
            de_mask-(batch size, output sequence length, output sequence length)
            en_out-encoder output (batch size, input sequence length, embedding dimensions)
            en_mask-(batch size, 1, input sequence length)
        ouputs:
            de_out-(next decoder output) (batch size, output sequence length, embedding dimensions)
        '''

        de_nrm = self.norm_1(de_out)
        #Self Attention 
        self_attn, self_scores = self.attn_1(de_nrm, de_nrm, de_nrm, de_mask)
        de_out = de_out + self.dropout_1(self_attn)
        de_nrm = self.norm_2(de_out)
        #DecoderEncoder Attention
        en_attn, en_scores = self.attn_2(de_nrm, en_out, en_out, en_mask) 
        de_out = de_out + self.dropout_2(en_attn)
        de_nrm = self.norm_3(de_out)
        de_out = de_out + self.dropout_3(self.ff(de_nrm))
        return de_out
    
class EncoderLayer(nn.Module):
    def __init__(self, emb_dim, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(emb_dim)
        self.dropout_1 = nn.Dropout(dropout)
        self.attn = MultiHeadAttention(heads, emb_dim, dropout=dropout)
        self.norm_2 = Norm(emb_dim)
        self.ff = FeedForward(emb_dim, dropout=dropout)
        self.dropout_2 = nn.Dropout(dropout)
        
    def forward(self, vector_sequence, mask):
        '''
        input:
            vector_sequence of shape (batch size, sequence length, embedding dimensions)
            source_mask (mask over input sequence) of shape (batch size, 1, sequence length)
        output: sequence of vectors after embedding, postional encoding, attention and normalization
            shape (batch size, sequence length, embedding dimensions)
        '''
        x2 = self.norm_1(vector_sequence)
        x2_attn, x2_scores = self.attn(x2,x2,x2,mask)
        vector_sequence = vector_sequence + self.dropout_1(x2_attn)
        x2 = self.norm_2(vector_sequence)
        vector_sequence = vector_sequence + self.dropout_2(self.ff(x2))
        return vector_sequence
    
class Transformer(nn.Module):
    '''
    If your target sequence is `see` `ya` and you want to train on the entire 
    sequence against the target, you would use `<sos>` `see`  `ya`
    as the de_out (decoder ouputs so far) and compare the 
    output de_out (next decoder output) `see` `ya` `<eos>` 
    as the target in the loss function. The inclusion of the `<sos>`
    for the (decoder ouputs so far) and `<eos>` for the 
    '''
    def __init__(self, emb_dim, n_layers, heads, dropout):
        super().__init__()
        
        self.n_layers = n_layers
        self.pe = PositionalEncoder(emb_dim, dropout=dropout)
        self.encodelayers = get_clones(EncoderLayer(emb_dim, heads, dropout), n_layers)
        self.decodelayers = get_clones(DecoderLayer(emb_dim, heads, dropout), n_layers)
        self.norm = Norm(emb_dim)
        
    def forward(self, de_vecs, de_mask, en_vecs, en_mask):
        '''
        inputs:
            de_vecs - decoder ouputs so far 
                      (batch size, output sequence length, embedding dimensions)
            de_mask (batch size, output sequence length, output sequence length)
            en_vecs - encoder output 
                      (batch size, input sequence length, embedding dimensions)
            en_mask (batch size, 1, input sequence length)
        outputs:
            de_vecs - next decoder output (batch size, output sequence length, embedding dimensions)
        '''
        
         
        en_vecs = self.pe(en_vecs) # Added this newly 
        
        for i in range(self.n_layers):
            en_vecs = self.encodelayers[i](en_vecs, en_mask)
        
        de_vecs = self.pe(de_vecs) # Added this newly
        
        for i in range(self.n_layers):
            de_vecs = self.decodelayers[i](de_vecs, de_mask, en_vecs, en_mask)
            
        return self.norm(de_vecs)
    
# What we call Transformer here corresponds to a modification of
# https://github.com/clam004/chat-transformer/blob/main/scripts/Transformer.py
# Where we have left out the embedding step so that
# embedding word vectors can be handled by the Elastic Vocabulary Module 