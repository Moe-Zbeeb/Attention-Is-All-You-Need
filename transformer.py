import torch
import torch.nn as nn
from torch.nn import functional as F

# hyperparameters
batch_size = 16 # how many independent sequences will we process in parallel?
block_size = 32 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 100
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0
# ------------

torch.manual_seed(1337)
#for some input.txt 
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read() 

# here are all the unique characters that occur in this text
chars = sorted(list(set(text))) # make a bank of chars 
vocab_size = len(chars) # the number of unique characters 
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) } # encoder bank  take a string, output a list of integers 
itos = { i:ch for i,ch in enumerate(chars) } # decoder bank take a list of integers, output a string
encode = lambda s: [stoi[c] for c in s] # encoder function   take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder function take a list of integers, output a string

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long) #encode all words the text into integers  
#data is a list of integers wherre each integer represents a character in the text
n = int(0.9*len(data)) # first 90% will be train, rest will leave for testing 
train_data = data[:n] # train split 
val_data = data[n:]  # test split  

# data loading
def get_batch(split):
    # generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data  
    #we choose between where to split, 
    #so if we are at training we take a batch from the training data, if we are at validation we take a batch from the validation data
    ix = torch.randint(len(data) - block_size, (batch_size,))  
    #block size is the size of the context,a lengh of the sequence we are going to model upon   
    #block size is context lengh a transformer looks upon to make predictions  
    #ix is a starting index of the context, we are going to model upon  
    #We take batch size random starters of the context, so we can model upon them 
    x = torch.stack([data[i:i+block_size] for i in ix])  
    #X is a batch of context, we are going to model upon 
    #for each starter ix we go forward  block_size steps and take the context  
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])  
    #Y is the target, we are going to predict 
    #for each starter ix we go forward  block_size steps and take the context, but shifted by one step  
    # for example if we have a context "hello" we want to predict "ello"   
    # in numbers we are talking about the same thing, but shifted by one step 
    # like 1,2,3,4,5 -> 2,3,4,5,6 for the first sblock is the model_upon and the second block is the target
    x, y = x.to(device), y.to(device) 
    #move the data to the device, in other words to the GPU or CPU 
    return x, y  
    #then we return the context and the target, so we can model upon them, as context to be the input and target to be the output

@torch.no_grad()# we don't need to compute gradients for this function
def estimate_loss():
    out = {}# we will store the loss for train and val splits
    model.eval()#evaluation mode, this is important because we don't want to update the model parameters
    for split in ['train', 'val']: #choose between 2 splits the train and the validation 
        losses = torch.zeros(eval_iters)# we will store the loss for each iteration
        for k in range(eval_iters):# we will iterate over the number of iterations
            X, Y = get_batch(split)#take the batch of data, X, Y are the context and the target
            logits, loss = model(X, Y)#compute the loss
            losses[k] = loss.item()#store the loss
        out[split] = losses.mean()#store the mean loss for the split
    model.train()#return the model to the training mode
    return out#return the losses for the train and val splits, the return is a set of losses, basically one loss for the train and one for the val

class Head(nn.Module):
    """ one head of self-attention """ 
    """ one head of self-attention is a linear layer that takes the input and produces the key, query and value matrices """  
    """ the key, query and value matrices are then used to compute the attention scores """  
    """ think of the key as asking questions, the query as the answers and the value as the information we want to refine based on the questions  and answers"""

    def __init__(self, head_size):  
        #parameters is the head size   
        #head size is the size of the key, query and value matrices  
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False) 
        #key is a linear layer that takes the input and produces the key matrix, the key matrix is the matrix that asks questions  
        # the layer dimensions are n_embd x head_size
        self.query = nn.Linear(n_embd, head_size, bias=False)  
        #query is a linear layer that takes the input and produces the query matrix, the query matrix is the matrix that answers the questions
        # the layer dimensions are n_embd x head_size
        self.value = nn.Linear(n_embd, head_size, bias=False)  
        #value is a linear layer that takes the input and produces the value matrix, the value matrix is the matrix that contains the information we want to refine based on the questions and answers  
        # the layer dimensions are n_embd x head_size  
        #so as we can see these are linear layers for same dimensionality, n_emb x head_size, that take the input and produce the key, query and value matrices  
        #important information is that we can not use the same layer to output the tree matrices, because the layers would learn the same thing, and we want to learn different things 
        #this will cause symmetrical attention, and we want to avoid that, we want to learn different things, so we can have a better attention mechanism
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        #tril is a lower triangular matrix, we use it to mask the attention scores, so we can avoid looking into the future
        self.dropout = nn.Dropout(dropout) 

    
    """ the forward function is the function that computes the attention scores and the weighted aggregation of the values """  
    """ the forward function takes the input matrix which is of dimensions and produces the key, query and value matrices """
    def forward(self, x):
        B,T,C = x.shape # B is the batch size, T is the context size, C is the embedding size    
        #for example we have batch size (examples we need to process in parallel), the context size (the size of the context we are going to model upon),   
        #And the embedding size (the size of the embedding)
        k = self.key(x)   # (B,T,C) 
        #the key matrix dimensions are (B,T,head size)
        q = self.query(x) # (B,T,C)  
        #the query matrix dimensions are (B,T,head size)
        # compute attention scores ("affinities")
        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B, T, head size) @ (B, head size, T) -> (B, T, T) 
        #the attention scores are computed by multiplying the query and key matrices and then dividing by the square root of the embedding size  
        #the dimensions of the attention scores are (B,T,T) which means that for each token in the context we have the attention scores for all the tokens in the context  
        #so we can know what context tokens are important for each token in the context 
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)     
        #we need to mask the future, or in other words 
        #we need to avoid looking into the future, so we mask the attention scores for the future tokens with negative infinity 
        #for example if we have a context "hello" we want to predict "ello" 
        #we don't want to look at the "e" when we are predicting "e"  
        #to give different example  
        #when predicting "e" we don't want to look at the "l" 
        #so we mask the attention scores for the future tokens with negative infinity, this will be  
        #great to softmax which is at -inf 0 
        wei = F.softmax(wei, dim=-1) # (B, T, T) 
        #will take the attention scores and apply the softmax function to get the probabilities 
        wei = self.dropout(wei)  
        #apply dropout to the probabilities, which means that we will randomly set some of the probabilities to zero, this will help to avoid overfitting  
        # perform the weighted aggregation of the values
        v = self.value(x) # (B,T,C)  
        #the value matrix dimensions are (B,T,head size)  
        #remmber that this a multiplication of B,T,C and B,C,head size so the result is B,T,head size
        out = wei @ v # (B, T, T) @ (B, T, head size) -> (B, T, head size)  
        #the weighted aggregation of the values is computed by multiplying the attention scores and the value matrix  

        return out

class MultiHeadAttention(nn.Module):
    """ multiple heads of self-attention in parallel """  
    """ multiple heads of self-attention in parallel is a set of heads that take the input and produce the key, query and value matrices """  
    """ the key, query and value matrices are then used to compute the attention scores like before but we need the parallelism to learn different things """ 
    """ Think of it as many people asking questions, answering questions and refining information in parallel """ 


    def __init__(self, num_heads, head_size): 
        #parameters are the number of heads and the head size which are 2 hyperparameters 
        #in gpt 4 we have 96 attention heads 
        super().__init__()  
        #initialize the parent class
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])  
        #create a list of heads, each head is an instance of the Head class, we have num_heads of them  
        #each one is of head size equal to the universe head size 
        self.proj = nn.Linear(n_embd, n_embd) 
        #the projection layer is a linear layer that takes the output of the heads and projects it back to the embedding size 
        #the layer dimensions are n_embd x n_embd  
        #this is important because the output of the heads is of head size, and we want to project it back to the embedding size  
        self.dropout = nn.Dropout(dropout) 
        #self.dropout is a dropout layer that will help to avoid overfitting

    def forward(self, x):  
        #the forward function is the function that computes the attention scores and the weighted aggregation of the values
        out = torch.cat([h(x) for h in self.heads], dim=-1)  
        #for each head in the heads list we compute the attention scores and the weighted aggregation of the values
        out = self.dropout(self.proj(out))  
        #apply dropout to the output of the heads and then project it back to the embedding size
        return out  
        #the output is the output of the heads, which is the attention scores and the weighted aggregation of the values  
        #concatenated and then projected back to the embedding size  
        #the projection to embedding size is optional, but it is a good practice to have it, because it will help to avoid overfitting

class FeedFoward(nn.Module):
    """ a simple linear layer followed by a non-linearity """  
    """  
    sequential layer in which we have linear layer of dim n_embd x 4*n_embd, 
    followed by a ReLU non-linearity, 
    followed by a linear layer of dim 4*n_embd x n_embd, 
    followed by a dropout layer  
    The choose 4 is from ATTNETION IS ALL YOU NEED  """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)  
    #the forward function is the function that computes the output of the feedforward layer 
    #The net is the layer we defined it earlier 

class Block(nn.Module):
    """ Transformer block: communication followed by computation """  
    """ takes the input and produces the output by applying the multi-head attention and the feedforward layer"""

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: the number of heads we'd like
        super().__init__()   
        #the choose head size is from the paper ATTENTION IS ALL YOU NEED
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)#self-attention layer
        self.ffwd = FeedFoward(n_embd)#feedforward layer
        self.ln1 = nn.LayerNorm(n_embd)#layer normalization layer
        self.ln2 = nn.LayerNorm(n_embd)#layer normalization layer

    def forward(self, x):#the forward function is the function that computes the output of the block
        x = x + self.sa(self.ln1(x))#apply attention layer with residual connection
        x = x + self.ffwd(self.ln2(x))#apply feedforward layer with residual connection
        return x

class BigramLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  
        #the embedding space we will work with is a vocab size by n embd 
        #as for each vocab we have an embedding 
        self.position_embedding_table = nn.Embedding(block_size, n_embd)  
        #the position embedding table is a block size by n embd   
        #in which we have the position embeddings for each position in the context 
        #so each poistion in the block is a vector of n embd 
        #a row by linear means 
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])  
        #the blocks are a sequential layer in which we have n layer blocks
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)#linear layer that takes the output of the blocks and produces the logits for the next token  
        #the layer dimensions are n_embd x vocab_size 
        #which means that the output of the blocks is projected to the vocab size, so we can get the logits for the next token

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) 
        # (T,C) holds positions information 
        x = tok_emb + pos_emb # (B,T,C) add the token and position embeddings
        x = self.blocks(x) # (B,T,C) apply the blocks
        x = self.ln_f(x) # (B,T,C) apply the final layer norm
        logits = self.lm_head(x) # (B,T,vocab_size) get the logits for the next token which are T steps ahead each step is one embedding vector

        if targets is None: # if we are not given the targets, we return the logits and no loss  
                            # to differentiate between the inference and training modes
            loss = None
        else:
            B, T, C = logits.shape      
            logits = logits.view(B*T, C) #taking the logits and reshaping them to B*T, C inorder to apply the cross entropy loss
            #the cross entropy loss takes the logits and the targets and computes the loss 
            targets = targets.view(B*T) #taking the targets and reshaping them to B*T inorder to apply the cross entropy loss
            loss = F.cross_entropy(logits, targets) #computing the cross entropy loss

        return logits, loss #returning the logits and the loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:] 
            # (B, T) get the last block_size tokens, idx_cond is the context we are going to model upon
            # get the predictions
            logits, loss = self(idx_cond) #get the logits for the next token based on a idx_cond which is the context we are going to model upon
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C) get the logits for the last token
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1) sample the next token based on the probabilities
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1) append the sampled index to the context
        return idx

model = BigramLanguageModel()
m = model.to(device)
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=2000)[0].tolist()))
