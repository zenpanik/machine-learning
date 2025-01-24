# Notes from Let's build GPT: from scratch, in code, spelled out. video from Andrej Karpathy
---
## Tokenizers
Tokenize the input text - the process of converting raw input text to sequence of integers according to a vocabulary of possible elements.

Option 1:
Transform individual characters into integers:
- use enumeration <br>
- create encoder - take input string and produce sequence of integers output <br>
- create decoder - take input sequence of integers and produce string output <br>

OpenAI - tiktoken
Google - SentencePiece

## Training
### Data
Never feed the entire text all at once. This is very heavily computational. Split the dataset into chunks.

Use Random Sampling to create batches or chunks with respect to max blocks/context_length. 
- Max block size
- train_data = block_size + 1
- simultanously train to make prediction for all of the positions in the block [ up to context + 1 for prediction]
- x = train_data[:block_size]
- y = train_data[1:block_size + 1]
- context = x[:t+1] for t in range(block_size]
- target = y[t]

Setting up the learning this way we teach the transformer to be able to predict a token
with as little as 1 context and up to block size context. After we exceed the limit of block size
then we need to start truncating the beginning of the context

Then we get input - output matrices:
input shape = [batch size, max context_length]
target shape = [batch size, max context_length]

Then for each training tuple we can generate training examples having context lenght of
1 to max context length.

2d arrays:
rows from input maps to rows from target
for each column idx from target array we need to select elements from 1 to idx from input array

this way the total number of training examples is 4*8 = 32
each row of the target is the lagged with 1 index input row + 1 element that is in the future and does not belong to the input

### Feeding the NN
token embedding table [vocal size, vocab size]

map the input matrix to and get the predictions:
- Batch , Time, Channel

the loss is expected to be:

-ln(1/vocab size)

### Optimizers
Stochastic Gradient Descent
Adam - optimizer is designed to be appropriate for non-stationary objectives and problems with very noisy and/or sparse gradients.
AdamW - decoupling weight decay from the gradient update in Adam





