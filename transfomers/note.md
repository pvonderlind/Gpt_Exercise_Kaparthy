## Self Attention

### Main mechanism of self attention in simple words:
Each token gets assigned three encodings in a head of self-attention.
Each token received embeddings for these 3 encodings of a size defined
as the __'head size'__.
  * __Key__: Show other tokens what this token has to offer (e.g. 'I am a noun in third position of this context ...')
  * __Query__: A token in position three is as a vowel looking for nouns in other positions 
  * __Value__: What the token has to offer in terms of information should other
  tokens be interested in it via the key.
### Encoder and Decoder blocks
Self attention itself allows full communication between tokens. For
the purpose of writing a GPT, it is necessary to differentiate between encoders and decoders:
* __Encoder self-attention__: Has full communication between tokens.
* __Decoder self-attention__: Has restricted communication going backwards in time, e.g. as not to allow
tokens to 'cheat'.
### Scaled attention
It is necessary to normalize the self-attention weights by the square root 
of the head-size. This is important, since the weights feed into a softmax
activation function. If we do not normalize and the weights are very positive or negaqtive,
the __softmax will converge to a one-hot encoding!__

### Layer Normalization
* __Difference to Batch Normalization__: in BN we make sure that every unit in a layer has unit gaussian variance in
their activation. For this we take batch mean and variance of the batch dimension. For layer norm, we do everything
in the same way, but take the mean and variance over the first dimension (feature) and not the batch dim!
* This also has the benefit of not maintaining momentum and buffers for mean and variance over batches.
* __Difference to Self-Attention paper__: Nowadays (~5 years after the paper being released), it is common to apply
LayerNorm __before__ the layers and not after like in their architecture plots.