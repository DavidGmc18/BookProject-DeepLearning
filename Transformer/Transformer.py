import keras
from . import Attention, FeedForwardNetwork, Embedding
import numpy as np
import tensorflow as tf

class SelfDecoderLayer(keras.layers.Layer):
    def __init__(self, d_model: int, n_heads: int, ffn_dim: int, dropout: int, activation: str="gelu", **kwargs):
        super().__init__(**kwargs)
        self.attn = Attention.CausalSelfAttention(
            n_heads=n_heads,
            d_model=d_model,
            dropout=dropout)
        self.ffn = FeedForwardNetwork.FeedForwardNetwork(d_model, ffn_dim, dropout, activation)

    def call(self, x, padding_mask=None):
        x = self.attn(x, padding_mask)
        return self.ffn(x)
    

class SelfDecoderStack(keras.layers.Layer):
    def __init__(self, n_layers: int, d_model: int, n_heads: int, ffn_dim: int, dropout: float, activation: str="gelu",**kwargs):
        super().__init__(**kwargs)
        self.stack = [SelfDecoderLayer(d_model, n_heads, ffn_dim, dropout, activation) for l in range(n_layers)]

    def call(self, x, padding_mask=None):
        for decoder in self.stack:
            x = decoder(x, padding_mask)
        return x
    

class SelfDecoderModel(keras.Model):
    def __init__(self, vocab_size: int, context_len: int, n_layers: int, d_model: int, n_heads: int, ffn_dim: int, 
                 dropout: float=0.1, activation: str="gelu", softmax: bool=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.vocab_size = vocab_size
        self.context_len = context_len
        self.n_layers = n_layers
        self.d_model = d_model
        self.n_heads = n_heads
        self.ffn_dim = ffn_dim
        self.dropout = dropout
        self.activation = activation
        self.softmax = softmax

        self.embedding = Embedding.Embedding(vocab_size, context_len, d_model)
        self.decoder_stack = SelfDecoderStack(n_layers, d_model, n_heads, ffn_dim, dropout, activation)
        self.unembedding = Embedding.UnEmbedding(vocab_size, softmax=softmax)

        # TODO test, added support for shorter seqs
        self.build(input_shape=(None, None))
        self.name_model()
    
    def call(self, x):
        padding_mask = self.embedding.compute_mask(x)
        x = self.embedding(x)
        x = self.decoder_stack(x, padding_mask)
        return self.unembedding(x)

    def save(self, filepath=None, overwrite=True, save_format=None, **kwargs):
        if filepath == None:
            if save_format == None:
                save_format = "keras"
            filepath = f"Model Storage/{self.name}.{save_format}"
        return super().save(filepath, overwrite, save_format, **kwargs)
    
    def name_model(self, units=["p", "K", "M", "B", "T"]):
        params = self.count_params()
        if params==0:
            self._name = "Transformer"
            return
        digits = int(np.log10(params))
        thousands_x = int(digits/3)
    
        #too big numbers
        if thousands_x >= len(units):
            self._name = f"Transformer-{round(params/(1000**(len(units)-1)))}{units[-1]}"

        #normal case
        elif digits%3 != 0 and digits%3 != 1:
            self._name = f"Transformer-{round(params/(1000**thousands_x))}{units[thousands_x]}"
    
        #add decimal if it is single or double digit, but not add if decimal would be 0
        elif round(params/(1000**thousands_x) - round(params/(1000**thousands_x)), 1) == 0:
            self._name = f"Transformer-{params/(1000**thousands_x):.0f}{units[thousands_x]}"
        else:
            self._name = f"Transformer-{params/(1000**thousands_x):.1f}{units[thousands_x]}"

    # TODO implement penalization for repeating
    # TODO test
    # ids -> None - index -1 for each sample
    #        integer - index for each sample (can be negative to count from end)
    #        array of integers - specify index for each sample separetly (integer can be negative to count from end)
    # truncate -> None - if seq is longer than model context_len it will throw error
    #             left - truncate tokens from left up to model context_len (done only when needed)
    #             right - truncate tokens from right up to model context_len (done only when needed)
    def predict_token(self, inputs, ids=None, temp=None, truncate=None):
        # fix shape
        if inputs.ndim != 2:
            inputs = tf.expand_dims(inputs, axis=0)
    
        # truncate
        if inputs.shape[1] > self.context_len:
            if truncate == "left":
                inputs = inputs[:, -self.context_len:]
            elif truncate == "right":
                inputs = inputs[:, :self.context_len]
            else:
                raise ValueError(f"Input sequence length {inputs.shape[1]} exceeds the allowed context length of {self.context_len}. Truncate parameter has not been provided. You can provide 'left' or 'right' to specify how to truncate the sequence.")
    
        # predict
        predictions = self(inputs)

        # get correct ids
        if type(ids) != list:
            ids = -1 if ids == None else ids
            ids = inputs.shape[1]+ids if ids < 0 else ids
            ids = [ids] * inputs.shape[0]
        predictions = tf.gather(predictions, ids, axis=1, batch_dims=1) #TODO test gather

        # apply temperature (return shape based on input) 
        if temp:
            predictions = predictions / temp
            return tf.squeeze(tf.random.categorical(predictions, num_samples=1, dtype=tf.int32), axis=1)
        else:
            return tf.cast(tf.argmax(predictions, axis=1), dtype=tf.int32)