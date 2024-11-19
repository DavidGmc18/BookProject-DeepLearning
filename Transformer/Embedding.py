import keras
import tensorflow as tf

# TODO test, added support for shorter seqs
# TODO does not support properly truncating long seqs, meaning positional encoding is relative to first seen token, not to 
# real first token (most often <bos>) as it better. Research if that is really better
class PositionalEmbedding(keras.layers.Embedding):
    def __init__(self, context_len: int, d_model: int):
        super(PositionalEmbedding, self).__init__(context_len, d_model)
        self.context_len = context_len

    def call(self, x):
        range = tf.range(tf.shape(x)[1])
        range = tf.expand_dims(range, axis=0)
        range = tf.repeat(range, repeats=[tf.shape(x)[0]], axis=0)
        return super().call(range)
      

class Embedding(keras.layers.Embedding):
    def __init__(self, vocab_size: int, context_len: int, d_model: int, **kwargs):
        super().__init__(vocab_size, d_model, mask_zero=True, **kwargs)
        self.pos_embedding = PositionalEmbedding(context_len, d_model)
        self.add = keras.layers.Add()

    def call(self, x):
        return self.add([super().call(x), self.pos_embedding(x)]) 
  

class UnEmbedding(keras.layers.Dense):
    def __init__(self, vocab_size, softmax=False, **kwargs):
            super().__init__(vocab_size, activation="softmax" if softmax else None, use_bias=False, **kwargs)
    
    def call(self, x):
        return super().call(x)