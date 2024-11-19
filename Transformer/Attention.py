import keras
import tensorflow as tf

class BaseAttention(keras.layers.MultiHeadAttention):
  def __init__(self, n_heads, d_model, dropout, **kwargs):
    super().__init__(num_heads=n_heads, key_dim=d_model//n_heads, dropout=dropout, **kwargs)
    self.layernorm = keras.layers.LayerNormalization()
    self.add = keras.layers.Add()


class CausalSelfAttention(BaseAttention):
  def call(self, x, padding_mask=None):
    norm_out = self.layernorm(x)
    attn_out = super().call(
        query=norm_out,
        value=norm_out,
        key=norm_out,
        attention_mask=_create_attn_mask(padding_mask),
        use_causal_mask = True)
    return self.add([x, attn_out])
  

def _create_attn_mask(padding_mask):
  if padding_mask == None:
    return
  return tf.expand_dims(padding_mask, -1) & tf.expand_dims(padding_mask, -2)