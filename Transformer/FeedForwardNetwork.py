import keras

class FeedForwardNetwork(keras.layers.Layer):
  def __init__(self, d_model: int, ffn_dim: int, dropout: float, activation: str="gelu", **kwargs):
    super().__init__(**kwargs)
    self.layernorm = keras.layers.LayerNormalization()
    self.ffn = keras.Sequential([
      keras.layers.Dense(ffn_dim, activation=activation, kernel_initializer="HeNormal"),
      keras.layers.Dense(d_model)
    ])
    self.add = keras.layers.Add()
    self.drop_out = keras.layers.Dropout(dropout)

  def call(self, x):
    norm_out=x
    norm_out = self.layernorm(x)
    ffn_out = self.ffn(norm_out)
    ffn_out = self.drop_out(ffn_out)
    return self.add([x, ffn_out])