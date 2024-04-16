# %%
import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense, LayerNormalization, Dropout
from keras.layers import Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.optimizers import Adam
from torch import permute


# %%

# Hyperparameters
patch_size = 16
embed_dim = 768
num_heads = 12
num_layers = 12
mlp_dim = 3072
num_classes = 10

# Input image shape
img_size = 224
channels = 3

# Input layer
inputs = Input(shape=(img_size, img_size, channels))

# Patch extraction
patches = Conv2D(embed_dim, (patch_size, patch_size), strides=patch_size)(inputs)
patches = Flatten()(patches)

# Positional embeddings
pos_emb = np.zeros((1, patches.shape[1], embed_dim))
pos_emb = Model(inputs, pos_emb)(inputs)

# Combine patch and positional embeddings
x = patches + pos_emb

# Transformer encoder
for _ in range(num_layers):
    # Multi-head self-attention
    attn = Dense(embed_dim * num_heads, activation=None)(x)
    attn = np.reshape((patches.shape[1], num_heads, embed_dim))(attn)
    attn = permute((2, 1, 3))(attn)
    attn = Dense(embed_dim)(attn)
    attn = Dropout(0.1)(attn)
    x = x + attn
    x = LayerNormalization()(x)

    # MLP
    mlp = Dense(mlp_dim, activation='gelu')(x)
    mlp = Dropout(0.1)(mlp)
    mlp = Dense(embed_dim)(mlp)
    x = x + mlp
    x = LayerNormalization()(x)

# Classification head
x = Flatten()(x)
outputs = Dense(num_classes, activation='softmax')(x)

# Define the model
model = Model(inputs, outputs)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, batch_size=32)