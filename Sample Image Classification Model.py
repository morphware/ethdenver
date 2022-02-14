#!/usr/bin/env python
# coding: utf-8

# #### Import Dependicies

# In[5]:


import tensorflow as     tf

from   tensorflow import keras


# ## Pre-processing
# ### Train-test Split

# In[6]:


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()


# In[7]:


validation_set_size     = 5000
maximum_pixel_intensity = 255.0


# In[8]:


X_valid, X_train = X_train_full[:validation_set_size] / maximum_pixel_intensity, X_train_full[validation_set_size:] / maximum_pixel_intensity
y_valid, y_train = y_train_full[:validation_set_size], y_train_full[validation_set_size:]
X_test           = X_test / maximum_pixel_intensity


# ### Categories

# In[9]:


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


# ### Model

# In[10]:


model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation='relu'),
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])


# In[11]:


model.compile(loss='sparse_categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])


# ### `keras` Callbacks

# #### `Checkpoint` Callback

# In[12]:


outfile_pathname  = 'trained_model.h5' 
checkpoint_cb     = keras.callbacks.ModelCheckpoint(outfile_pathname, save_best_only=True)


# #### `EarlyStopping` Callback

# In[13]:


early_stopping_cb = keras.callbacks.EarlyStopping(patience=10)


# ## Processing
# ### Actual Training Process

# In[17]:


history = model.fit(X_train, y_train, epochs=20,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])


# ## Post-processing
# ### Charting model accurracy

# In[ ]:


import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()



