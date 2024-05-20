import os
import tensorflow as tf
from keras.utils.vis_utils import plot_model

# Manually add Graphviz to the PATH (if necessary)
os.environ["PATH"] += os.pathsep + 'C:/Program Files/Graphviz/bin'

# Create a simple model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, input_shape=(20,), activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Plot the model
plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
print("Model plot saved as model.png")

#import pydot
#print(pydot.__version__)