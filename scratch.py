import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

WINDOW_SIZE = 20
THRESHOLD = 100

with open('data.json') as f:
    data = tf.squeeze(tf.ragged.constant(json.load(f)), axis=1)

# split into train and test sets
train_size = int(data.shape[0] * 0.67)
train, test = data[0:train_size,:], data[train_size:data.shape[0],:]

# def create_data_labels(ts):
#     tuples, labels = [], []
#     for i in range(0, len(ts) - WINDOW_SIZE, WINDOW_SIZE):
#         tuples.append(ts[i : i + WINDOW_SIZE])
#         labels.append(1 if any(val > THRESHOLD for val in ts[i : i + WINDOW_SIZE]) else 0)
    
#     return tf.convert_to_tensor(tuples), tf.convert_to_tensor(labels)

def create_data_labels(ts):
    # rnn style
    # ts is a list of a time series
    tuples, labels = [], []
    for i in range(0, len(ts) - WINDOW_SIZE):
        tuples.append(ts[i : i + WINDOW_SIZE])
        labels.append(1 if ts[i + WINDOW_SIZE] >= 100 else 0)
    
    return tf.convert_to_tensor(tuples), tf.convert_to_tensor(labels)

# print(tf.concat(list(zip(*[create_data_labels(list(sample)) for sample in data]))[1], axis=0))

def split(data):
    return [tf.concat(lst, axis=0) for lst in zip(*[create_data_labels(list(sample)) for sample in data])]

X_train, y_train = split(train)
X_test, y_test = split(test)

model = tf.keras.Sequential([
        layers.Dense(2, activation="relu", name="layer1"),
        layers.Dense(3, activation="relu", name="layer2"),
        layers.Dense(2, name="layer3"),
    ])


model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )

model.fit(
    X_train,
    y_train,
    batch_size=64,
    epochs=2
)

model.summary()

print("test acc: ", model.evaluate(X_test, y_test, batch_size=128)[1])