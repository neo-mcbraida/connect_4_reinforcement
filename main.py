from math import exp
from threading import active_count
from numpy.core.fromnumeric import argmax
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import random
from tensorflow.python.keras.backend import backend

from tensorflow.python.ops.gen_array_ops import batch_to_space

previous_states = []
predictions = []
previous_losses = []
previous_outputs = []

path = "C:/Users/Neo/Documents/gitrepos/Connect4-Python"


optimizer = tf.keras.optimizers.Adam(learning_rate=0.00025, clipnorm=1.0)
loss_func = tf.keras.losses.Huber()

# (None, 6, 7,)
#input_shape = (None, 42, 1,)

def create_model():
    inp_shape = (6, 7, 1)

    # model = keras.Sequential()
    # model.add(layers.Dense(56, activation="relu", input_dim=inp_shape))
    # model.add(layers.Dense(128, activation="relu"))
    # model.add(layers.Dense(128, activation="relu"))
    # model.add(layers.Dense(256, activation="relu"))
    # model.add(layers.Dense(128, activation="relu"))
    # model.add(layers.Dense(56, activation="relu"))
    # model.add(layers.Dense(32, activation="relu"))
    # model.add(layers.Dense(16, activation="relu"))
    # model.add(layers.Dense(7, activation="relu"))
    
    # model.summary()

    # input = layers.Input(shape=inp_shape)
    # move = model(input)    

    # return keras.Model(input, move)
    learning_rate = 0.001
    init = tf.keras.initializers.HeUniform()
    model = keras.Sequential()
    model.add(layers.Conv2D(16, kernel_size=(4, 4), strides=(1, 1), input_shape=inp_shape))
    model.add(layers.ReLU())
    #model.add(layers.Conv2D(16, kernel_size=(4, 4), strides=(1, 1)))
    #model.add(layers.ReLU())
    #model.add(layers.Conv2D(24, (4, 4), strides=(1, 1)))
    #model.add(layers.ReLU())
    #model.add(layers.Conv2D(8, kernel_size=4, strides=1))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation="relu", kernel_initializer=init))
    model.add(layers.Dense(64, activation="relu", kernel_initializer=init))
    model.add(layers.Dense(64, activation="relu", kernel_initializer=init))
    model.add(layers.Dense(48, activation="relu", kernel_initializer=init))
    model.add(layers.Dense(24, activation="relu", kernel_initializer=init))
    model.add(layers.Dense(7, activation="linear"))
    # model.add(layers.Dense(42, input_shape=inp_shape, activation='relu', kernel_initializer=init))
    # model.add(layers.Dense(42, activation='relu', kernel_initializer=init))
    # model.add(layers.Dense(42, activation='relu', kernel_initializer=init))
    # model.add(layers.Dense(28, activation='relu', kernel_initializer=init))
    # model.add(layers.Dense(12, activation='relu', kernel_initializer=init))
    # model.add(layers.Dense(7, activation='relu', kernel_initializer=init))
    model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
    return model


def load_model():
    return tf.keras.models.load_model('C:/Users/Neo/Documents/gitrepos/Connect4-Python/model.h5')

working_model = create_model() # wild
training_model = create_model() # stable
training_model.set_weights(working_model.get_weights())

# working_model = load_model()

def get_expected(predictions, won):
    if won:
        expected = [0, 0, 0, 0, 0, 0, 0]
        expected[np.argmax(predictions)] = 1
    else:
        expected = predictions.numpy()
        expected[0][np.argmax(predictions)] = 0
    return np.array(expected)

def training_prediction(board, turn):
    input = np.array(board)#.flatten()
    input = np.array([input*np.array(turn)])
    previous_states.append(input)
    predictions = working_model(input, training=False)
    prediction = argmax(predictions)
    previous_outputs.append(predictions)
    return prediction

def get_move(board, turn=1):
    #input = np.array(board).flatten()
    #print(input.shape)]
    if turn == 1:
        predictions = working_model(np.array([board]), training=False)
    else:
        predictions = training_model(np.array([board]), training=False)
    prediction = argmax(predictions)
    return prediction


# history = [state, action, nextstate, reward, done]
def train(history : list, batch_size):
    global working_model, training_model
    learning_rate = 0.6
    discount_factor = 0.9

    
    batch = random.sample(history, batch_size)

    states = np.array([event[0] for event in batch])
    next_states = np.array([event[2] for event in batch])
    #rewards = np.array([event[3] for event in history])
    current_q_predicts = working_model.predict(states)
    next_q_predicts = training_model.predict(next_states)

    inps = []
    outs = []

    for i, (state, action, next_state, reward, done) in enumerate(batch):
        #if done:
        max_future_q = reward
        if not done:
            max_future_q += discount_factor * np.max(next_q_predicts[i])

        q_values = current_q_predicts[i]
        q_value = (1 - learning_rate) * (q_values[action]) + learning_rate * max_future_q
        q_values[action] = q_value
        
        inps.append(state)
        outs.append(q_values)
    inps = np.array(inps)
    outs = np.array(outs)
    working_model.fit(inps, outs, batch_size=batch_size, verbose=0, shuffle=True)

def update_weights(won):
    global predictions, previous_states, previous_outputs
    # if won: reward = 100.0
    # else: reward = -100.0
    #loss = reward
    #desired_outs = np.full_like(np.arange(len(previous_states), dtype=float), reward)
    #outputs = working_model(np.array(previous_states))
    for prediction, state in zip(previous_outputs, previous_states):#i, goal in enumerate(desired_out):
        with tf.GradientTape() as tape:
            prediction = training_model(state, training=True)
            expected = np.array(get_expected(prediction, won))
            expected = tf.convert_to_tensor(expected)
            loss = loss_func(expected, prediction)
            grads = tape.gradient(loss, training_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, working_model.trainable_variables))
    previous_states = []
    predictions = []
    #previous_losses = []
    previous_outputs = []

def update_working_weights():
    global working_model
    training_model.set_weights(working_model.get_weights())
