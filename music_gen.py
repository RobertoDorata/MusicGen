import matplotlib.pyplot as plt
import tensorflow as tf
from midi_processing import *
from tensorflow.keras.layers import Dense, Input, LSTM, Reshape, RepeatVector
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

tf.random.set_seed(7)
np.random.seed(7)


def preprocessing_for_original_corpus_analysis(__corpus, __dictionary):
    hot_encoded = np.zeros((len(__corpus), len(__dictionary)))
    values = list(__dictionary.values())
    for _i, word in enumerate(__corpus):
        hot_encoded[_i, values.index(word)] = 1

    return np.argmax(hot_encoded, axis=-1)


# split the dataset into training and testing
def split_dataset(_dataset, _labels, test_size):
    indices = np.arange(_dataset.shape[0])
    np.random.shuffle(indices)
    new_dataset = _dataset[indices, :, :]
    new_labels = _labels[:, indices, :]
    split_point = int(test_size * _dataset.shape[0])
    train_dataset = new_dataset[split_point:, :, :]
    train_labels = new_labels[:, split_point:, :]
    test_dataset = new_dataset[:split_point, :, :]
    test_labels = new_labels[:, :split_point, :]
    return train_dataset, train_labels, test_dataset, test_labels


# plot the musical elements and their offset (their position in the song)
def plot_analysis(values):
    plt.figure()
    plt.xlabel('offset')
    plt.ylabel('musical element')
    plt.plot(list(range(len(values))), values)
    plt.show()


# plot the training and validation loss
def plot_history(network_history):
    plt.figure()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.plot(x_plot, network_history.history['loss'])
    plt.plot(x_plot, network_history.history['val_loss'])
    plt.legend(['Training', 'Validation'])

    plt.show()


# read the midi file describing the song "Nuvole bianche".
dataset, labels, number_of_elements, chords, dictionary, _corpus = midi_to_data('Nuvole_Bianche.mid')

original_data_for_analysis = preprocessing_for_original_corpus_analysis(_corpus, dictionary)

# plot the musical elements of "Nuvole bianche"
plot_analysis(original_data_for_analysis)

# split the dataset into training and validation with a ratio of 80% (training) - 20% (validation)
training_dataset, training_labels, val_dataset, val_labels = split_dataset(dataset, labels, 0.2)

# number of hidden LSTM states
number_of_hidden_states = 50

# reshape the data to fit the LSTM layer. The LSTM layer expects each element to be
# a vector of dimension number_of_element.
reshape_layer = Reshape((1, number_of_elements))
# LSTM layer
LSTM_layer = LSTM(number_of_hidden_states, return_state=True)
# softmax layer used to predict the next element (multi-class classification problem)
softmax_layer = Dense(number_of_elements, activation='softmax')

# input layers
input_layer_1 = Input(shape=(training_dataset.shape[1], number_of_elements))
ith_initial_h_1 = Input(shape=(number_of_hidden_states,), name='initial_h')
ith_initial_c_1 = Input(shape=(number_of_hidden_states,), name='initial_c')

next_h_1 = ith_initial_h_1
next_c_1 = ith_initial_c_1

# We build the LSTM considering its unrolled form. We use a FOR loop and the functional API.
outputs_1 = []

# each instance is a sequence of musical elements
length_of_each_instance = training_dataset.shape[1]

for i in range(length_of_each_instance):
    ith_input = (lambda ith: ith[:, i, :])(input_layer_1)
    ith_input = reshape_layer(ith_input)
    next_h_1, _, next_c_1 = LSTM_layer(ith_input, initial_state=[next_h_1, next_c_1])
    out = softmax_layer(next_h_1)
    outputs_1.append(out)

model_1 = Model(inputs=[input_layer_1, ith_initial_h_1, ith_initial_c_1], outputs=outputs_1)

model_1.summary()

# TRAINING:
# Optimizer:        Adam  with a learning rate ten times higher than the standard one (0.001)
#                   which is adjusted during training with the decay parameter;
#                   Beta parameters are used with default values;
# Loss function:    categorical cross-entropy (multi-class classification problem);
# Metrics:          categorical accuracy.
model_1.compile(optimizer=Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01),
                loss='categorical_crossentropy', metrics=['accuracy'])

# get the number of training and validation instances
number_of_training_instances = training_dataset.shape[0]
number_of_validation_instances = val_dataset.shape[0]

# we use zeros as initial states for the LSTM layer
initial_h_1 = np.zeros((number_of_training_instances, number_of_hidden_states))
initial_c_1 = np.zeros((number_of_training_instances, number_of_hidden_states))
initial_h_1_val = np.zeros((number_of_validation_instances, number_of_hidden_states))
initial_c_1_val = np.zeros((number_of_validation_instances, number_of_hidden_states))
n_epochs = 60

history = model_1.fit([training_dataset, initial_h_1, initial_c_1], list(training_labels), epochs=n_epochs,
                      validation_data=([val_dataset, initial_h_1_val, initial_c_1_val], list(val_labels)))

# plot the training and validation loss
x_plot = list(range(1, n_epochs + 1))
plot_history(history)

# after training, the output has to be used as the next input
# for a time series equal to the musical elements to be generated
input_layer_2 = Input(shape=(1, number_of_elements))
ith_initial_h_2 = Input(shape=(number_of_hidden_states,), name='a0')
ith_initial_c_2 = Input(shape=(number_of_hidden_states,), name='c0')

next_h_2 = ith_initial_h_2
next_c_2 = ith_initial_c_2
x = input_layer_2

outputs_2 = []

# a sequence of 12 musical elements is generated
# this affects the duration of the generated MIDI file
generated_sequence_length = 12

# a sequence of 125 musical elements (for analysis purposes)
# generated_sequence_length = 125

for i in range(generated_sequence_length):
    next_h_2, _, next_c_2 = LSTM_layer(x, initial_state=[next_h_2, next_c_2])
    out = softmax_layer(next_h_2)
    outputs_2.append(out)
    x = tf.math.argmax(out, axis=-1)
    x = tf.one_hot(x, number_of_elements)
    x = RepeatVector(1)(x)

model_2 = Model(inputs=[input_layer_2, ith_initial_h_2, ith_initial_c_2], outputs=outputs_2)

model_2.summary()

# random values are used as initial input
seed_input = np.random.rand(1, 1, number_of_elements)
seed_h = np.random.rand(1, number_of_hidden_states)
seed_c = np.random.rand(1, number_of_hidden_states)

predictions = model_2.predict([seed_input, seed_h, seed_c])
# For each prediction, we take the index of maximum likelihood
predictions = np.argmax(predictions, axis=-1)

# We convert the predictions so that we get a midi file,
# placing the predicted struct together with the original chords.
midi_output = data_to_midi(predictions, chords, dictionary)

midi_output.write('midi', "output.midi")

# plot the generated musical elements
plot_analysis(predictions)
