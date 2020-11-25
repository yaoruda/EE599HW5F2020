import h5py
import numpy as np
from tensorflow.keras.layers import Input, Dense, GRU
from tensorflow.keras import Model

##### generate toy data

#### example from lecture
train_seq_length = 4
feature_dim = 2
num_seqs = 8
DEMO = True

#### example from after lecture - trained to show how RNN learns and works well.
train_seq_length = 16 #4
feature_dim = 2
num_seqs = 500 #8
DEMO = False

x =  np.random.randint(0, high=2, size = (num_seqs * train_seq_length, feature_dim) )
x = np.sign( x - 0.5 )
y = np.sum( ( x == np.roll(x, 1, axis = 0) ), axis = 1 )
### y[n] = number of agreements between x[n], x[n-1]
x = x.reshape( (num_seqs, train_seq_length, feature_dim) )
y = y.reshape( (num_seqs, train_seq_length, 1) )


######  Define/Build/Train Training Model
training_in_shape = x.shape[1:]
training_in = Input(shape=training_in_shape)
# training_in = Input(batch_shape=(None,train_seq_length,feature_dim)) this works too
foo = GRU(4, return_sequences=True, stateful=False)(training_in)
training_pred = Dense(1)(foo)

training_model = Model(inputs=training_in, outputs=training_pred)
training_model.compile(loss='mean_squared_error', optimizer='adam')
training_model.summary()

training_model.fit(x, y, batch_size=2, epochs=100)

##### define the streaming-infernece model
streaming_in = Input(batch_shape=(1,None,feature_dim))  ## stateful ==> needs batch_shape specified
foo = GRU(4, return_sequences=False, stateful=True )(streaming_in)
streaming_pred = Dense(1)(foo)
streaming_model = Model(inputs=streaming_in, outputs=streaming_pred)

streaming_model.compile(loss='mean_squared_error', optimizer='adam')
streaming_model.summary()

##### copy the weights from trained model to streaming-inference model
training_model.save_weights('weights.hd5', overwrite=True)
streaming_model.load_weights('weights.hd5')

if DEMO:
    ##### demo the behaivor
    print('\n\n******the streaming-inference model can replicate the sequence-based trained model:\n')
    for s in range(num_seqs):
        print(f'\n\nRunning Sequence {s} with STATE RESET:\n')
        in_seq = x[s].reshape( (1, train_seq_length, feature_dim) )
        seq_pred = training_model.predict(in_seq)
        seq_pred = seq_pred.reshape(train_seq_length)
        for n in range(train_seq_length):
            in_feature_vector = x[s][n].reshape(1,1,feature_dim)
            single_pred = streaming_model.predict(in_feature_vector)[0][0]
            print(f'Seq-model Prediction, Streaming-Model Prediction, difference [{n}]: {seq_pred[n] : 3.2f}, {single_pred : 3.2f}, {seq_pred[n] - single_pred: 3.2f}')
        streaming_model.reset_states()

    print('\n\n******streaming-inference state needs reset between sequences to replicate sequence-based trained model:\n')
    for s in range(num_seqs):
        print(f'\n\nRunning Sequence {s} with NO STATE RESET:\n')
        in_seq = x[s].reshape( (1, train_seq_length, feature_dim) )
        seq_pred = training_model.predict(in_seq)
        seq_pred = seq_pred.reshape(train_seq_length)
        for n in range(train_seq_length):
            in_feature_vector = x[s][n].reshape(1,1,feature_dim)
            single_pred = streaming_model.predict(in_feature_vector)[0][0]
            print(f'Seq-model Prediction, Streaming-Model Prediction, difference [{n}]: {seq_pred[n] : 3.2f}, {single_pred : 3.2f}, {seq_pred[n] - single_pred: 3.2f}')
        #### NO STATE RESET HERE: streaming model will treat multiples sequences as one long sequence,
        #### so after first sequence, the streaming output will differ, difference will decay with time from start up as effect of intial state fades

    for s in range(2):
        N = np.random.randint(1, 10)
        print(f'\n\n******streaming-inference can work on an sequences of indefinite length -- running length {N}:\n')
        for n in range(N):
            x_sample =  np.random.randint(0, high=2, size = ( 1, 1, feature_dim) )
            x_sample = np.sign( x_sample - 0.5 )
            single_pred = streaming_model.predict(x_sample)[0][0]
            print(f'Streaming-Model Prediction[{n}]:  {single_pred : 3.2f}')
        streaming_model.reset_states()
