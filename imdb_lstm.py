from comet_ml import Experiment

#create an experiment with your api key
experiment = Experiment(api_key="361WL8rDtXAqJ3EEJqLC07JtG",
                        project_name='imdb',
                        auto_param_logging=False)

import numpy
from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Flatten
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

params = {
            'batch_size':128,
            'num_classes':1,
            'epochs':20,
            'optimizer':'adam',
            'activation':'relu',
            'batch_size':128,
            'top_words':5000,
            'max_review_length':500,
            'embedding_vector_length':100
}

(X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=params['top_words'])

X_train = sequence.pad_sequences(X_train, maxlen=params['max_review_length'])
X_test = sequence.pad_sequences(X_test, maxlen=params['max_review_length'])

# create the model
model = Sequential()
model.add(Embedding(params['top_words'], params['embedding_vector_length'], input_length=params['max_review_length']))
model.add(LSTM(100))

model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer=params['optimizer'], metrics=['accuracy'])
#print model.summary() to preserve automatically in `Output` tab
print(model.summary())
params.update({'total_number_of_parameters':model.count_params()})

#will log metrics with the prefix 'train_'
with experiment.train():
    model.fit(X_train, y_train,
              epochs=params['epochs'],
              batch_size=params['batch_size'],
              verbose=1,
              validation_data=(X_test, y_test))

#will log metrics with the prefix 'test_'
with experiment.test():
    loss, accuracy = model.evaluate(X_test, y_test)
    metrics = {
        'loss':loss,
        'accuracy':accuracy
    }
    experiment.log_multiple_metrics(metrics)


experiment.log_multiple_params(params)
experiment.log_dataset_hash(X_train) #creates and logs a hash of your data
