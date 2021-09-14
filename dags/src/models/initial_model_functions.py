import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import pickle
import logging
import os
from .initial.dataset_loader import load_audio_files, load_transcripts, load_spectrograms_with_transcripts, load_spectrograms_with_transcripts_in_batches
from .initial.resize_and_augment import resize_audios_mono, augment_audio, equalize_transcript_dimension
from .initial.FeatureExtraction import FeatureExtraction
from .initial.transcript_encoder import fit_label_encoder, encode_transcripts, decode_predicted
# from models import model_1, model_2, model_3, model_4
from .initial.new_model import my_model
from jiwer import wer
import librosa   #for audio processing
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import warnings
import mlflow
import mlflow.keras
import os
import logging

def load_preprocess(**kwargs):
    # len(os.listdir('../../../../data/'))
    sample_rate = 44100
    logging.info("Sample rate specified")

    audio_files, maximum_length = load_audio_files('/usr/local/airflow/data/model-data/train/wav/', sample_rate, True)
    logging.info('loaded audio files')

    print("The longest audio is", maximum_length/sample_rate, 'seconds long')
    print("max length", maximum_length)

    demo_audio = list(audio_files.keys())[0]

    transcripts = load_transcripts("/usr/local/airflow/data/model-data/train/trsTrain.txt")
    logging.info('loaded transcripts')

    audio_files = resize_audios_mono(audio_files, 440295)
    print("resized shape", audio_files[demo_audio].shape)

    # audio_files = augment_audio(audio_files, sample_rate)
    # print("augmented shape", audio_files[demo_audio].shape)

    char_encoder = fit_label_encoder(transcripts)
    transcripts_encoded = encode_transcripts(transcripts, char_encoder)
    enc_aug_transcripts = equalize_transcript_dimension(audio_files, transcripts_encoded, 200)


    def load_data(audio_files, encoded_transcripts):
        X_train = []
        y_train = []
        for audio in audio_files:
            X_train.append(audio_files[audio])
            y_train.append(encoded_transcripts[audio])
        return np.array(X_train), np.array(y_train)

    X_train, y_train = load_data(audio_files, enc_aug_transcripts)
    print(X_train.shape, y_train.shape)
    X_val, y_val = X_train[-10:], y_train[-10:]
    X_test, y_test = X_train[:10], y_train[:10]#X_train[-20:-10], y_train[-20:-10]
    X_train, y_train = X_train[:-20], y_train[:-20]
    
    return X_train, y_train, X_test, y_test, X_val, y_val


def construct_model(num_classes, input_shape):

	# construct model framework
	# source: https://keras.io/examples/mnist_cnn/

	model = Sequential()
	model.add(Conv2D(32, kernel_size=(3, 3),
					 activation='relu',
					 input_shape=input_shape))
	model.add(Conv2D(64, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.25))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(num_classes, activation='softmax'))

	model.compile(loss=keras.losses.categorical_crossentropy,
				  optimizer=keras.optimizers.Adadelta(),
				  metrics=['accuracy'])
	return model


def fit_model(**kwargs):
    
    # fit model along preprocessed data and constructed model framework

    ti = kwargs['ti']
    loaded = ti.xcom_pull(task_ids='load_preprocess')

    logging.info('variables successfully fetched from previous task')

    X_train = loaded[0]
    y_train = loaded[1]
    X_test = loaded[2]
    y_test = loaded[3]
    X_val = loaded[4]
    y_val = loaded[5]

    import tensorflow as tf
    from .initial.new_model import LogMelgramLayer, CTCLayer
    model = tf.keras.models.load_model('/usr/local/airflow/dags/src/models/new_model_v1_2000.h5', 
                                        custom_objects = {
                                            'LogMelgramLayer': LogMelgramLayer ,
                                            'CTCLayer': CTCLayer}
                                        )
    print(model.summary())

    import numpy as np
   
    
    X_train = X_train.tolist()
    X_val = X_val.tolist()
    logging.info("Variables successfully converted to list")
    # cant convert to tensor
    # X_train = tf.convert_to_tensor(
    # X_train, dtype=None, dtype_hint=None, name=None
    # )
    # X_val = tf.convert_to_tensor(
    # X_val, dtype=None, dtype_hint=None, name=None   
    # )

    X_train = np.asarray(X_train)
    X_val = np.asarray(X_val)
    print("++++++++++++++++++++++++++ converted to numpy")
    # we have a numpy object array filled with np float arrays
    history = model.fit([X_train, y_train], 
                        validation_data = [X_val, y_val], 
                        batch_size = kwargs['batch_size'], epochs = kwargs['epochs'])
                        # bs=25, epochs=100

    model.save(os.getcwd() + kwargs['initial_model_path'])
