import sys
import os

import sys
import os

from setuptools import setup, find_packages
setup(
    name = 'initial',
    packages = find_packages(),
)

from initial import dataset_loader

from . import dataset_loader
from . import FeatureExtraction


from .initial.dataset_loader import load_audio_files, load_transcripts, load_spectrograms_with_transcripts, load_spectrograms_with_transcripts_in_batches
from .resize_and_augment import resize_audios_mono, augment_audio, equalize_transcript_dimension
from .FeatureExtraction import FeatureExtraction
from .transcript_encoder import fit_label_encoder, encode_transcripts, decode_predicted
# from models import model_1, model_2, model_3, model_4
from .new_model import my_model
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
# len(os.listdir('../../ALFFA_PUBLIC/ASR/AMHARIC/data/train/wav/'))

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

    audio_files = augment_audio(audio_files, sample_rate)
    print("augmented shape", audio_files[demo_audio].shape)

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

mlflow.set_tracking_uri('../')
mlflow.keras.autolog()


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
    from .new_model import LogMelgramLayer, CTCLayer
    model = tf.keras.models.load_model('/usr/local/airflow/dags/src/models/new_model_v1_2000.h5', 
                                        custom_objects = {
                                            'LogMelgramLayer': LogMelgramLayer ,
                                            'CTCLayer': CTCLayer}
                                        )
    print(model.summary())

    history = model.fit([X_train, y_train], 
                        validation_data = [X_val, y_val], 
                        batch_size = kwargs['batch_size'], epochs = kwargs['epochs'])
                        # bs=25, epochs=100

    model.save(os.getcwd() + kwargs['initial_model_path'])

if __name__ == "__main__":
    load_preprocess()