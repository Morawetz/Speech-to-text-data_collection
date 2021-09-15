Module dags.src.models.initial.dataset_loader
=============================================

Functions
---------

    
`load_audio_files(path: str, sampling_rate: int, to_mono: bool) ‑> (<class 'dict'>, <class 'int'>)`
:   Load the audio files and produce a dictionary mapping the audio filenames 
    to numpy arrays of the audio sampled at the given sample rate.
    Inputs: 
    path - a path to the directory that contains the audio files
    sample_rate - the sampling rate for the audio files
    to_mono - a boolean value denoting whether to convert signal to mono
    Returns:
    audio_files - audios - a dictionary mapping the wav file names to the sampled audio array
    max_length - the maximum length of a sampled audio array in our dataset

    
`load_spectrograms_with_transcripts(mfcc_features: dict, encoded_transcripts: dict, path: str)`
:   Loads the spectrogram images as numpy arrays
    Inputs:
    mfcc_features - a python dictionary mapping the wav file names to the mfcc 
                    coefficients of the sampled audio files
    encoded_transcripts - a python dictionary mapping the wav file names to the 
                          encoded transcripts of those audio files.
    path - the path to the directory that contains the spectrogram images
    Returns:
    X_train - a numpy array containing the mfcc spectrograms of the sampled audio files
    y_train - a numpy array containing the encoded transcripts of the sampled audio files
              in the same order as they appear in X_train

    
`load_spectrograms_with_transcripts_in_batches(mfcc_features: dict, encoded_transcripts: dict, batch_size: int, batch_no: int, path: str)`
:   Loads the spectrogram images as numpy arrays
    Inputs:
    mfcc_features - a python dictionary mapping the wav file names to the mfcc 
                    coefficients of the sampled audio files
    encoded_transcripts - a python dictionary mapping the wav file names to the 
                          encoded transcripts of those audio files.
    batch_size - the size of the batch when loading
    batch_no - the index of the batch
    path - the path to the directory that contains the spectrogram images
    Returns:
    X_train - a numpy array containing the mfcc spectrograms of the sampled audio files
    y_train - a numpy array containing the encoded transcripts of the sampled audio files
              in the same order as they appear in X_train

    
`load_transcripts(filepath: str) ‑> dict`
:   Load the transcript file and produce a dictionary mapping the audio filenames 
    to the transcripts for those audio files.
    Inputs: 
    filepath - a path to the transcript file
    Returns:
    transcripts - a python dictionary mapping the wav file names to the transcripts
                  of those audio files.