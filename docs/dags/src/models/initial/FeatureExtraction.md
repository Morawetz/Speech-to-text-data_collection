Module dags.src.models.initial.FeatureExtraction
================================================

Classes
-------

`FeatureExtraction()`
:   

    ### Methods

    `extract_features(self, audios: dict, sample_rate: int) ‑> dict`
    :   The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of 
        features (usually about 10–20) which concisely describe the overall shape of a 
        spectral envelope. It models the characteristics of the human voice.
        We compute the Mel frequency cepstral coefficients for each audio file.
        Inputs: 
        audios - a dictionary mapping the wav file names to the sampled audio array
        sample_rate - the sample rate for the audio
        Returns:
        mfcc_features - a python dictionary mapping the wav file names to the mfcc 
                        coefficients of the sampled audio files

    `save_mel_spectrograms(self, audios: dict, sample_rate: int, path: str) ‑> int`
    :   The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of 
        features (usually about 10–20) which concisely describe the overall shape of a 
        spectral envelope. It models the characteristics of the human voice.
        A Spectrogram captures the nature of the audio as an image by decomposing 
        it into the set of frequencies that are included in it.
        We plot the MFCC spectrogram for each audio file, and save the plots as .png 
        image files to the given target directory.
        Inputs: 
        mfccs - a python dictionary mapping the wav file names to the mfcc 
                coefficients of the sampled audio files
        sample_rate - the sampling rate for the audio
        path - the file path to the target directory
        Returns:
        0 if the spectrograms were saved successfully, and 
        raises a FileNotFoundError if the given path doesn't exist

    `save_mfcc_spectrograms(self, mfccs: dict, sample_rate: int, path: str) ‑> int`
    :   The Mel frequency cepstral coefficients (MFCCs) of a signal are a small set of 
        features (usually about 10–20) which concisely describe the overall shape of a 
        spectral envelope. It models the characteristics of the human voice.
        A Spectrogram captures the nature of the audio as an image by decomposing 
        it into the set of frequencies that are included in it.
        We plot the MFCC spectrogram for each audio file, and save the plots as .png 
        image files to the given target directory.
        Inputs: 
        mfccs - a python dictionary mapping the wav file names to the mfcc 
                coefficients of the sampled audio files
        sample_rate - the sampling rate for the audio
        path - the file path to the target directory
        Returns:
        0 if the spectrograms were saved successfully, and 
        raises a FileNotFoundError if the given path doesn't exist