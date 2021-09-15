Module dags.src.models.initial.resize_and_augment
=================================================

Functions
---------

    
`augment_audio(audios: dict, sample_rate: int) ‑> dict`
:   Here we shift the wave by sample_rate/10 factor. This will move the wave to the 
    right by given factor along time axis. For achieving this I have used numpy’s 
    roll function to generate time shifting, time stretching, and pitch shifting.
    Inputs: 
    audios - a dictionary mapping the wav file names to the sampled audio array
    sample_rate - the sample rate for the audio
    Returns:
    audios - a python dictionary mapping the wav file names to the augmented 
            audio samples

    
`equalize_transcript_dimension(mfccs, encoded_transcripts, truncate_len)`
:   Make all transcripts have equal number of characters by padding the the short
    ones with spaces

    
`resize_audios_mono(audios: dict, max_length: int) ‑> dict`
:   Here we pad the sampled audio with zeros so tha all of the sampled audios 
    have equal length
    Inputs: 
    audios - a dictionary mapping the wav file names to the sampled audio array
    max_length - the maximum length of a sampled audio array in our dataset
    Returns:
    audios - a python dictionary mapping the wav file names to the padded
            audio samples