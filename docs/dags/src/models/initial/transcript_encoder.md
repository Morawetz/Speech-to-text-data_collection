Module dags.src.models.initial.transcript_encoder
=================================================

Functions
---------

    
`decode_predicted(pred, encoder)`
:   remove the blank character from the predictions and decode the integers back to
    amharic characters.

    
`encode_transcripts(transcripts: dict, encoder: sklearn.preprocessing._label.LabelEncoder) ‑> dict`
:   This function takes an sklearn label encoder that has already been fitted with
    the amharic characters from the transcripts, along with the original transcript
    and encodes the transcripts for each audio using the given label encoder.
    Input:
    transcripts - a python dictionary mapping the wav file names to the transcripts
                  of those audio files.
    encoder - an sklearn label encoder that has been fitted with all the characters 
              in the transcripts.
    Returns:
    transcripts_encoded - a python dictionary mapping the wav file names to the encoded transcripts
                          of those audio files.

    
`fit_label_encoder(transcripts: dict) ‑> sklearn.preprocessing._label.LabelEncoder`
:   This function encodes the amharic characters in the given dictiionary of 
    transcripts into integers.
    Input:
    transcripts - a python dictionary mapping the wav file names to the transcripts
                  of those audio files.
    Returns:
    encoder - an sklearn label encoder that has been fitted with all the characters 
    in the transcripts.