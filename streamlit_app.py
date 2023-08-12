!pip install openai

import openai
import torchaudio
import streamlit as st
from speechbrain.pretrained import EncoderDecoderASR

st.title('Transcription Corrector')

# File uploader for audio file
uploaded_file = st.file_uploader("Upload an audio file", type=['wav', 'mp3', 'm4a'])

if uploaded_file is not None:
    # Load ASR model
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-crdnn-rnnlm-librispeech",
        savedir="tmpdir",
    )

    # Transcribe the uploaded audio
    signal, sample_rate = torchaudio.load(uploaded_file)
    transcription = asr_model.transcribe_batch(signal)

    st.subheader('Original Transcription:')
    st.write(transcription)

    # Get OpenAI API key from secrets
    openai_api_key = st.secrets['openai_api_key']
    openai.api_key = openai_api_key

    # Define the conversation with the transcription assistant
    messages = [
        {"role": "system", "content": "You are a transcription assistant."},
        {"role": "user", "content": f"The following is a transcription with some errors, correct these errors:\n\n{transcription}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages
    )

    corrected_transcription = response['choices'][0]['message']['content']

    st.subheader('Corrected Transcription:')
    st.write(corrected_transcription)
