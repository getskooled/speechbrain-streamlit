import io
import pydub
import resampy

import librosa
import librosa.display

import streamlit as st
import numpy as np

from matplotlib import pyplot as plt
from scipy.io import wavfile

import torch
import torchaudio
import speechbrain as sb

from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.pretrained import SpectralMaskEnhancement

SAMPLE_RATE = 16000

asr_model = EncoderDecoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-en",
                                          savedir="asr-wav2vec2-commonvoice-en")

enhancer = SpectralMaskEnhancement.from_hparams(source="speechbrain/metricgan-plus-voicebank",
                                                 savedir="tmp_dir")


plt.rcParams["figure.figsize"] = (12, 10)


def create_audio_player(audio_data, sample_rate):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=sample_rate, data=audio_data)
    return virtualfile

@st.cache
def handle_uploaded_audio_file(uploaded_file):
    a = pydub.AudioSegment.from_file(
        file=uploaded_file, format=uploaded_file.name.split(".")[-1]
    )

    channel_sounds = a.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max

    return fp_arr[:, 0], a.frame_rate

def add_h_space():
    st.markdown("<br></br>", unsafe_allow_html=True)

def plot_wave(source, sample_rate):
    fig, ax = plt.subplots()
    img = librosa.display.waveshow(source, sr=sample_rate, x_axis="time", ax=ax)
    return plt.gcf()

def do_audio_processing(source, sample_rate, option):

    batch = source.unsqueeze(0)
    rel_length = torch.tensor([1.0])

    if option == 'Speech Recognition':
        st.markdown(
        f"<h4 style='text-align: center; color: black;'>Audio</h5>",
        unsafe_allow_html=True,)

        st.audio(create_audio_player(source.cpu().detach().numpy(), sample_rate))
        result = asr_model.transcribe_batch(batch, rel_length)
        st.markdown("---")
        st.write(result)

    elif option == 'Speech Enhancement':
        cols = [1, 1]
        col1, col2 = st.columns(cols)

        with col1:
            st.markdown(
                f"<h4 style='text-align: center; color: black;'>Original</h5>",
                unsafe_allow_html=True,
            )
            st.audio(create_audio_player(source.cpu().detach().numpy(), sample_rate))
        with col2:
            st.markdown(
                f"<h4 style='text-align: center; color: black;'>Wave plot </h5>",
                unsafe_allow_html=True,
            )
            st.pyplot(plot_wave(source.cpu().detach().numpy(), sample_rate))
            add_h_space()

        cols = [1, 1]
        col1, col2 = st.columns(cols)
        enhanced = enhancer.enhance_batch(batch, rel_length)
        enhanced = enhanced.cpu().detach().numpy()
        with col1:
            st.markdown(
                f"<h4 style='text-align: center; color: black;'>Original</h5>",
                unsafe_allow_html=True,
            )
            st.audio(create_audio_player(np.transpose(enhanced, (1,0)), sample_rate))
        with col2:
            st.markdown(
                f"<h4 style='text-align: center; color: black;'>Wave plot </h5>",
                unsafe_allow_html=True,
            )
            st.pyplot(plot_wave(enhanced, sample_rate))
            add_h_space()

def action(file_uploader, option):
    if file_uploader is not None:
      source, sample_rate = handle_uploaded_audio_file(file_uploader)
      source = resampy.resample(source, sample_rate, SAMPLE_RATE, axis=0, filter='kaiser_best')
      source = torch.FloatTensor(source)
      do_audio_processing(source, SAMPLE_RATE, option)

def main():
    placeholder = st.empty()
    placeholder2 = st.empty()
    placeholder.markdown(
        "# Processing of sound and speech by SpeechBrain framework\n"
        "### Select the processing procedure in the sidebar.\n"
        "Once you have chosen processing procedure, select or upload an audio file\n. "
        'Then click "Apply" to start! \n\n'
    )
    placeholder2.markdown(
        "After clicking start,the result of the selected procedure are visualized."
    )

    option = st.sidebar.selectbox('Audio Processing Task', options=('Speech Recognition', 'Speech Enhancement'))
    st.sidebar.markdown("---")
    st.sidebar.markdown("(Optional) Upload an audio file here:")
    file_uploader = st.sidebar.file_uploader(
        label="", type=[".wav", ".wave", ".flac", ".mp3", ".ogg"]
    )
    st.sidebar.markdown("---")
    if st.sidebar.button("Apply"):
        placeholder.empty()
        placeholder2.empty()
        action(file_uploader=file_uploader,
               option=option)


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Speech brain audio file processing")
    main()