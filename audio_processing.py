import io
import pydub

import librosa
import librosa.display

import numpy as np
import resampy

from scipy.io import wavfile

from matplotlib import pyplot as plt

import torch
import torchaudio
from speechbrain.dataio.dataio import read_audio
from speechbrain.pretrained import SpeakerRecognition
from speechbrain.pretrained import EncoderDecoderASR
from speechbrain.pretrained import SpectralMaskEnhancement
from speechbrain.pretrained import SepformerSeparation


def load_audio_sample(file):
    y, sr = librosa.load(file, sr=22050)
    return y, sr


def create_audio_player(audio_data, sample_rate):
    virtualfile = io.BytesIO()
    wavfile.write(virtualfile, rate=sample_rate, data=audio_data)
    return virtualfile

def handle_uploaded_audio_file(file_name):
    a = pydub.AudioSegment.from_file(
        file=file_name, format=file_name.split(".")[-1])

    channel_sounds = a.split_to_mono()
    samples = [s.get_array_of_samples() for s in channel_sounds]

    fp_arr = np.array(samples).T.astype(np.float32)
    fp_arr /= np.iinfo(samples[0].typecode).max
    return fp_arr[:, 0], a.frame_rate


def plot_transformation(y, sr, option_name):
    D = librosa.stft(y)  # STFT of y
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_db, x_axis='time', y_axis='linear', ax=ax)
    ax.set(title=option_name)
    fig.colorbar(img, ax=ax, format="%+2.f dB")

def plot_wave(y, sr):
    fig, ax = plt.subplots()
    img = librosa.display.waveshow(y, sr=sr, x_axis="time", ax=ax)
    return plt.gcf()


if __name__ == '__main__':
    asr_model = EncoderDecoderASR.from_hparams(
        source="speechbrain/asr-wav2vec2-commonvoice-en",
        savedir="asr-wav2vec2-commonvoice-en")

    source, sample_rate = handle_uploaded_audio_file("interview.mp3")

    enhancer = SpectralMaskEnhancement.from_hparams(
        source="speechbrain/metricgan-plus-voicebank",
        savedir="tmp_dir")

    decoded_audio = resampy.resample(source, sample_rate, 16000, axis=0, filter='kaiser_best')
    decoded_audio = torch.FloatTensor(decoded_audio)
    batch = decoded_audio.unsqueeze(0)
    rel_length = torch.tensor([1.0])

    enhanced = enhancer.enhance_batch(batch, rel_length)
    enhanced = enhanced.cpu().detach().numpy()
    #enhanced = np.transpose(enhanced, (1,0))

    #plot_wave(enhanced, 16000)

    plot_transformation(enhanced, 16000, 'enchanced')

    plt.show()

    wavfile.write("example.wav", 16000, enhanced)



    result = asr_model.transcribe_batch(batch, rel_length)
    print(result)

    enchanser = SpectralMaskEnhancement.enhance_file()

    '''
    waveform = self.load_audio(path)
    # Fake a batch:
    batch = waveform.unsqueeze(0)
    rel_length = torch.tensor([1.0])
    predicted_words, predicted_tokens = self.transcribe_batch(
        batch, rel_length


    asr_model.transcribe_file()
   '''


