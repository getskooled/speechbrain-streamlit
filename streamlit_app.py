import os
import streamlit as st
import boto3
import time
from moviepy.editor import VideoFileClip
import speechbrain as sb
import torchaudio

AWS_ACCESS_KEY_ID = os.environ["AWS_ACCESS_KEY_ID"]
AWS_SECRET_ACCESS_KEY = os.environ["AWS_SECRET_ACCESS_KEY"]
UPLOAD_AWS_BUCKET_NAME = os.environ["UPLOAD_AWS_BUCKET_NAME"]
MAX_TRANSCRIPTION_ATTEMPTS = 4

def s3_client():
    return boto3.client(
        service_name='s3',
        region_name='eu-west-1',
        aws_access_key_id=AWS_ACCESS_KEY_ID.strip(),
        aws_secret_access_key=AWS_SECRET_ACCESS_KEY.strip()
    )

def upload_file_to_bucket(file_path, bucket, file_name):
    s3 = s3_client()
    try:
        s3.upload_file(file_path, bucket, file_name)
        st.success("File Successfully Uploaded")
        return True
    except FileNotFoundError:
        st.error("File not found")
        return False

def extract_audio_from_video(video_file):
    video = VideoFileClip(video_file)
    audio_file = "audio.wav"
    audio = video.audio
    audio.write_audiofile(audio_file)
    return audio_file

def transcribe_audio_with_speechbrain(audio_file):
    # Load pre-trained ASR model
    asr_model = sb.models.CRDNN.from_hparams(
        source="speechbrain/asr-crdnn-rnnlm-librispeech",
        savedir="tmpdir",
    )
    # Load audio file
    signal, fs = torchaudio.load(audio_file)
    # Transcribe audio
    transcription = asr_model.encode_text(signal)
    return transcription

def main():
    st.set_page_config(page_title="Transcription AI Demo")
    st.header("Transcription AI Demo")

    vid = st.file_uploader("Upload your video", type="mp4")
    if vid is not None:
        transcription_attempts = 1
        video_path = os.path.join("/tmp", vid.name)
        with open(video_path, "wb") as f:
            f.write(vid.read())

        upload_file_to_bucket(video_path, UPLOAD_AWS_BUCKET_NAME, vid.name)
        audio_file = extract_audio_from_video(video_path)
        
        while transcription_attempts < MAX_TRANSCRIPTION_ATTEMPTS:
            with st.spinner("Transcribing..."):
                time.sleep(5)
            try:
                transcription = transcribe_audio_with_speechbrain(audio_file)
                st.success("Transcription Successful")
                st.header("Transcription")
                st.markdown(transcription)
                break
            except Exception as e:
                st.error(f"An error occurred: {e}")
                transcription_attempts += 1

if __name__ == "__main__":
    main()
