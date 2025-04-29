# Audio Denoising with DEMUCS

This project uses the DEMUCS model for audio denoising. Upload a noisy .wav file, and the model will process it to return an enhanced version.
Setup

Install dependencies:

    pip install -r requirements.txt

Run Streamlit app:

    streamlit run denoise.py

How it Works

    Upload a noisy audio file.

    The DEMUCS model denoises the audio.

    Play or download the enhanced audio.
