import streamlit as st
import torch
import torchaudio
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import librosa
import soundfile as sf
from io import BytesIO
import os
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt
from tqdm import tqdm
import glob
import re
import sounddevice as sd
import wavio


# Define your DEMUCS model class
class GLU(nn.Module):
  def forward(self,x):
    x,gate=x.chunk(2,dim=1)
    return x*torch.sigmoid(gate)

class Encoder(nn.Module):
  def __init__(self,L=5,H=48,K=8,S=4):
    super().__init__()
    layers=[]
    in_channels=1
    for i in range(L):
      out_channels=H * (2**i)
      layers.append(nn.Conv1d(in_channels,out_channels,kernel_size=K,stride=S,padding=K//2))
      layers.append(nn.ReLU())
      layers.append(nn.Conv1d(out_channels, 2 * out_channels, kernel_size=1,stride=1))
      layers.append(GLU())
      in_channels=out_channels

    self.encoder=nn.Sequential(*layers)

  def forward(self, x):
    skips = []
    num_layers = len(self.encoder) // 4  # Since each layer consists of (Conv -> ReLU -> Conv -> GLU)

    for i in range(num_layers):
        conv1 = self.encoder[i * 4](x)  # First Conv1D layer
        relu = self.encoder[i * 4 + 1](conv1)  # ReLU activation
        conv2 = self.encoder[i * 4 + 2](relu)  # Second Conv1D layer
        x = self.encoder[i * 4 + 3](conv2)  # GLU activation

        skips.append(x)
        # print(f"Encoder Block {i + 1}: Output Shape {x.shape}")  # Debugging print

    return x, skips

class Sequence_Modeling(nn.Module):
  def __init__(self,L,H):
    super().__init__()
    hidden_size = H * 2**(L-1)
    self.lstm=nn.LSTM(input_size=hidden_size,hidden_size=hidden_size,num_layers=2,batch_first=True,bidirectional=False) #batch_first= true for lstm to take this form (batch,time,channels)

  def forward(self,x):
    x=x.permute(0,2,1)  #. (batch_size,channels,time) ---> (batch,time,channels)
    x,_=self.lstm(x)
    x=x.permute(0,2,1)
    return x

class Decoder(nn.Module):
    def __init__(self, L=5, H=48, K=8, S=4):
        super().__init__()
        self.levels = nn.ModuleList()
        in_channels = H * (2 ** (L - 1))

        # Build decoder levels from deepest to shallowest (i=L-1 down to 0)
        for i in range(L - 1, -1, -1):
            out_channels = H * (2 ** (i - 1)) if i > 0 else 1
            layers = []

            layers.append(nn.Conv1d(in_channels, 2 * in_channels, kernel_size=1))
            layers.append(GLU())  # Halves channels back to in_channels

            # Transposed Convolution
            output_padding = (
                S - 1 if (i == L-1 or i == L-2) else
                1 if (i == L-3 or i == L-4) else
                0
            )
            layers.append(nn.ConvTranspose1d(
                in_channels, out_channels,
                kernel_size=K, stride=S,
                padding=K // 2, output_padding=output_padding
            ))

            # Add ReLU for all except the last layer
            if i > 0:
                layers.append(nn.ReLU())

            self.levels.append(nn.Sequential(*layers))
            in_channels = out_channels  # Update for next level

    def forward(self, x, skips):
        # Process levels from deepest to shallowest (same order as self.levels)
        for level in self.levels:
            skip = skips.pop()  # Retrieve skip in reverse encoder order
            x = x + skip
            x = level(x)
            # print(f"Decoder Level Output Shape: {x.shape}")
        return x


class DEMUCS(nn.Module):

    def __init__(self, L=5, H=48, K=8, S=4, U=4):
        super().__init__()
        # self.upsample = nn.Upsample(scale_factor=U, mode="linear", align_corners=False)
        # self.downsample = nn.AvgPool1d(kernel_size=U, stride=U)
        self.upsample = torchaudio.transforms.Resample(orig_freq=16000, new_freq=16000 * U)
        self.downsample = torchaudio.transforms.Resample(orig_freq=16000 * U, new_freq=16000)

        self.encoder = Encoder(L, H, K, S)
        self.sequence_modeling = Sequence_Modeling(L, H)
        self.decoder = Decoder(L, H, K, S)

    def forward(self, x):
        # x = self.upsample(x)
        x, skips = self.encoder(x)
        x = self.sequence_modeling(x)
        x = self.decoder(x, skips)
        # x = self.downsample(x)
        return x

# Load trained model
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Model Hyperparameters (same as training)
    # Set parameters for the model
    L = 5   # Number of encoder layers
    H = 48  # Base number of channels
    K = 8   # Kernel size
    S = 4   # Stride
    U = 4   # Upsampling factor

    model = DEMUCS(L=L, H=H, K=K, S=S, U=U).to(device)
    checkpoint_path = "/Users/sanimpandey/Desktop/Minor/denoise/epoch_75.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    return model, device

model, device = load_model()

# Function to process audio
def enhance_audio(model, noisy_audio):
    segment_length = 16000  # Fixed segment size used during training

    # Convert to tensor & add necessary dimensions
    noisy_audio = torch.tensor(noisy_audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(device)

    # Ensure length is a multiple of 16,000
    current_length = noisy_audio.shape[-1]
    pad_needed = (segment_length - (current_length % segment_length)) % segment_length
    noisy_audio = F.pad(noisy_audio, (0, pad_needed), mode="constant", value=0)

    # Process chunks of 16,000 samples
    chunks = noisy_audio.unfold(dimension=-1, size=segment_length, step=segment_length).transpose(1, 2)
    chunks = chunks.squeeze(2)  # Shape: [batch, 1, segment_length]

    with torch.no_grad():
        enhanced_chunks = torch.cat([model(chunk.unsqueeze(1)) for chunk in chunks], dim=0)

    # Reshape the output to match the expected 1D format
    enhanced_audio = enhanced_chunks.view(-1).cpu().numpy()

    return enhanced_audio


# # Streamlit UI
st.title("🔉 Audio Denoising with DEMUCS")

st.write("Upload a noisy audio file, and the model will enhance it.")

uploaded_file = st.file_uploader("Upload a WAV file", type=["wav"])

if uploaded_file is not None:
    # Load audio
    noisy_audio, sr = librosa.load(uploaded_file, sr=16000)

    st.audio(uploaded_file, format="audio/wav", start_time=0)

    # Process audio
    st.write("Processing audio...")
    enhanced_audio = enhance_audio(model, noisy_audio)

    # Save enhanced audio to a buffer
    output_buffer = BytesIO()
    sf.write(output_buffer, enhanced_audio, samplerate=16000, format="WAV")  # 🔹 Now it's correctly shaped
    output_buffer.seek(0)

    # Display audio player & download button
    st.audio(output_buffer, format="audio/wav", start_time=0)
    st.download_button(label="Download Enhanced Audio",
                    data=output_buffer,
                    file_name="enhanced_audio.wav",
                    mime="audio/wav")


    st.success("✅ Enhancement complete! Listen or download the enhanced audio.")

# st.title("🔉 Real-time Audio Denoising with DEMUCS")

# duration = st.slider("Recording Duration (seconds)", 1, 10, 3)
# sample_rate = 16000

# if st.button("🎤 Start Recording"):
#     st.write("🎙️ Recording...")
#     recorded_audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
#     sd.wait()
#     st.success("✅ Recording Complete!")
    
#     audio_bytes = BytesIO()
#     wavio.write(audio_bytes, recorded_audio, sample_rate, sampwidth=2)
#     audio_bytes.seek(0)
#     st.audio(audio_bytes, format="audio/wav")
    
#     # noisy_audio, sr = librosa.load(audio_bytes, sr=16000)
#     # enhanced_audio = enhance_audio(model, noisy_audio)
    
#     # output_buffer = BytesIO()
#     # sf.write(output_buffer, enhanced_audio, samplerate=16000, format="WAV")
#     # output_buffer.seek(0)
    
#     # st.audio(output_buffer, format="audio/wav")
#     # st.download_button("Download Enhanced Audio", output_buffer, "enhanced_audio.wav", "audio/wav")



#     # Load and process the noisy audio
#     noisy_audio, sr = librosa.load(audio_bytes, sr=16000)
#     enhanced_audio = enhance_audio(model, noisy_audio)

#     # Save enhanced audio to buffer
#     output_buffer = BytesIO()
#     sf.write(output_buffer, enhanced_audio, samplerate=16000, format="WAV")
#     output_buffer.seek(0)

#     # Display both noisy and enhanced audio
#     st.write("🔊 **Noisy Audio (Original Input)**")
#     st.audio(audio_bytes, format="audio/wav")

#     st.write("🎵 **Enhanced Audio (Denoised Output)**")
#     st.audio(output_buffer, format="audio/wav")

#     # Provide download option for enhanced audio
#     st.download_button("Download Enhanced Audio", output_buffer, "enhanced_audio.wav", "audio/wav")




