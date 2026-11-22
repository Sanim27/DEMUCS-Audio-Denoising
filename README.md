# DEMUCS-Based Audio Denoising

A deep learning-based speech enhancement system using a customized **DEMUCS (Deep Extractor for Music Sources)** architecture. The model operates directly on raw audio waveforms and uses convolutional encoder-decoder layers, LSTM-based temporal modeling, and skip connections to suppress background noise while preserving speech characteristics.

This repository contains the trained model weights, model architecture required for inference, and a Streamlit-based frontend for demonstrating audio denoising.

## Project Background

This work is part of a broader academic project titled **"Traditional Filters and AI for Enhanced Signal De-noising,"** conducted at the Department of Electronics and Computer Engineering, Tribhuvan University, Institute of Engineering, Pulchowk Campus.

The broader project investigated and compared conventional digital signal processing (DSP) techniques with deep learning-based approaches for audio denoising.

The traditional approaches explored in the project included:

- Spectral Subtraction
- FFT-based Denoising with Adaptive Thresholding
- Spectral Denoising with Frequency-Domain Filtering and Peak Evaluation
- Wiener Filtering

For the deep learning component, we implemented and trained a DEMUCS-based architecture to perform speech enhancement directly in the time domain.

**Scope of this repository:** This repository focuses exclusively on the deep learning component of the broader project. It provides the model implementation, trained weights, and an interactive frontend for running inference. It does not contain the complete project report, traditional DSP implementations, original training dataset, or the full model training pipeline.

## Model Architecture

DEMUCS is an encoder-decoder architecture designed to process audio directly in the time domain, eliminating the need for an intermediate spectrogram representation during inference.

Our implementation follows an encoder–sequence modeling–decoder structure with U-Net-style skip connections.

### 1. Encoder

The encoder extracts hierarchical representations from the input audio waveform using multiple one-dimensional convolutional layers.

The implementation consists of:

- Five convolutional encoder layers.
- ReLU activations for introducing non-linearity.
- Gated Linear Units (GLUs) for controlling information flow.
- Progressive downsampling to extract increasingly abstract audio features.

The number of channels increases progressively across the encoder layers, allowing the network to capture increasingly complex characteristics of the speech signal.

Given an input waveform \(x\), the encoder produces a latent representation:

\[
z = E(x)
\]

### 2. Sequence Modeling

The encoded representation is passed through a two-layer Long Short-Term Memory (LSTM) network.

Since speech is inherently sequential, temporal modeling is important for identifying relationships between speech components across time. The LSTM captures these temporal dependencies, helping the network distinguish speech characteristics from background noise.

The sequence modeling component is defined as:

\[
\hat{z} = \operatorname{LSTM}(z)
\]

### 3. Decoder

The decoder reconstructs the enhanced audio waveform from the encoded representation.

It consists of transposed convolutional layers that progressively upsample the latent features.

Skip connections transfer intermediate feature representations from the encoder to corresponding decoder layers. This helps preserve important speech details that might otherwise be lost during downsampling.

The final decoder layer produces the enhanced waveform:

\[
\hat{y} = D(\hat{z})
\]

### Overall Architecture

The processing pipeline can be summarized as:

```text
       Noisy Audio Waveform
                 |
                 v
       +-------------------+
       |      Encoder      |
       |  1D Convolutions  |
       |    ReLU + GLU     |
       +-------------------+
                 |
                 v
       +-------------------+
       |   LSTM Network    |
       |   2 LSTM Layers   |
       +-------------------+
                 |
                 v
       +-------------------+
       |      Decoder      |
       | Transposed Conv.  |
       |    GLU + ReLU     |
       +-------------------+
                 |
                 v
        Enhanced Audio
        (Speech Output)
```

Encoder features are also passed to corresponding decoder layers through skip connections.

### Model Configuration

| Parameter | Configuration |
|---|---|
| Architecture | DEMUCS-based |
| Input representation | Raw audio waveform |
| Encoder depth | 5 layers |
| Base channels (H) | 48 |
| Kernel size (K) | 8 |
| Stride (S) | 4 |
| Sequence modeling | LSTM |
| Number of LSTM layers | 2 |
| LSTM direction | Unidirectional |
| Decoder | Transposed 1D convolutions |
| Activation functions | ReLU and GLU |
| Input sampling rate | 16 kHz |
| Processing segment | 1 second |
| Model checkpoint | `epoch_75.pth` |

## Dataset and Preprocessing

The model was trained using the **Valentini-Botinhao dataset**, which contains paired clean and noisy speech recordings from multiple speakers.

The dataset includes noisy speech samples generated using different background noise sources, including environmental sounds, music, and babble noise.

The preprocessing steps described in the project report include:

1. **Resampling:** Audio recordings were converted to a common sampling rate of 16 kHz.
2. **Normalization:** Waveform amplitudes were normalized to maintain consistent input levels.
3. **Segmentation:** Audio was divided into fixed-duration training samples.

The model was trained using one-second audio samples.

## Training Methodology

The model was trained using supervised learning with paired noisy and clean speech signals.

The training objective combined waveform-level and frequency-domain losses to encourage both accurate waveform reconstruction and preservation of spectral characteristics.

### Loss Functions

**1. L1 Waveform Loss**

The L1 loss measures the absolute difference between the predicted enhanced waveform and its corresponding clean reference.

\[
L_{\text{waveform}} =
\frac{1}{T}\sum_{t=1}^{T}
|y_t-\hat{y}_t|
\]

This encourages the model to reconstruct the clean speech waveform accurately.

**2. Multi-Resolution STFT Loss**

The Multi-Resolution Short-Time Fourier Transform (MR-STFT) loss compares the predicted and reference signals across multiple spectral resolutions.

It consists of spectral convergence and log-magnitude components:

\[
L_{\text{STFT}}=L_{\text{sc}}+L_{\text{mag}}
\]

This encourages the model to preserve both fine-grained and broader spectral characteristics.

### Training Configuration

| Parameter | Value |
|---|---|
| Dataset | Valentini-Botinhao |
| Training approach | Supervised learning |
| Epochs | 75 |
| Batch size | 64 |
| Optimizer | Adam |
| Initial learning rate | \(10^{-4}\) |
| Weight decay | Applied |
| Gradient clipping | Applied |
| Loss functions | L1 Waveform + MR-STFT |
| Training environment | Google Colab CPU |
| Hardware | Intel Xeon, 2 vCPUs, 13 GB RAM |

The final trained checkpoint included in this repository is `epoch_75.pth`.

**Note:** The original training scripts and dataset are not included in this repository. The training configuration above is documented from the accompanying academic project report.

## Inference and Frontend

The repository includes a Streamlit-based frontend that allows users to interact with the trained model without manually running the inference pipeline.

### Application Workflow

1. Upload a noisy WAV audio file.
2. The application loads the audio and resamples it to 16 kHz.
3. The waveform is divided into one-second segments.
4. Each segment is passed through the trained DEMUCS model.
5. The enhanced segments are combined into the output waveform.
6. The application provides audio playback and an option to download the enhanced WAV file.

The model is loaded in evaluation mode, and inference is performed without gradient computation.

### Repository Structure

```text
DEMUCS-Audio-Denoising/
│
├── denoise.py       # DEMUCS architecture, inference pipeline,
│                    # and Streamlit frontend
│
├── epoch_75.pth     # Trained model checkpoint (75 epochs)
│
├── requirements.txt # Python dependencies
│
└── README.md        # Project documentation
```

## Installation and Usage

### Prerequisites

- Python 3.9 or compatible Python environment
- pip
- A CPU or CUDA-compatible GPU

### 1. Clone the Repository

```bash
git clone https://github.com/Sanim27/DEMUCS-Audio-Denoising.git

cd DEMUCS-Audio-Denoising
```

### 2. Create a Virtual Environment

```bash
python -m venv venv

# Linux / macOS
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure the Model Checkpoint Path

Before running the application, ensure that the checkpoint path in `denoise.py` points to the downloaded `epoch_75.pth` file.

For portability, the checkpoint can be referenced relative to the script:

```python
from pathlib import Path

checkpoint_path = Path(__file__).resolve().parent / "epoch_75.pth"
```

### 5. Run the Application

```bash
streamlit run denoise.py
```

Open the local URL provided by Streamlit in your browser.

Upload a noisy WAV file to generate and listen to its enhanced version.

## Evaluation and Observations

The broader project evaluated the deep learning model using objective speech enhancement metrics:

- **PESQ (Perceptual Evaluation of Speech Quality):** Evaluates perceived speech quality.
- **STOI (Short-Time Objective Intelligibility):** Measures speech intelligibility.
- **SNR (Signal-to-Noise Ratio):** Measures the ratio between desired speech and background noise.

The project report documents comparisons between DEMUCS-based denoising and traditional signal processing techniques.

The experiments indicated that:

- DEMUCS performed particularly well in highly noisy environments, suppressing background noise while retaining important speech characteristics.
- Compared with the traditional methods investigated, the model demonstrated better adaptability to varying and unpredictable noise conditions.
- In low-noise scenarios, the model occasionally introduced artifacts, including subtle reverberation and waveform distortions.
- Processing consecutive one-second segments without overlap was identified as a possible contributor to discontinuities and artifacts.

These observations highlight the trade-off between aggressive noise suppression and preserving the natural characteristics of speech.

## Limitations and Future Improvements

The current implementation provides a demonstration of the trained model rather than a fully optimized real-time speech enhancement system.

Potential areas for further improvement include:

- Introducing overlapping audio segmentation and smoother reconstruction between consecutive segments.
- Improving performance in low-noise environments to reduce unnecessary speech suppression.
- Exploring additional training data and noise conditions to improve generalization.
- Optimizing inference for lower latency and real-time processing.
- Evaluating deployment on resource-constrained devices.

**Important:** Although the broader project investigated real-time noise reduction as an application, the DEMUCS model was trained and evaluated in a controlled environment. Real-time deployment feasibility and inference latency were not formally benchmarked in the reported experiments.

## Academic Project Information

**Project Title:** Traditional Filters and AI for Enhanced Signal De-noising

**Institution:** Tribhuvan University, Institute of Engineering, Pulchowk Campus

**Department:** Electronics and Computer Engineering
