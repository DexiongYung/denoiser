import math
import torch
import torchaudio
import numpy as np


def apply_speed_and_pitch_change(noise_audio, sample_rate, speed_factor):
    """Change both the speed and pitch of the noise audio."""
    return torchaudio.functional.resample(noise_audio, sample_rate, int(sample_rate / speed_factor))


def time_stretch(noise_audio, sample_rate, stretch_factor):
    """Apply time stretching to the noise audio without changing its pitch."""
    # Note: Time stretching can be complex; here, we're using a simple resampling approach for illustration.
    stretched = torchaudio.functional.resample(noise_audio, sample_rate, int(sample_rate * stretch_factor))
    return torchaudio.functional.resample(stretched, int(sample_rate * stretch_factor), sample_rate)


def apply_echo(noise_audio, sample_rate, delay_ms, decay, repeats=1):
    """
    Apply an echo effect to an audio clip within the original audio length.

    Parameters:
    - noise_audio: A torch.Tensor representing the audio signal.
    - sample_rate: An integer representing the sample rate of the audio.
    - delay_ms: The delay of the echo in milliseconds.
    - decay: The decay rate of the echo. Each echo will be this fraction of the volume of the previous one.
    - repeats: The number of echo repetitions.

    Returns:
    - A torch.Tensor representing the audio signal with the echo effect applied within the original length.
    """
    delay_samples = int(sample_rate * delay_ms / 1000.0)
    extended_audio = noise_audio.clone()  # Work with a copy of the original audio

    # Apply the echoes within the original length
    for i in range(1, repeats + 1):
        start = delay_samples * i
        end = extended_audio.size(-1)
        # Make sure not to exceed the original audio length
        if start < end:
            extended_audio[..., start:end] += noise_audio[..., :end-start] * (decay ** i)
    
    return extended_audio


def dynamic_noise_distance_adjustment(sample_rate, noise_audio, min_distance, max_distance, segments):
    """
    Adjusts the background noise distance dynamically over the audio sequence.

    Parameters:
    - sample_rate (int): The sample rate of the audio signals.
    - noise_audio (torch.Tensor): The noise signal to be adjusted.
    - min_distance (int): The minimum distance in meters.
    - max_distance (int): The maximum distance in meters.
    - segments (int): Number of segments to divide the audio into for distance adjustments.

    Returns:
    - torch.Tensor: The audio signal with dynamically adjusted noise, matching the original sequence length.
    """
    # Initialize the adjusted noise signal with the same shape as the original
    adjusted_noise_signal = torch.zeros_like(noise_audio)
    
    # Total length of the noise_audio tensor
    total_length = noise_audio.shape[-1]  # Assuming shape is either (samples,) or (channels, samples)
    
    # Calculate the exact segment length, accounting for any remainder
    segment_length = total_length // segments
    remainder = total_length % segments
    
    current_distance = np.random.uniform(min_distance, max_distance)
    
    for segment in range(segments):
        # Adjust the segment_end for the last segment or normally increment it
        segment_start = segment * segment_length
        segment_end = segment_start + segment_length
        if segment == segments - 1:  # Add remainder to the last segment
            segment_end += remainder

        # Calculate distances for linear interpolation
        next_distance = np.random.uniform(min_distance, max_distance)
        distances = np.linspace(current_distance, next_distance, segment_end - segment_start)
        
        # Calculate attenuations based on distances
        attenuations = torch.tensor([10 ** (-6 * math.log2(distance) / 20) for distance in distances], dtype=torch.float32)
        
        # Apply attenuations
        if noise_audio.dim() == 2:
            # For 2D tensors, apply attenuation per channel
            adjusted_noise_signal[:, segment_start:segment_end] = noise_audio[:, segment_start:segment_end] * attenuations
        else:
            # For 1D tensors
            adjusted_noise_signal[segment_start:segment_end] = noise_audio[segment_start:segment_end] * attenuations
        
        current_distance = next_distance

    return adjusted_noise_signal