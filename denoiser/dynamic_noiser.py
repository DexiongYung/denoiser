import math
import torch
import torchaudio
import torchaudio.transforms as T

def adjust_volume(noise_audio, adjustment_db):
    """Adjust the volume of the noise audio."""
    return noise_audio * torch.tensor([10.0**(adjustment_db / 20.0)])

def apply_bandpass_filter(noise_audio, sample_rate, low_freq, high_freq):
    """Apply a band-pass filter to the noise audio."""
    bp_filter = T.BandpassBiquad(sample_rate, low_freq, high_freq)
    return bp_filter(noise_audio)

def apply_speed_and_pitch_change(noise_audio, sample_rate, speed_factor):
    """Change both the speed and pitch of the noise audio."""
    return torchaudio.functional.resample(noise_audio, sample_rate, int(sample_rate / speed_factor))

def time_stretch(noise_audio, sample_rate, stretch_factor):
    """Apply time stretching to the noise audio without changing its pitch."""
    # Note: Time stretching can be complex; here, we're using a simple resampling approach for illustration.
    stretched = torchaudio.functional.resample(noise_audio, sample_rate, int(sample_rate * stretch_factor))
    return torchaudio.functional.resample(stretched, int(sample_rate * stretch_factor), sample_rate)

def apply_echo(noise_audio, sample_rate, delay_ms, decay):
    """Apply an echo effect to the noise audio."""
    echo = T.Echo(sample_rate, delay_ms, decay)
    return echo(noise_audio)

def apply_environmental_sounds(noise_audio, environmental_audio):
    """Mix in environmental sounds with the noise audio."""
    # Adjusting the length of environmental audio to match the noise audio
    if environmental_audio.size(1) > noise_audio.size(1):
        environmental_audio = environmental_audio[:, :noise_audio.size(1)]
    else:
        repeat_times = (noise_audio.size(1) // environmental_audio.size(1)) + 1
        environmental_audio = environmental_audio.repeat(1, repeat_times)[:,:noise_audio.size(1)]
    # Mixing
    mixed_audio = noise_audio + environmental_audio
    return mixed_audio

def mix_clean_and_noise(clean_audio, noise_audio, snr_db):
    """Mix clean audio with noise audio based on a specified SNR."""
    clean_rms = clean_audio.pow(2).mean().sqrt()
    noise_rms = noise_audio.pow(2).mean().sqrt()

    # Calculate the desired noise RMS based on the SNR
    desired_noise_rms = clean_rms / (10**(snr_db / 20))

    # Adjust the noise to achieve the desired SNR
    noise_audio = noise_audio * (desired_noise_rms / noise_rms)

    # Mix the clean and adjusted noise audio
    return clean_audio + noise_audio

def simulate_noise_at_distance(audio_signal, noise_signal, distance_meters, initial_distance=1):
    """
    Simulate the effect of a noise source at different distances.
    
    :param audio_signal: Tensor representing the original audio signal.
    :param noise_signal: Tensor representing the noise to be added.
    :param distance_meters: The distance from the source at which to simulate the noise.
    :param initial_distance: The initial distance of the noise source (usually 1 meter).
    :param sampling_rate: The sampling rate of the audio and noise signals.
    :return: The audio signal with the simulated distant noise added.
    """
    # Attenuation factor: sound intensity drops by 6 dB for every doubling of distance.
    attenuation_factor = -6 * math.log2(distance_meters / initial_distance)
    
    # Adjust the amplitude of the noise based on the attenuation factor.
    # Convert dB to amplitude ratio: 20*log10(amplitude_ratio) = attenuation_factor
    amplitude_ratio = 10 ** (attenuation_factor / 20)
    attenuated_noise = noise_signal * amplitude_ratio
    
    # Ensure the noise is the same length as the audio signal
    if attenuated_noise.shape[1] > audio_signal.shape[1]:
        attenuated_noise = attenuated_noise[:, :audio_signal.shape[1]]
    else:
        # Repeat the noise signal if it's shorter than the audio signal
        repeat_factor = math.ceil(audio_signal.shape[1] / attenuated_noise.shape[1])
        attenuated_noise = attenuated_noise.repeat(1, repeat_factor)[:, :audio_signal.shape[1]]
    
    # Mix the original audio signal with the attenuated noise
    mixed_signal = audio_signal + attenuated_noise
    
    return mixed_signal