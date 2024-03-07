import torch
import torchaudio

def load_audio_files(file_paths, target_sample_rate=None):
    waveforms = []
    for file_path in file_paths:
        waveform, sample_rate = torchaudio.load(file_path)
        if target_sample_rate is not None and sample_rate != target_sample_rate:
            waveform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=target_sample_rate)(waveform)
        waveforms.append(waveform)
    return waveforms

def adjust_waveforms_to_common_length(waveforms, target_length):
    adjusted_waveforms = []
    original_lengths = []  # Store the original lengths before padding
    for waveform in waveforms:
        original_lengths.append(waveform.size(1))
        if waveform.size(1) > target_length:
            # Truncate
            start = torch.randint(high=waveform.size(1) - target_length + 1, size=(1,)).item()
            adjusted_waveform = waveform[:, start:start + target_length]
        else:
            # Pad
            pad_amount = target_length - waveform.size(1)
            adjusted_waveform = torch.nn.functional.pad(waveform, (0, pad_amount), "constant", 0)
        adjusted_waveforms.append(adjusted_waveform)
    return adjusted_waveforms, original_lengths

def batch_process_waveforms(waveforms):
    # Example processing: scale the max amplitude to 0.8 for all waveforms
    scale_max_value = 0.8
    processed_waveforms = []
    for waveform in waveforms:
        max_val = waveform.abs().max()
        if max_val > 0:
            scaling_factor = scale_max_value / max_val
            waveform = waveform * scaling_factor
        processed_waveforms.append(waveform)
    return processed_waveforms

# Load and process files
file_paths = ['path_to_file_1.wav', 'path_to_file_2.wav']  # Add your file paths here
target_sample_rate = 16000  # Define a target sample rate
waveforms = load_audio_files(file_paths, target_sample_rate)

# Determine a common length (e.g., the length of the longest file)
target_length = max([waveform.size(1) for waveform in waveforms])

# Adjust all waveforms to this common length and keep track of original lengths
adjusted_waveforms, original_lengths = adjust_waveforms_to_common_length(waveforms, target_length)

# Convert list of waveforms to a batch tensor
waveform_batch = torch.stack(adjusted_waveforms)

# Process the batch tensor
processed_waveform_batch = batch_process_waveforms(waveform_batch)

# Remove padding (if any) before saving
for i, (waveform, original_length) in enumerate(zip(processed_waveform_batch, original_lengths)):
    # Ensure waveform is 2D: [channels, time]
    waveform = waveform[:, :original_length]  # Trim to the original length
    torchaudio.save(f'processed_file_{i}.wav', waveform, target_sample_rate)
