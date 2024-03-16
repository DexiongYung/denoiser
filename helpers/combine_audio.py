import math
import torch
import torchaudio
import random
import os

def load_and_process(file_path_a, file_path_b, scale_max_value):
    # Load WAV files
    waveform_a, sample_rate_a = torchaudio.load(file_path_a)
    waveform_b, sample_rate_b = torchaudio.load(file_path_b)

    if sample_rate_b != sample_rate_a:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate_b, new_freq=sample_rate_a)
        waveform_b = resampler(waveform_b)

    # Scale B's max value
    max_val_b = waveform_b.abs().max()
    scaling_factor = scale_max_value / max_val_b
    waveform_b_scaled = waveform_b * scaling_factor

    # Match the lengths
    len_a = waveform_a.shape[1]
    len_b_scaled = waveform_b_scaled.shape[1]

    if len_b_scaled > len_a:
        # If B is longer, clip randomly
        start = random.randint(0, len_b_scaled - len_a)
        waveform_b_processed = waveform_b_scaled[:, start:start + len_a]
    else:
        # If B is shorter, duplicate random parts
        waveform_b_processed = waveform_b_scaled.repeat(1, math.ceil(len_a / waveform_b_scaled.shape[1]))[:,:len_a]

    # Combine A and B
    combined_waveform = waveform_a + waveform_b_processed

    return combined_waveform, sample_rate_a


def random_file_from_random_dir(dir_list):
    # Randomly select a directory from the list
    selected_dir = random.choice(dir_list)
    
    # List all files in the selected directory
    files = [f for f in os.listdir(selected_dir) if os.path.isfile(os.path.join(selected_dir, f))]
    
    # If the directory is empty or has no files, return None
    if not files:
        return None
    
    # Randomly select a file from the list of files
    selected_file = random.choice(files)
    
    # Return the path to the selected file
    return os.path.join(selected_dir, selected_file)


def get_all_files_from_folder(folder_path):
    all_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


# Example usage
dir_list = ['/mnt/dylan_disk/YT_audio/lectures']


clip_paths = get_all_files_from_folder("/mnt/dylan_disk/clips")

for cp in clip_paths:
    noise_clip = random_file_from_random_dir(dir_list=dir_list)
    combined_waveform, sample_rate = load_and_process(cp, noise_clip, random.uniform(0.05, 0.5))
    torchaudio.save(f"/mnt/dylan_disk/noise_clips_lecture/{os.path.basename(cp)}", combined_waveform, sample_rate)
