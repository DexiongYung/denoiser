import os
import wave
import json
import random
import math
import torchaudio

# Function to get the duration of a WAV file in milliseconds
def get_wav_duration_ms(wav_path):
    tnsr, sr = torchaudio.load(wav_path)
    return math.ceil(tnsr.shape[-1] / sr) * 1000

# Function to generate a list of lists with WAV file paths and their lengths
def generate_wav_info_json(folder_path, noise_path, max_num:int=None):
    wav_info_list = list()
    noise_info_list = list()
    for _, filename in enumerate(os.listdir(folder_path)):
        if len(wav_info_list) == max_num:
            break 

        if filename.endswith(".wav"):
            full_path = os.path.join(folder_path, filename)
            length_ms = get_wav_duration_ms(full_path)
            
            if length_ms < 4000:
                continue

            wav_info_list.append([full_path, length_ms])
            if noise_path:
                noise_info_list.append([os.path.join(noise_path, filename), length_ms])
    return wav_info_list, noise_info_list

# Example usage
folder_path = '/mnt/dylan_disk/clips'
noise_path = "/mnt/dylan_disk/noise_clips"
wav_info_list, noise_info_list = generate_wav_info_json(noise_path, folder_path, max_num=None)

indexes = list(range(len(wav_info_list)))
random.shuffle(indexes)

mid_index = len(indexes) // 2
first_half_indexes = indexes[:mid_index]
second_half_indexes = indexes[mid_index:]

for name, idx_list in (("tr", first_half_indexes), ("ts", second_half_indexes)):
    clean = [wav_info_list[i] for i in idx_list]
    noise = [noise_info_list[i] for i in idx_list]

    with open(f"/home/vapi/denoiser/egs/large/{name}/noisy.json", "w") as file:
        json.dump(clean, file, indent=4)

    with open(f"/home/vapi/denoiser/egs/large/{name}/clean.json", "w") as file:
        json.dump(noise, file, indent=4)