# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import torch
from collections import namedtuple
import json
from pathlib import Path
import math
import glob
import random
import os
import sys
from omegaconf import ListConfig
import torchaudio
from torch.nn import functional as F

from torchaudio.transforms import MelSpectrogram
from denoiser.dynamic_noiser import apply_echo, apply_speed_and_pitch_change, dynamic_noise_distance_adjustment
from .dsp import convert_audio

Info = namedtuple("Info", ["length", "sample_rate", "channels"])


def get_info(path):
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames'):
        # new version of torchaudio
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels)


def find_audio_files(path, exts=[".wav"], progress=True):
    audio_files = []
    for root, _, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))
    meta = []
    for idx, file in enumerate(audio_files):
        info = get_info(file)
        meta.append((file, info.length))
        if progress:
            print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    meta.sort()
    return meta

class DynamicNoiser:
    def __init__(self, noise_folder_path: str, funcs_w_args: dict) -> None:
        self.aug_key_to_func = dict(
            apply_echo=apply_echo, 
            dynamic_noise_distance_adjustment=dynamic_noise_distance_adjustment)
        
        print(f"The functions selected for augmentation with their parameters: \n{funcs_w_args}")
        print(f"Noise folder path: {noise_folder_path}")
        
        for key in funcs_w_args.keys():
            assert key in self.aug_key_to_func, f"Given: {key}, but this is not a known augmentation function. Known ones are: {self.aug_key_to_func.keys()}"

        self.funcs_w_args = funcs_w_args
        self.wav_files = self.get_all_wavs_from_folder_path(fp=noise_folder_path)
    
    def get_all_wavs_from_folder_path(self, fp):
        wav_files = []
    
        # Walk through the directory and its subdirectories
        for root, _, _ in os.walk(fp):
            for file in glob.glob(os.path.join(root, '*.wav')):
                wav_files.append(file)
        
        return wav_files
    
    def adjust_audio_length(self, audio_tensor, target_length):
        """
        Adjusts the length of an audio tensor to a specific target length by clipping or duplicating it.
        If the target length is shorter, it clips a random segment from the audio_tensor.
        Handles both 1D and 2D tensors, assuming 2D tensors are shaped as (channels, samples).

        Parameters:
        - audio_tensor: A PyTorch tensor representing the audio signal, can be 1D or 2D.
        - target_length: The desired length of the audio tensor.

        Returns:
        - A PyTorch tensor representing the adjusted audio signal with the specified target length.
        """
        current_length = audio_tensor.shape[-1]  # Get the current length from the last dimension

        if current_length == target_length:
            # The audio is already the correct length
            return audio_tensor
        elif current_length > target_length:
            # Clip a random segment from the audio to the target length
            max_start_index = current_length - target_length
            start_index = torch.randint(low=0, high=max_start_index + 1, size=(1,)).item()
            if audio_tensor.dim() == 1:
                return audio_tensor[start_index:start_index + target_length]
            else:  # For 2D tensors
                return audio_tensor[:, start_index:start_index + target_length]
        else:
            # Duplicate the audio to reach the target length
            repeat_times = target_length // current_length
            extra_samples = target_length % current_length

            duplicated_audio = audio_tensor.repeat(1, repeat_times) if audio_tensor.dim() == 2 else audio_tensor.repeat(repeat_times)
            if extra_samples > 0:
                if audio_tensor.dim() == 2:
                    duplicated_audio = torch.cat((duplicated_audio, audio_tensor[:, :extra_samples]), dim=1)
                else:
                    duplicated_audio = torch.cat((duplicated_audio, audio_tensor[:extra_samples]))
            
            return duplicated_audio

    
    def forward(self, waveform, sample_rate):
        noise_fp = random.choice(self.wav_files)
        noise_waveform, noise_sample_rate = torchaudio.load(noise_fp)
    
        # Check if resampling is necessary
        if sample_rate != noise_sample_rate:
            # Create a resample transform
            resample_transform = torchaudio.transforms.Resample(orig_freq=noise_sample_rate, new_freq=sample_rate)
            # Apply the resample transform to the waveform
            noise_waveform = resample_transform(noise_waveform)
        
        # Adjust noise length to original sequence length
        noise_waveform = self.adjust_audio_length(audio_tensor=noise_waveform, target_length=waveform.shape[-1])

        # Sample augmentation and sample parameters
        for key, args in self.funcs_w_args.items():
            sampled_args = dict(sample_rate = sample_rate, noise_audio = noise_waveform)
            for arg_key, val in args.items():
                if isinstance(val, ListConfig):
                    if isinstance(val[0], int) and isinstance(val[1], int):
                        sampled_args[arg_key] = random.randint(*val)
                    else:
                        sampled_args[arg_key] = random.uniform(*val)
                else:
                    sampled_args[arg_key] = val
            
            noise_waveform = self.aug_key_to_func[key](**sampled_args)
        
        if noise_waveform.shape[0] > 1:
            noise_waveform = noise_waveform.mean(dim=0)

        merged_waveform = torch.concat([waveform, noise_waveform.unsqueeze(0)], dim=0)
        assert merged_waveform.shape[-1] == waveform.shape[-1], f"merged shape: {merged_waveform.shape}, waveform shape: {waveform.shape}"
        return merged_waveform.mean(dim=0).unsqueeze(0)

class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                 pad=True, with_path=False, sample_rate=None,
                 channels=None, convert=False, mel_args=None, dynamic_noiser_args:dict=None):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.convert = convert

        if dynamic_noiser_args:
            self.dynamic_noiser = DynamicNoiser(**dynamic_noiser_args)
        else:
            self.dynamic_noiser = None

        if mel_args is not None:
            self.mel_spectrogram = MelSpectrogram(sample_rate=sample_rate, **mel_args)
        else:
            self.mel_spectrogram = None

        for _, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = self.length
            if torchaudio.get_audio_backend() in ['soundfile', 'sox_io']:
                out, sr = torchaudio.load(str(file),
                                          frame_offset=offset,
                                          num_frames=num_frames or -1)
            else:
                out, sr = torchaudio.load(str(file), offset=offset, num_frames=num_frames)
            
            # TODO: This is a hack to convert to mono
            if out.shape[0] == 2:
                out = out.mean(dim=0, keepdim=True)

            target_sr = self.sample_rate or sr
            target_channels = self.channels or out.shape[0]
            if self.convert:
                out = convert_audio(out, sr, target_sr, target_channels)
            else:
                if sr != target_sr:
                    raise RuntimeError(f"Expected {file} to have sample rate of "
                                       f"{target_sr}, but got {sr}")
                if out.shape[0] != target_channels:
                    raise RuntimeError(f"Expected {file} to have sample rate of "
                                       f"{target_channels}, but got {sr}")
            
            if self.dynamic_noiser:
                out = self.dynamic_noiser.forward(waveform=out, sample_rate=target_sr)

            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))
            
            if self.mel_spectrogram:
                mel_out = self.mel_spectrogram(out)
                out = torch.concat([out, mel_out], dim=0)

            if self.with_path:
                return out, file
            else:
                return out


if __name__ == "__main__":
    meta = []
    for path in sys.argv[1:]:
        meta += find_audio_files(path)
    json.dump(meta, sys.stdout, indent=4)
