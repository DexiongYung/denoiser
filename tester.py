import torchaudio
from denoiser.audio import DynamicNoiser

audio_clip_path = 'clean.wav'
audio_clip, audio_sr = torchaudio.load(audio_clip_path)

func_w_args = {"apply_echo": {"delay_ms": (50, 500), "decay": (0.05, 0.5), "repeats": (0, 3)}, 
               "dynamic_noise_distance_adjustment":{"min_distance": 2, "max_distance": 6, "segments": (1, 5)}}

dn = DynamicNoiser("/mnt/dylan_disk/YT_audio", funcs_w_args=func_w_args)

for i in range(5):
    noised_audio = dn.forward(waveform=audio_clip[0], sample_rate=audio_sr)
    torchaudio.save(f"./{i}.wav", noised_audio.unsqueeze(0), audio_sr)