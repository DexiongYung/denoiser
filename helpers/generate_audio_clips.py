import os
import pandas as pd
from pydub import AudioSegment


def clip_audio(input_file, output_file, start_time_ms, end_time_ms):
    """
    Clips a part of an audio file and saves it as a new file.

    :param input_file: Path to the input audio file.
    :param output_file: Path where the clipped audio will be saved.
    :param start_time: Start time in milliseconds to clip the audio.
    :param end_time: End time in milliseconds to clip the audio.
    """
    clip_length = end_time_ms - start_time_ms
    clip_length /= 1000
    # Check if clip length is 0 or over 38 seconds
    
    if clip_length < 4 or clip_length > 38:
        print("Clip length is either 4 or over 38 seconds. Operation aborted.")
        return None
    # Load the audio file
    audio = AudioSegment.from_file(input_file)
    
    clipped_audio = audio[start_time_ms:end_time_ms]
    
    # Save the clipped audio to a file
    clipped_audio.export(output_file, format="wav")  # You can change the format if needed


def f(x):
    start = x["start_time"]
    end = x["end_time"]

    base_name = os.path.basename(x["file_path"])
    file_name_without_extension = os.path.splitext(base_name)[0]
    out = f"/mnt/dylan_disk/clips/{file_name_without_extension}_s{str(start)}_e{str(end)}.wav"

    clip_audio(input_file=x["file_path"], output_file=out, start_time_ms=start, end_time_ms=end)

df = pd.read_csv("/home/vapi/End-of-turn/Data/start_end_audio.csv")
df.apply(f, axis=1)