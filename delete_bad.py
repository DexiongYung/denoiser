import os
import wave

def delete_zero_length_wav_files(folder_path):
    # List all files in the given folder
    for filename in os.listdir(folder_path):
        # Check if the file is a WAV file
        if filename.endswith('.wav'):
            file_path = os.path.join(folder_path, filename)
            try:
                # Open the WAV file and get its frame rate and number of frames
                with wave.open(file_path, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    # Calculate the duration of the WAV file
                    duration = frames / float(rate)
                    # If the file's duration is 0 seconds, delete the file
                    if duration == 0:
                        os.remove(file_path)
                        print(f"Deleted {file_path} because it is 0 seconds long.")
            except wave.Error as e:
                print(f"Error opening {file_path}: {e}")

# Replace 'your_folder_path_here' with the path to the folder you want to clean up
folder_path = '/mnt/dylan_disk/audio_clipped'
delete_zero_length_wav_files(folder_path)
