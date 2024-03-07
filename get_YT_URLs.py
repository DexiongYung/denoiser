from googleapiclient.discovery import build
from pytube import YouTube
import os
import subprocess
from pydub import AudioSegment
import math
from pydub import AudioSegment
import os


def download_audio_as_wav(youtube_url, output_path):
    # Download the audio stream
    yt = YouTube(youtube_url)
    audio_stream = yt.streams.get_audio_only()
    audio_file = audio_stream.download(output_path=output_path, filename_prefix="audio_")

    # Convert the downloaded file to WAV using ffmpeg
    wav_filename = os.path.join(output_path, "audio_" + yt.title.replace("/", "_") + ".wav")  # Replace "/" with "_" to avoid file path issues
    subprocess.run(['ffmpeg', '-i', audio_file, wav_filename])

    # Optionally, remove the original downloaded file
    os.remove(audio_file)


# Set up YouTube Data API
api_key = 'AIzaSyDQpbjOiMeLXYCvKXrZwTSCXJ5BJcKIWBY'
youtube = build('youtube', 'v3', developerKey=api_key)

def youtube_search(query, max_results=2000):
    # Make a search request
    request = youtube.search().list(
        q=query,
        part='id,snippet',
        type='video',
        maxResults=max_results
    )
    response = request.execute()

    # Extract video links
    videos = []
    for item in response['items']:
        video_id = item['id']['videoId']
        video_link = f'https://www.youtube.com/watch?v={video_id}'
        videos.append(video_link)

    return videos


def chunk_and_delete_original(file_path, chunk_length_sec=350):
    # Load the WAV file
    audio = AudioSegment.from_file(file_path)

    # Calculate the number of chunks
    audio_length_sec = len(audio) / 1000  # Convert from milliseconds to seconds
    num_chunks = math.ceil(audio_length_sec / chunk_length_sec)

    # Only proceed if audio is longer than the chunk length; otherwise, no need to chunk
    if audio_length_sec > chunk_length_sec:
        for i in range(num_chunks):
            start_ms = i * chunk_length_sec * 1000
            end_ms = start_ms + chunk_length_sec * 1000
            chunk = audio[start_ms:end_ms]

            # Define the filename for the chunk
            chunk_filename = f"{file_path[:-4]}_chunk{i}.wav"
            
            # Export the chunk as a WAV file
            chunk.export(chunk_filename, format="wav")
            print(f"Exported {chunk_filename}")

    # Delete the original file
    os.remove(file_path)
    print(f"Deleted the original file: {file_path}")


def process_all_wav_files(folder_path, f):
    # List all files in the specified folder
    for filename in os.listdir(folder_path):
        # Check if the file is a WAV file
        if filename.endswith(".wav"):
            # Construct the full file path
            file_path = os.path.join(folder_path, filename)
            # Process the WAV file
            f(file_path)


def delete_long_audio_files(folder_path, max_duration_seconds=38):
    # List all files in the given folder
    for filename in os.listdir(folder_path):
        # Check if the file is an audio file (you might need to adjust the extensions)
        if filename.endswith(('.mp3', '.wav', '.flac', '.ogg')):
            try:
                # Load the audio file
                audio_path = os.path.join(folder_path, filename)
                audio = AudioSegment.from_file(audio_path)
                
                # Check the duration of the audio file
                duration_seconds = len(audio) / 1000.0 # pydub uses milliseconds
                
                # If the audio is longer than the max_duration, delete it
                if duration_seconds > max_duration_seconds:
                    os.remove(audio_path)
                    print(f"Deleted {filename} as it was longer than {max_duration_seconds} seconds.")
            except Exception as e:
                print(f"Could not process {filename}. Error: {e}")
