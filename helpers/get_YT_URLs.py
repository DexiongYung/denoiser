from googleapiclient.discovery import build
from pytube import YouTube
import os
import argparse
import subprocess
from pydub import AudioSegment
import math
from pydub import AudioSegment
import os


def create_folder_if_not_exists(folder_path):
    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # If it does not exist, create it
        os.makedirs(folder_path)
        print(f"The folder '{folder_path}' was created.")
    else:
        print(f"The folder '{folder_path}' already exists.")


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

class YouTubeCaller:
    def __init__(self):
        api_key = 'AIzaSyDQpbjOiMeLXYCvKXrZwTSCXJ5BJcKIWBY'
        self.youtube = build('youtube', 'v3', developerKey=api_key)


    def get_urls_from_youtube_search(self, query, max_results=2000):
        # Make a search request
        request = self.youtube.search().list(
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


def main():
    # Create the parser
    parser = argparse.ArgumentParser(description='Process a query string.')

    # Add the "query" argument
    # The "type" is set to str, indicating the expected type is a string
    parser.add_argument('--query', type=str, required=True, help='The query string to process')
    parser.add_argument('--output_fp', type=str, required=True, help='Output folder path')
    parser.add_argument('--max_outputs', type=int, required=True, help='Maximum number of query URLs to return')
    # Parse the command-line arguments
    args = parser.parse_args()

    create_folder_if_not_exists(args.output_fp)

    yt = YouTubeCaller()
    urls = yt.get_urls_from_youtube_search(query=args.query, max_results=args.max_outputs)

    for url in urls:
        download_audio_as_wav(url, args.output_fp)


if __name__ == '__main__':
    main()