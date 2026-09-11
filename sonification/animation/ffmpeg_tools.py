import ffmpeg
import os


def unique_filename(base_path, counter_start=0):
    '''
    AIM: Generate a non-overwriting filename.
    '''

    counter = counter_start

    root, ext = os.path.splitext(base_path)   #splits into path (root) and file (ext)

    while os.path.exists(f'{root}-{counter}{ext}'):
        counter += 1

    return f'{root}-{counter}{ext}', counter


def merge_audio_video(video_path, audio_path, output_path, overwrite=False):
    '''
    Merge a WAV audio file and MP4 animation into a final MP4!
    '''
    
    if overwrite and os.path.exists(output_path):
        os.remove(output_path)
        
    input_video = ffmpeg.input(video_path)
    input_audio = ffmpeg.input(audio_path)
    
    #uncomment this line for a "test" version of the merged MP4 output
    #ffmpeg.output(input_video.video, input_audio.audio, '/Users/k215c316/Desktop/test.mp4',codec='copy').run(quiet=True)
    
    ffmpeg.output(input_video.video, input_audio.audio, output_path, codec='copy').run(quiet=True)