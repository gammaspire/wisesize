from io import BytesIO

from mido import MidiFile
from pygame import mixer


def midi_to_memfile(midi_file):
    '''
    AIM: self-explanatory. convert the MIDI array to a memfile.
    * memfile is a sort of in-memory byte stream. can play the notes in realtime without needing to save+open+play+close.
    '''
    memfile = BytesIO()
    midi_file.writeFile(memfile)
    memfile.seek(0)   #rewind to t=0?
    return memfile


def get_midi_length(midi_file):
    '''
    AIM: determine the playback length of a MIDIFile object, in seconds.
    '''
    memfile = BytesIO()
    midi_file.writeFile(memfile)
    memfile.seek(0)
    mid = MidiFile(file=memfile)
    return mid.length


def play_memfile(memfile):
    '''
    AIM: play an in-memory MIDI file using the pygame imports
    '''
    mixer.init()
    memfile.seek(0)
    mixer.music.load(memfile)
    mixer.music.play()