import numpy as np
from audiolazy import str2midi
from midiutil import MIDIFile


#dictionary for different key signatures
NOTE_DICT = {
   'C Major': 'C2-D2-E2-F2-G2-A2-B2-C3-D3-E3-F3-G3-A3-B3-C4-D4-E4-F4-G4-A4-B4-C5-D5-E5-F5-G5-A5-B5',
   'G Major': 'G1-A1-B1-C2-D2-E2-F#2-G2-A2-B2-C3-D3-E3-F#3-G3-A3-B3-C4-D4-E4-F#4-G4-A4-B4-C5-D5-E5-F#5',
   'D Major': 'D2-E2-F#2-G2-A2-B2-C#3-D3-E3-F#3-G3-A3-B3-C#4-D4-E4-F#4-G4-A4-B4-C#5-D5-E5-F#5-G5-A5-B5-C#6',
   'A Major': 'A1-B1-C#2-D2-E2-F#2-G#2-A2-B2-C#3-D3-E3-F#3-G#3-A3-B3-C#4-D4-E4-F#4-G#4-A4-B4-C#5-D5-E5-F#5-G#5',
   'E Major': 'E2-F#2-G#2-A2-B2-C#3-D#3-E3-F#3-G#3-A3-B3-C#4-D#4-E4-F#4-G#4-A4-B4-C#5-D#5-E5-F#5-G#5-A5-B5-C#6-D#6',
   'B Major': 'B1-C#2-D#2-E2-F#2-G#2-A#2-B3-C#3-D#3-E3-F#3-G#3-A#3-B4-C#4-D#4-E4-F#4-G#4-A#4-B5-C#5-D#5-E5-F#5-G#5-A#5',
   'F# Major':'F#2-G#2-A#2-B2-C#3-D#3-E#3-F#3-G#3-A#3-B3-C#4-D#4-E#4-F#4-G#4-A#4-B4-C#5-D#5-E#5-F#5-G#5-A#5-B5-C#6-D#6-E#6', 
   'Gb Major':'Gb1-Ab1-Bb1-Cb2-Db2-Eb2-F2-Gb2-Ab2-Bb2-Cb3-Db3-Eb3-F3-Gb3-Ab3-Bb3-Cb4-Db4-Eb4-F4-Gb4-Ab4-Bb4-Cb5-Db5-Eb5-F5',
   'Db Major':'Db2-Eb2-F2-Gb2-Ab2-Bb2-C3-Db3-Eb3-F3-Gb3-Ab3-Bb3-C4-Db4-Eb4-F4-Gb4-Ab4-Bb4-C5-Db5-Eb5-F5-Gb5-Ab5-Bb5-C6',
   'Ab Major':'Ab1-Bb1-C2-Db2-Eb2-F2-G2-Ab2-Bb2-C3-Db3-Eb3-F3-G3-Ab3-Bb3-C4-Db4-Eb4-F4-G4-Ab4-Bb4-C5-Db5-Eb5-F5-G5', 
   'Eb Major':'Eb2-F2-G2-Ab2-Bb2-C3-D3-Eb3-F3-G3-Ab3-Bb3-C4-D4-Eb4-F4-G4-Ab4-Bb4-C5-D5-Eb5-F5-G5-Ab5-Bb5-C6-D6',
   'Bb Major':'Bb1-C2-D2-Eb2-F2-G2-A2-Bb2-C3-D3-Eb3-F3-G3-A3-Bb3-C4-D4-Eb4-F4-G4-A4-Bb4-C5-D5-Eb5-F5-G5-A5',
   'F Major': 'F2-G2-A2-Bb2-C3-D3-E3-F3-G3-A3-Bb3-C4-D4-E4-F4-G4-A4-Bb4-C5-D5-E5-F5-G5-A5-Bb5-C6-D6-E6'}


#typical sonification mapping function; maps value(s) from one range to another range; returns floatsN
def map_value(value, min_value, max_value, min_result, max_result):
        '''
        AIM: normalization of some value(s) from an array (with min_value, max_value) onto a new scale (with min_result, max_result).
        '''
        result = min_result + (value - min_value)/(max_value - min_value)*(max_result - min_result)
        return result
    

def build_scaled_data(mean_list, y_scale=0.5):
    '''
    AIM: normalize the mean pixel values to [0,1] then apply the y-scale.
    '''
    y_data = map_value(np.asarray(mean_list), min(mean_list), max(mean_list), 0, 1)
    return y_data ** y_scale
    
    

def build_midi_data(mean_list, note_names, y_scale=0.5, reverse=False):
    '''
    AIM: map average pixel strip intensities onto MIDI data
    '''
    
    y_data_scaled = build_scaled_data(mean_list, y_scale=y_scale)

    #the following converts note names into midi notes
    note_midis = [str2midi(n) for n in note_names]  #list of midi note numbers
    n_notes = len(note_midis)

    #MAPPING DATA TO THE MIDI NOTES!        
    midi_data = []
    #for every data point, map y_data_scaled values such that smallest/largest px is lowest/highest note
    for value in y_data_scaled:   #assigns midi note number to whichever y_data_scaled[i] is nearest
        #apply the "note inversion" if desired --> high values either assigned high notes or, if inverted, low notes
        if not reverse:
            note_index = round(map_value(value, 0, 1, 0, n_notes-1))
        else:
            note_index = round(map_value(value, 0, 1, n_notes-1, 0))
        midi_data.append(note_midis[note_index])

    #print('midi_data',midi_data)
    return midi_data


def build_velocity_data(mean_list, vel_min, vel_max, y_scale=0.5):
    '''
    AIM: map average strip intensities onto MIDI velocities. velocity dictates the sound volume!
    '''
    y_data_scaled = build_scaled_data(mean_list, y_scale=y_scale)
    
    vel_data = []
    for value in y_data_scaled:
        note_velocity = round(map_value(value, 0, 1, vel_min, vel_max)) #larger values, heavier sound
        vel_data.append(note_velocity)
    
    return vel_data


def build_time_data(mean_list, strips_per_beat=10):
    '''
    AIM: convert number of strips to time "step" coordinates. 
    '''
    return (np.arange(len(mean_list)) / strips_per_beat)


def build_relative_track(midi_data, midi_data_alt, threshold=0.10):
    '''
    AIM: create the comparison track for W1 and W3 data.
    * this "comparison" will comprise a percussion (at full volume) that sounds whenever the difference between the W1 and W3 tracks exceeds the threshold of 0.10. 
    * this track is then superimposed on the merged wavfile of the W1 and W3 sonified data.
    '''

    #calculate relative difference between the W1 and W3 tracks
    relative_diff = (np.asarray(midi_data)-np.asarray(midi_data_alt))/(np.asarray(midi_data))
    
    #convert relative difference to a bool array
    bool_vals = (np.abs(relative_diff)>threshold)
    
    #convert bools to integers, then multiply by 127 (the maximum velocity). 
    #That is, the relative_diff values that pass the threshold will have volume while the others will be silent.
    
    #I then convert this array to a list.
    
    relative_vel = np.ndarray.tolist(bool_vals.astype(int)*127)

    #little to no relative difference? NO VOLUME (relative_vel value is 0)! 
    #if there is a difference, then lowest midi note value (relative_vel value 127 -- so there is volume)
    relative_midi = np.ndarray.tolist(np.zeros(len(relative_vel)) + np.min(midi_data))
    
    return relative_midi, relative_vel


#for ONE wavelength band
def build_single_track_midi(midi_data, vel_data, t_data, bpm, duration, program):
    '''
    AIM: create a single track MIDIFile object from the pitch, velocity, and time data.
    program : the INSTRUMENT! must be an integer.
    '''
    midi_file = MIDIFile(1)
    
    midi_file.addTempo(track=0, time=0, tempo=bpm)
    
    midi_file.addProgramChange(tracknum=0, channel=0, time=0, program=program)
    
    for pitch, velocity, time in zip(midi_data, vel_data, t_data):
        midi_file.addNote(track=0, channel=0, pitch=int(pitch), time=float(time), duration=duration, volume=int(velocity))
        
    return midi_file


#for ONE SINGLE NOTE of the ONE single wavelength band...
#sort of a "builder function," if you will.
def build_preview_midi(pitch, velocity, bpm, program, duration=0.5):
    '''
    AIM: create a one-note MIDIFile object for preview playback when the user clicks a strip on the GUI.
    '''
    midi_file = MIDIFile(1)
    midi_file.addTrackName(0, 0, 'Preview')
    
    midi_file.addTempo(track=0, time=0, tempo=bpm)
    midi_file.addProgramChange(tracknum=0, channel=0, time=0, program=program)
    midi_file.addNote(track=0, channel=0, pitch=int(pitch), time=0, duration=duration, volume=int(velocity))
    
    return midi_file


#for the W1 + W3 OVERLAY!
def build_overlay_track_midi(midi_data, midi_data_alt, t_data, bpm, duration, program, compare_program=47, threshold=0.10):
    '''
    AIM: create a three track MIDIFile object containing
    * Track 0 -- first wavelength band
    * Track 1 -- comparison wavelength band
    * Track 2 -- percussion/comparison track (program = 47) indicating "significant" differences
    relative_midi, relative_vel : the Track 2 specifications...when to play percussion, when to not.
    '''
    
    relative_midi, relative_vel = build_relative_track(midi_data, midi_data_alt, threshold=threshold)
    
    midi_file = MIDIFile(3)
    
    for track in range(3):
        midi_file.addTempo(track=track, time=0, tempo=bpm)
        
    midi_file.addProgramChange(tracknum=0, channel=0, time=0, program=program)

    midi_file.addProgramChange(tracknum=1, channel=0, time=0, program=program)

    midi_file.addProgramChange(tracknum=2, channel=1, time=0, program=compare_program)
        
    for (p1, p2, p_rel, v_rel, t) in zip(midi_data, midi_data_alt, relative_midi, relative_vel, t_data):
        midi_file.addNote(track=0, channel=0, pitch=int(p1), time=float(t), duration=duration, volume=100)
        midi_file.addNote(track=1, channel=0, pitch=int(p2), time=float(t), duration=duration, volume=100)
        midi_file.addNote(track=2, channel=1, pitch=int(p_rel), time=float(t), duration=duration, volume=v_rel)

    return midi_file