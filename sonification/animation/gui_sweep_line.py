import matplotlib.animation as animation
from pygame import mixer


def create_gui_sweep_animation(fig, ax, xmin, ymin, xmax, ymax, length_of_file, duration, t_data, 
                               midi_data, all_line_coords, update_func):
    '''
    AIM: create the moving red sweeping bar shown during in-GUI playback of the audio.
    '''
    
    line, = ax.plot([], [], lw=1)
    sweep_line, v = ax.plot(xmin, ymin, xmax, ymax, lw=2, color='red')
    
    len_of_song_ms = (length_of_file - duration) * 1.e3
    
    nintervals = len(midi_data)-1
    
    duration_interval = (len_of_song_ms / nintervals)
    
    line_anim = animation.FuncAnimation(fig, update_func, frames=len(t_data), interval=duration_interval,
                                        fargs=(sweep_line, all_line_coords, t_data, length_of_file, duration),
                                        blit=True, repeat=False)
    return sweep_line, line_anim


def update_gui_sweep(num, line, all_line_coords, t_data, length_of_file, duration):
    '''
    AIM: update position of the sweeping line/bar/1D rectangle
    '''
    
    current_pos = mixer.music.get_pos() #milliseconds
    current_time_sec = (current_pos / 1.e3) #seconds
    
    #find the index corresponding to the current time
    frame = min(int((current_time_sec / (length_of_file - duration)) * len(t_data)), len(t_data)-1)
    
    line_xdat, line_ydat = map(list, zip(*all_line_coords[frame]))
    line.set_data([line_xdat[0], line_xdat[-1]], [line_ydat[0], line_ydat[-1]])
    return line,