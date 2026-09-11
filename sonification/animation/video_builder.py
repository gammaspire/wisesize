import numpy as np
from sonification.midi_builder import map_value
import matplotlib.animation as animation

from matplotlib import figure
from scipy.stats import scoreatpercentile
from astropy.visualization import simple_norm


def create_base_figure(dat, band, galaxy_name, t_data, midi_data, vel_data, xmin, xmax, ymin, ymax, v1, v2):
    '''
    AIM: build the matplotlib.Figure components shared by both the single-band and overlay MP4 animations...
    '''
    fig = figure.Figure(layout='constrained')
    
    spec = fig.add_gridspec(2,2)
    
    ax1 = fig.add_subplot(spec[0,:])
    ax2 = fig.add_subplot(spec[1,0])
    
    #NOTE: only visible when adding the W1/W3 overlay...can ignore for now (set invisible)
    ax3 = fig.add_subplot(spec[1,1])
    ax3.set_visible(False)

    ax1.scatter(t_data, midi_data, vel_data, alpha=0.5, edgecolors='black', color='green', label=band)
    
    norm_im = simple_norm(dat, 'asinh', min_percent=0.5, max_percent=99.9, min_cut=v1, max_cut=v2)
    
    #the galaxy image
    ax2.imshow(dat, origin='lower', norm=norm_im, cmap='gray', alpha=0.9)
    ax2.text(0.05, 0.95, band, horizontalalignment='left', verticalalignment='top', transform=ax2.transAxes,
             color='green', fontsize=13, fontweight='bold', backgroundcolor='white')
    
    #the scatterplot "highlighting circle" initialization
    point1a, = ax1.plot([], [], 'ro', markersize=20, alpha=0.2)
    
    #the galaxy image's "sweeping bar" initialization
    line2, = ax2.plot([], [], lw=1)
    
    sweep_line2, v = ax2.plot(xmin, ymin, xmax, ymax, lw=2, color='green')

    ax1.set_xlabel('Time interval (s)', fontsize=12)
    ax1.set_ylabel('MIDI note', fontsize=12)
    fig.suptitle(galaxy_name,fontsize=15)

    return {'fig':fig, 'spec':spec, 'ax1':ax1, 'ax2':ax2, 'ax3':ax3, 'point1a':point1a, 'line2':sweep_line2}


def add_overlay_subplot(ax1, ax3, dat_alt, band_alt, t_data, midi_data_alt, 
                        vel_data_alt, xmin, xmax, ymin, ymax):
    '''
    AIM: Create the W1/W3 overlay subplot and add alternate MIDI points.
    '''

    ax3.set_visible(True)
    
    ax1.scatter(t_data, midi_data_alt, vel_data_alt, alpha=0.5, edgecolors='black', color='tab:orange', label=band_alt)

    v1 = scoreatpercentile(dat_alt,0.5)
    v2 = scoreatpercentile(dat_alt,99.9)

    norm_im = simple_norm(dat_alt, 'asinh', min_percent=0.5, max_percent=99.9, min_cut=v1, max_cut=v2)

    ax3.imshow(dat_alt, origin='lower', norm=norm_im, cmap='gray', alpha=0.9)

    sweep_line3, = ax3.plot([xmin, xmax], [ymin, ymax], lw=2, color='tab:orange')

    ax3.text(0.05, 0.95, band_alt, horizontalalignment='left', verticalalignment='top', 
             transform=ax3.transAxes, color='tab:orange', fontsize=13, fontweight='bold', backgroundcolor='white')
    
    point1b, = ax1.plot([], [], 'bo', markersize=20, alpha=0.2)
    
    return {'ax1':ax1, 'ax3': ax3, 'point1b':point1b, 'line3': sweep_line3}


#FOR SINGLE-BAND
def update_line_one(num, point1a, line2, xvals_anim, t_data, midi_data, all_line_coords):
    '''
    AIM: update animation objects for a single band animation -- will use for the MP4
    '''
    
    i = int(xvals_anim[num])
    
    point1a.set_data(t_data[i], midi_data[i])
    
    line_xdat, line_ydat = map(list, zip(*all_line_coords[i]))
    
    line2.set_data([line_xdat[0], line_xdat[-1]], [line_ydat[0], line_ydat[-1]])
    
    return point1a, line2,


#USED FOR W1+W3 MERGER
def update_line_all(num, point1a, point1b, line2, line3, xvals_anim, t_data, midi_data, midi_data_alt, all_line_coords):
    '''
    AIM: update animation objects for W1/W3 overlay animation :-)
    '''
    #i = self.xvals_anim[num]
    #line1.set_data([i, i], [self.ymin_anim-5, self.ymax_anim+5])

    xvals = map_value(xvals_anim,0,np.max(xvals_anim),0,len(midi_data)-1)
    i = int(xvals[num])
    point1a.set_data(t_data[i],midi_data[i])
    point1b.set_data(t_data[i],midi_data_alt[i])

    xvals_alt = map_value(xvals_anim,0,np.max(xvals_anim),0,len(all_line_coords)-1)
    i_alt = int(xvals_alt[num])

    line_xdat, line_ydat = map(list, zip(*all_line_coords[i_alt]))
    line2.set_data([line_xdat[0], line_xdat[-1]], [line_ydat[0], line_ydat[-1]])
    line3.set_data([line_xdat[0], line_xdat[-1]], [line_ydat[0], line_ydat[-1]])

    return point1a, point1b, line2, line3,
    

def build_single_band_animation(fig, point1a, line2, xvals_anim, t_data, midi_data, all_line_coords):
    '''
    AIM: create a FuncAnimation object for the single-band MP4 animation
    '''
    line_anim = animation.FuncAnimation(fig, update_line_one, frames=len(xvals_anim), 
                                        fargs=(point1a, line2, xvals_anim, t_data, midi_data, all_line_coords),
                                        blit=True)
    return line_anim


def build_overlay_animation(fig, point1a, point1b, line2, line3, xvals_anim, t_data, 
                            midi_data, midi_data_alt, all_line_coords):
    '''
    AIM:
    Create a FuncAnimation object for the W1/W3 overlay animation
    '''

    line_anim = animation.FuncAnimation(fig, update_line_all, frames=len(xvals_anim),
                                        fargs=(point1a, point1b, line2, line3, xvals_anim, t_data, 
                                               midi_data, midi_data_alt, all_line_coords), blit=True)
    return line_anim


def save_animation(line_anim, output_path, n_frames, length_of_file):
    '''
    AIM: save a matplotlib animation as an MP4 file!
    '''
    fps = n_frames / length_of_file
    print(f'Frames: {n_frames}')
    print(f'FPS: {fps}')
    
    line_anim.save(output_path, fps=fps)