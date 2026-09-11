'''
Class layout adapted from 
https://stackoverflow.com/questions/7546050/switch-between-two-frames-in-tkinter/7557028#7557028
'''

import sys

from midi2audio import FluidSynth
from pygame import mixer                    #this library is what causes the loading delay methinks

import tkinter as tk
import numpy as np
import os
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from matplotlib import figure              #see self.fig, self.ax.

import matplotlib                          #I need this for matplotlib.use. sowwee.
matplotlib.use('TkAgg')                    #strange error messages will appear otherwise.

from scipy.stats import scoreatpercentile
from astropy.visualization import simple_norm
from astropy.io import fits
from tkinter import font as tkFont
from tkinter import messagebox
from tkinter import filedialog

from config.settings import read_params
from imaging.load_fits import (load_fits, get_galaxy_info, load_overlay_image)
from imaging.masks import (load_mask, create_default_mask, apply_mask)
from imaging.rectangles import (get_rectangle_bounds, find_closest_bar, get_xym, 
                                sample_rotated_rectangle, sample_vertical_rectangle)
from sonification.midi_builder import *
from sonification.playback import (midi_to_memfile, get_midi_length, play_memfile)
from animation.gui_sweep_line import (create_gui_sweep_animation, update_gui_sweep)
from animation.ffmpeg_tools import (unique_filename, merge_audio_video)
from animation.video_builder import (create_base_figure, add_overlay_subplot, 
                                     build_single_band_animation, build_overlay_animation, save_animation)

#create main window container, into which the first page will be placed.
class App(tk.Tk):
    
    def __init__(self, path_to_repos, initial_browsedir, soundfont, window_geometry):          #INITIALIZE; will always run when App class is called.
        tk.Tk.__init__(self)     #initialize tkinter; *args are parameter arguments, **kwargs can be dictionary arguments
        
        self.title('MIDI-chlorians: Sonification of Nearby Galaxies')
        self.geometry(window_geometry)
        self.resizable(True,True)
        self.rowspan=10
        
        #will be filled with heaps of frames and frames of heaps. 
        container = tk.Frame(self)
        container.pack(side='top', fill='both', expand=True)     #fills entire container space
        container.grid_rowconfigure(0,weight=1)
        container.grid_columnconfigure(0,weight=1)

        ## Initialize Frames
        self.frames = {}     #empty dictionary
        frame = MainPage(container, self, path_to_repos, initial_browsedir, soundfont)   #define frame  
        self.frames[MainPage] = frame     #assign new dictionary entry {MainPage: frame}
        frame.grid(row=0,column=0,sticky='nsew')   #define where to place frame within the container...CENTER!
        for i in range(self.rowspan):
            frame.columnconfigure(i, weight=1)
            frame.rowconfigure(i, weight=1)
        
        self.show_frame(MainPage)  #a method to be defined below (see MainPage class)
    
    def show_frame(self, cont):     #'cont' represents the controller, enables switching between frames/windows...I think.
        frame = self.frames[cont]
        frame.tkraise()   #will raise window/frame to the 'front;' if there is more than one frame, quite handy.
        
        
#inherits all from tk.Frame; will be on first window
class MainPage(tk.Frame):    
    
    def __init__(self, parent, controller, path_to_repos, initial_browsedir, soundfont):
        
        #generalized parameters given in params.txt file
        self.path_to_repos = path_to_repos
        self.initial_browsedir = initial_browsedir
        self.soundfont = soundfont
        
        #these variables will apply to the self.drawSq function, if the user desires to use it.
        self.bound_check=None
        self.x1=None
        self.x2=None
        self.y1=None
        self.y2=None
        self.angle=0
        
        #initiate a counter to ensure that files do not overwrite one another for an individual galaxy
        #note: NEEDED FOR THE SAVE WIDGET
        self.namecounter=0
        self.namecounter_ani=0
        self.namecounter_ani_both=0
        
        #isolate the key signature names --> need for the dropdown menu
        self.keyvar_options=list(NOTE_DICT.keys())

        self.keyvar = tk.StringVar()
        self.keyvar.set(self.keyvar_options[2])
        
        #defines the number of rows/columns to resize when resizing the entire window.
        self.rowspan=10
        
        #define a font
        self.helv20 = tkFont.Font(family='Helvetica', size=20, weight='bold')
        
        #first frame...
        tk.Frame.__init__(self,parent)
        
        #NOTE: columnconfigure and rowconfigure below enable the minimization and maximization of window to also affect widget size
        
        #create frame for save widgets...y'know, to generate the .wav and .mp4 
        self.frame_save=tk.LabelFrame(self,text='Save Files',padx=5,pady=5)
        self.frame_save.grid(row=4,column=1,columnspan=5)
        for i in range(self.rowspan):
            self.frame_save.columnconfigure(i,weight=1)
            self.frame_save.rowconfigure(i,weight=1)
        
        #create display frame, which will hold the canvas and a few button widgets underneath.
        self.frame_display=tk.LabelFrame(self,text='Display',font='Vendana 15',padx=5,pady=5)
        self.frame_display.grid(row=0,column=0,rowspan=9)
        for i in range(self.rowspan):
            self.frame_display.columnconfigure(i, weight=1)
            self.frame_display.rowconfigure(i, weight=1)
        
        #create buttons frame, which currently only holds the 'save' button, 'browse' button, and entry box.
        self.frame_buttons=tk.LabelFrame(self,text='File Browser',padx=5,pady=5)
        self.frame_buttons.grid(row=0,column=1,columnspan=2)
        for i in range(self.rowspan):
            self.frame_buttons.columnconfigure(i, weight=1)
            self.frame_buttons.rowconfigure(i, weight=1)
            
        #create soni frame, which holds the event button for converting data into sound (midifile).
        #there are also heaps of text boxes with which the user can manipulate the sound conversion parameters
        self.frame_soni=tk.LabelFrame(self,text='Parameters (Click "Sonify" to play)',padx=5,pady=5)
        self.frame_soni.grid(row=7,column=2,rowspan=2,sticky='se')
        for i in range(self.rowspan):
            self.frame_soni.columnconfigure(i, weight=1)
            self.frame_soni.rowconfigure(i, weight=1)
        
        #create editcanvas frame --> manipulates vmin, vmax, cmap of the display image
        self.frame_editcanvas = tk.LabelFrame(self,text='Change Display',padx=5,pady=5)
        self.frame_editcanvas.grid(row=7,column=1,sticky='s')
        for i in range(self.rowspan):
            self.frame_editcanvas.columnconfigure(i, weight=1)
            self.frame_editcanvas.columnconfigure(i, weight=1)
            
        #create box frame --> check boxes for lines vs. squares when interacting with the figure canvas
        self.frame_box = tk.LabelFrame(self,text='Change Rectangle Angle',padx=5,pady=5)
        self.frame_box.grid(row=8,column=1,sticky='s')
        for i in range(self.rowspan):
            self.frame_box.columnconfigure(i, weight=1)
            self.frame_box.rowconfigure(i, weight=1)
        
        self.galaxy_to_display()
        '''
        INSERT INITIATION FUNCTIONS TO RUN BELOW.
        '''
        self.initiate_vals()
        self.add_info_button()
        self.populate_soni_widget()
        self.populate_box_widget()
        self.populate_save_widget()
        self.init_display_size()
        self.populate_editcanvas_widget()
    
    ##########################################
    # And now...a plethora of helper methods #
    ##########################################
    
    def populate_box_widget(self):
        self.angle_box = tk.Entry(self.frame_box, width=15, borderwidth=2, bg='black', fg='lime green',
                                  font='Arial 20')
        self.angle_box.insert(0,'Rotation angle (deg)')
        self.angle_box.grid(row=0,column=0,columnspan=5)
        self.add_angle_buttons()
    
    def initiate_vals(self):
        self.var = tk.IntVar()
        self.val = tk.Label(self.frame_display,text='Mean Pixel Value: ',font='Arial 18')
        self.val.grid(row=8,column=2,padx=1,pady=(3,1),sticky='e')
        self.line_check = tk.Checkbutton(self.frame_display,text='Switch to Lines',
                                         onvalue=1,offvalue=0,command=self.change_canvas_event,
                                         variable=self.var,font='Arial 18')
        self.line_check.grid(row=9,column=2,padx=1,pady=(3,1),sticky='e')
    
    def galaxy_to_display(self):
        self.path_to_im = tk.Entry(self.frame_buttons, width=17, borderwidth=2, bg='black', fg='lime green', 
                                   font='Arial 20')
        self.path_to_im.insert(0,'image/path.fits')
        self.path_to_im.grid(row=0,column=0,columnspan=1)
        
        self.path_to_mask = tk.Entry(self.frame_buttons,width=17, borderwidth=2, bg='black',
                                     fg='lime green', font='Arial 20')
        self.path_to_mask.insert(0,'optional/mask.fits')
        self.path_to_mask.grid(row=0,column=1,columnspan=1)
        
        self.add_browse_button()
        self.add_browse_mask_button()
        self.add_enter_button()
    
    def populate_editcanvas_widget(self):
        
        #initiate v1, v2 sliders
        self.initiate_v1v2()

        #set up cmap dropdown menu
        self.set_cmap_menu()
        
        #add showmask checkbox
        self.add_showmask_check()
    
    def initiate_v1v2(self,min_v=0, max_v=1, min_px=0, max_px=1):
        
        self.v1slider = tk.Scale(self.frame_editcanvas, from_=min_px, to=max_px, orient=tk.HORIZONTAL,
                                command=self.change_vvalues)
        self.v2slider = tk.Scale(self.frame_editcanvas, from_=min_px, to=max_px, orient=tk.HORIZONTAL,
                                command=self.change_vvalues)
        
        v1lab = tk.Label(self.frame_editcanvas,text='vmin').grid(row=0,column=0)
        v2lab = tk.Label(self.frame_editcanvas,text='vmax').grid(row=1,column=0)
        
        self.v1slider.grid(row=0,column=1)
        self.v2slider.grid(row=1,column=1)
        
    def add_showmask_check(self):
        self.showmask = tk.IntVar()
        self.showmask_box = tk.Checkbutton(self.frame_editcanvas, text='Show/Hide Mask', 
                                           font='Arial 15', onvalue=1, offvalue=0, variable=self.showmask,
                                          command=self.add_mask)
        self.showmask_box.grid(row=4, column=0, columnspan=2)
        
    def set_cmap_menu(self):
        
        self.cmap_options = ['viridis', 'rainbow', 'plasma', 'spring', 
                             'Wistia', 'cool', 'gist_heat', 'winter', 
                             'Purples', 'Greens', 'Oranges', 'gray']
        
        #set up cmap dropdown menu
        self.cmapvar = tk.StringVar()
        self.cmapvar.set(self.cmap_options[0])
        
        self.cmap_menu = tk.OptionMenu(self.frame_editcanvas, self.cmapvar, *self.cmap_options, command=self.change_cmap)
        self.cmap_menu.config(font='Arial 15',padx=5,pady=5) 
        
        cmaplab = tk.Label(self.frame_editcanvas,text='cmap').grid(row=2,column=0)
        
        self.cmap_menu.grid(row=2,column=1)
        
        self.cmaprev = tk.IntVar()
        self.reverse_cmap = tk.Checkbutton(self.frame_editcanvas,text='Invert Colorbar', onvalue=1, offvalue=0, 
                                           variable = self.cmaprev, font='Arial 15', command=self.reverse_cmap)
        self.reverse_cmap.grid(row=3,column=0,columnspan=2)
    
    def change_vvalues(self, value):
        min_val = float(self.v1slider.get())
        max_val = float(self.v2slider.get())
        self.im.set_clim(vmin=min_val, vmax=max_val)
        self.canvas.draw()   
    
    def add_mask(self):
        if self.showmask.get()>0.:
            self.im.set_data(self.dat)
        else:
            self.im.set_data(self.dat_for_display)
        self.canvas.draw()
    
    #command to change color scheme of the image
    def change_cmap(self, value): 
        self.im.set_cmap(self.cmapvar.get())
        self.canvas.draw()
    
    #command to reverse the color schemes. for this version of matplotlib, reversal is as simple as appending _r 
    def reverse_cmap(self):
        if self.cmaprev.get()==1:
            colorb = self.cmapvar.get() + '_r'
        if self.cmaprev.get()==0:
            colorb = self.cmapvar.get()
        self.im.set_cmap(colorb)
        self.canvas.draw()
    
    #function for opening the file explorer window
    def browseFiles(self):
        filename = filedialog.askopenfilename(initialdir = self.initial_browsedir,
                                              title = "Select a File", filetypes = ([("FITS Files", ".fits")]))
        self.path_to_im.delete(0,tk.END)
        self.path_to_im.insert(0,filename)    
        
    def browseFilesMask(self):
        filename_alt = filedialog.askopenfilename(initialdir = self.initial_browsedir,
                                                  title = "Select a File", filetypes = ([("FITS Files", ".fits")]))
        self.path_to_mask.delete(0,tk.END)
        self.path_to_mask.insert(0,filename_alt)
    
    def populate_save_widget(self):
        self.add_save_button()
        self.add_saveani_button()
        self.add_w1w3merge_box()
    
    def populate_soni_widget(self):
        
        self.add_midi_button()
        
        #create all entry textboxes (with labels and initial values), midi button!

        #this checkbox inverts the note assignment such that high values have low notes and low values have high notes.
        self.var_rev = tk.IntVar()
        self.rev_checkbox = tk.Checkbutton(self.frame_soni, text='Note Inversion', onvalue=1, offvalue=0, variable=self.var_rev, font='Arial 17')
        self.rev_checkbox.grid(row=0,column=0,columnspan=2)
        
        ylab = tk.Label(self.frame_soni,text='yscale').grid(row=1,column=0)
        self.y_scale_entry = tk.Entry(self.frame_soni, width=10, borderwidth=2, bg='black', fg='lime green', 
                                      font='Arial 15')
        self.y_scale_entry.insert(0,'0.5')
        self.y_scale_entry.grid(row=1,column=1,columnspan=1)
        
        vmin_lab = tk.Label(self.frame_soni,text='Min Velocity').grid(row=2,column=0)
        self.vel_min_entry = tk.Entry(self.frame_soni, width=10, borderwidth=2, bg='black', fg='lime green', 
                                      font='Arial 15')
        self.vel_min_entry.insert(0,'10')
        self.vel_min_entry.grid(row=2,column=1,columnspan=1)
        
        vmax_lab = tk.Label(self.frame_soni,text='Max Velocity').grid(row=3,column=0)
        self.vel_max_entry = tk.Entry(self.frame_soni, width=10, borderwidth=2, bg='black', fg='lime green', 
                                      font='Arial 15')
        self.vel_max_entry.insert(0,'100')
        self.vel_max_entry.grid(row=3,column=1,columnspan=1)
        
        bpm_lab = tk.Label(self.frame_soni,text='BPM').grid(row=4,column=0)
        self.bpm_entry = tk.Entry(self.frame_soni, width=10, borderwidth=2, bg='black', fg='lime green', 
                                  font='Arial 15')
        self.bpm_entry.insert(0,'35')
        self.bpm_entry.grid(row=4,column=1,columnspan=1)
        
        xminmax_lab = tk.Label(self.frame_soni,text='xmin, xmax').grid(row=5,column=0)
        self.xminmax_entry = tk.Entry(self.frame_soni, width=10, borderwidth=2, bg='black', fg='lime green',
                                      font='Arial 15')
        self.xminmax_entry.insert(0,'x1, x2')
        self.xminmax_entry.grid(row=5,column=1,columnspan=1)
        
        key_lab = tk.Label(self.frame_soni,text='Key Signature').grid(row=6,column=0)
        self.key_menu = tk.OptionMenu(self.frame_soni, self.keyvar, *self.keyvar_options)
        self.key_menu.config(bg='black',fg='black',font='Arial 15')
        self.key_menu.grid(row=6,column=1,columnspan=1)
        
        program_lab = tk.Label(self.frame_soni,text='Instrument (0-127)').grid(row=7,column=0)
        self.program_entry = tk.Entry(self.frame_soni, width=10, borderwidth=2, bg='black', fg='lime green', 
                                      font='Arial 15')
        self.program_entry.insert(0,'0')
        self.program_entry.grid(row=7,column=1,columnspan=1)
        
        duration_lab = tk.Label(self.frame_soni,text='Duration (sec)').grid(row=8,column=0)
        self.duration_entry = tk.Entry(self.frame_soni, width=10, borderwidth=2, bg='black', fg='lime green', 
                                       font='Arial 15')
        self.duration_entry.insert(0,'0.1')
        self.duration_entry.grid(row=8,column=1,columnspan=1)
    
    def init_display_size(self):
        #aim --> match display frame size with that once the canvas is added
        #the idea is for consistent aestheticsTM
        self.fig = figure.Figure(figsize=(5,5))
        self.fig.subplots_adjust(left=0.06, right=0.94, top=0.94, bottom=0.06)

        self.ax = self.fig.add_subplot()
        self.im = self.ax.imshow(np.zeros(100).reshape(10,10))
        self.ax.set_title('Click "Browse Image" to the right to begin!',fontsize=15)
        self.text = self.ax.text(x=2.2,y=4.8,s='Your Galaxy \n Goes Here',color='red',fontsize=25)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_display) 
        
        #activate the draw square/rectangle/quadrilateral/four-sided polygon event
        self.connect_event=self.canvas.mpl_connect('button_press_event',self.drawSqRec)
        
        #add canvas 'frame'
        self.label = self.canvas.get_tk_widget()
        self.label.grid(row=0,column=0,columnspan=3,rowspan=6,sticky='nsew')
    
    def add_info_button(self):
        self.info_button = tk.Button(self.frame_display, text='Galaxy FITS Info', padx=15, pady=10, font='Ariel 20', command=self.popup_info)
        self.info_button.grid(row=8,column=0,sticky='w',rowspan=2)
    
    def add_save_button(self):
        self.save_button = tk.Button(self.frame_save, text='Save as WAV', padx=15, pady=10, font='Ariel 20',
                                     command=self.save_sound)
        self.save_button.grid(row=0,column=0)
    
    def add_saveani_button(self):
        self.saveani_button = tk.Button(self.frame_save, text='Save as MP4', padx=15, pady=10, font='Ariel 20', command=self.create_midi_animation)
        self.saveani_button.grid(row=0,column=1)
    
    def add_w1w3merge_box(self):
        self.var_w1w3 = tk.IntVar()
        self.w1w3merge_box = tk.Checkbutton(self.frame_save, text='Create Overlay MP3 (W1 & W3 only)', 
                                            font='Ariel 20', onvalue=1, offvalue=0, variable=self.var_w1w3)
        self.w1w3merge_box.grid(row=1,column=0,sticky='nsew',columnspan=2)
    
    def add_browse_button(self):
        self.button_explore = tk.Button(self.frame_buttons, text="Browse Image", padx=10, pady=5, 
                                        font=self.helv20, command=self.browseFiles)
        self.button_explore.grid(row=1,column=0)
        
    def add_browse_mask_button(self):
        self.button_mask_explore = tk.Button(self.frame_buttons, text="Browse Mask", padx=10, pady=5,
                                             font=self.helv20, command=self.browseFilesMask)
        self.button_mask_explore.grid(row=1,column=1)
        
    def add_enter_button(self):
        self.path_button = tk.Button(self.frame_buttons, text='Load/Refresh Canvas', padx=15, pady=10, font=self.helv20,bg='gray',command=self.initiate_canvas)
        self.path_button.grid(row=2,column=0,columnspan=2)
    
    def add_midi_button(self):
        self.midi_button = tk.Button(self.frame_soni, text='Sonify', padx=20, pady=10, font=self.helv20, 
                                     command=self.midi_setup_bar)
        self.midi_button.grid(row=9,column=0,columnspan=2)
    
    def add_angle_buttons(self):
        self.angle_button = tk.Button(self.frame_box, text='Rotate',padx=5,pady=10,font=self.helv20,
                                      command=self.create_rectangle)
        self.angle_button.grid(row=2,column=1,columnspan=3)
        self.incarrow = tk.Button(self.frame_box, text='+1',padx=1,pady=10,font='Ariel 14',
                                  command=self.increment)
        self.incarrow.grid(row=2,column=4,columnspan=1)                          
        self.decarrow = tk.Button(self.frame_box, text='-1',padx=1,pady=10,font='Ariel 14',
                                  command=self.decrement)
        self.decarrow.grid(row=2,column=0,columnspan=1)
    
    def change_angle(self, delta):
        self.angle = float(self.angle_box.get()) #grab current angle
        self.angle += delta #edit angle
        
        self.angle_box.delete(0, tk.END) #delete current textbox entry
        self.angle_box.insert(0, str(self.angle)) #update entry with incremented angle
        
        #automatically rotate the rectangle when + is clicked
        self.create_rectangle()
    
    def increment(self):
        self.change_angle(5)  #+/- 1 is too small
    
    def decrement(self):
        self.change_angle(-5) #+/- 1 is too small
    
    def parse_angle(self):
        try:
            angle = float(self.angle_box.get())
            #if the angle is no different from 0 (e.g., 180, 360, etc.), just set the angle = 0.
            if (angle / 90) % 2 == 0:
                angle = 0
            #if angle is 90, 270, etc., just approximate as 89.9 deg (avoids many problems -- including dividing by cos(90)=0 -- and 89.9 is sufficiently close to 90 degrees)
            elif (angle / 90) % 2 == 1:
                angle = 89.9
        except:
            angle = 0
            
            self.angle_box.delete(0, tk.END)
            self.angle_box.insert(0, str(angle))
        
        self.angle = angle
        return angle
    
    def remove_unrotated_rectangle(self):
        try:
            for line in [self.line_one,self.line_two,self.line_three,self.line_four]:
                line_to_remove = line.pop(0)
                line_to_remove.remove()
        except:
            pass
    
    def remove_rotated_rectangle(self):
        try:
            for line in [self.line_eins,self.line_zwei,self.line_drei,self.line_vier]:
                line_to_remove = line.pop(0)
                line_to_remove.remove()
        except:
            pass
    
    def update_rectangle_bounds(self):
        self.xmin, self.xmax, self.ymin, self.ymax = get_rectangle_bounds(self.event_bounds, self.angle, self.im_length,
            one_rot=getattr(self, 'one_rot', None), two_rot=getattr(self, 'two_rot', None),
            three_rot=getattr(self, 'three_rot', None), four_rot=getattr(self, 'four_rot', None))
    
    def update_rotated_coordinates(self):

        xym_dict = get_xym(self.event_bounds, self.angle)

        self.x_rot = xym_dict['x_rot']
        self.y_rot = xym_dict['y_rot']
        self.m_rot = xym_dict['m_rot']

        self.one_rot = xym_dict['one_rot']
        self.two_rot = xym_dict['two_rot']
        self.three_rot = xym_dict['three_rot']
        self.four_rot = xym_dict['four_rot']

        self.n_spaces = xym_dict['n_spaces']
    
    def sample_rectangle(self):
        if self.angle == 0:
            rec_dict = sample_vertical_rectangle(self.dat, self.xmin, self.xmax, self.ymin, self.ymax, 
                                                 image_alt=self.dat_alt if int(self.var_w1w3.get())>0 else None)
            self.mean_list_norot = rec_dict['mean_list']
        else:
            if int(self.var_w1w3.get())>0:   #if == 0, then self.dat_alt does not exist!
                rec_dict = sample_rotated_rectangle(self.dat, self.event_bounds, self.angle, self.dat_alt)
            else:
                rec_dict = sample_rotated_rectangle(self.dat, self.event_bounds, self.angle)
        return rec_dict
    
    def handle_first_rectangle_click(self, event):
        #reset the angle!
        self.angle = 0
        self.angle_box.delete(0,tk.END)
        self.angle_box.insert(0,str(self.angle))

        #if the corner is within the canvas, plot a dot to mark this 'first' corner
        if event.inaxes:
            self.bound_check=True
            dot = self.ax.scatter(self.x1,self.y1,color='crimson',s=10,marker='*')
            self.sq_mean_value = self.dat[int(self.x1),int(self.y1)]
            self.canvas.draw()
            #for whatever reason, placing dot.remove() here will delete the dot after the second click
            dot.remove()
    
    def handle_second_rectangle_click(self, event):
        #assign all event coordinates to an array
        self.event_bounds = [self.x1.copy(),self.y1.copy(),self.x2.copy(),self.y2.copy()]

        if event.inaxes:
            if (self.bound_check):
                self.create_rectangle(x_one=self.x1,x_two=self.x2,y_one=self.y1,y_two=self.y2)
                self.canvas.draw()

        #reset parameters for next iteration
        self.bound_check = None
        self.x1=None
        self.x2=None
        self.y1=None
        self.y2=None

        #similar phenomenon as dot.remove() above.
        try:
            self.remove_unrotated_rectangle()
        except:
            pass
    
    def draw_unrotated_rectangle(self, x1, x2, y1, y2):
        self.line_one = self.ax.plot([x1, x1], [y1, y2], color='crimson', linewidth=2)
        self.line_two = self.ax.plot([x1, x2], [y1, y1], color='crimson', linewidth=2)
        self.line_three = self.ax.plot([x2, x2], [y1, y2], color='crimson', linewidth=2)
        self.line_four = self.ax.plot([x1, x2], [y2, y2], color='crimson', linewidth=2)
    
    def draw_rotated_rectangle(self):

        x1, x2, x3, x4 = (self.one_rot[0], self.two_rot[0], self.three_rot[0], self.four_rot[0])
        y1, y2, y3, y4 = (self.one_rot[1], self.two_rot[1], self.three_rot[1], self.four_rot[1])
        
        self.line_eins = self.ax.plot([x1, x3], [y1, y3], color='crimson', linewidth=2)
        self.line_zwei = self.ax.plot([x1, x4], [y1, y4], color='crimson', linewidth=2)
        self.line_drei = self.ax.plot([x2, x3], [y2, y3], color='crimson', linewidth=2)
        self.line_vier = self.ax.plot([x2, x4], [y2, y4], color='crimson', linewidth=2)

        self.canvas.draw()
    
    def update_sampled_data(self, rec_dict):
        self.mean_list = rec_dict['mean_list']
        self.all_line_coords = rec_dict['all_line_coords']
    
    def update_xminmax_display(self):
        self.xminmax_entry.delete(0,tk.END)
        mean_px_min = '{:.2f}'.format(self.xmin)
        mean_px_max = '{:.2f}'.format(self.xmax)
        self.xminmax_entry.insert(0,f'{mean_px_min}, {mean_px_max}')
    
    def get_note_names(self):
        selected_sig = self.keyvar.get()
        print(selected_sig)
        return NOTE_DICT[selected_sig].split("-")  #converts into a proper list of note strings
    
    def load_overlay_data(self):
        overlay_dict = load_overlay_image(self.path_to_im.get(), self.band, self.mask_bool)
        self.band_alt = overlay_dict['band_alt']
        self.dat_alt = overlay_dict['dat_alt']
    
    def load_sonification_settings(self):
        #define various quantities required for midi file generation
        self.y_scale = float(self.y_scale_entry.get())
        self.strips_per_beat = 10
        self.vel_min = int(self.vel_min_entry.get())
        self.vel_max = int(self.vel_max_entry.get())
        self.bpm = int(self.bpm_entry.get())
        self.program = int(self.program_entry.get())   #the instrument!
        self.duration = float(self.duration_entry.get())
    
    def build_sonification_data(self, mean_strip_values_alt=None):

        self.t_data = build_time_data(self.mean_list, self.strips_per_beat)
        self.midi_data = build_midi_data(self.mean_list, self.note_names,self.y_scale, 
                                         reverse=(int(self.var_rev.get()) == 1))
        self.vel_data = build_velocity_data(self.mean_list, self.vel_min, self.vel_max, self.y_scale)

        if mean_strip_values_alt is not None:
            self.midi_data_alt = build_midi_data(mean_strip_values_alt, self.note_names, self.y_scale,
                                                 reverse=(int(self.var_rev.get()) == 1))
            self.vel_data_alt = build_velocity_data(mean_strip_values_alt, self.vel_min, self.vel_max, self.y_scale)
    
    def save_sound(self):
        
        #if self.memfile has been defined already, then save as .wav
        #notes: -file will automatically go to 'saved_wavfiles' directory
        #       -.wav will only save the most recent self.midi_file, meaning the user must click "Sonify" to 
                 #sonify their rectangle/parameter tweaks so that they might be reflected in the .wav
        
        if hasattr(self, 'midi_file'):
            
            midi_savename = self.path_to_repos+'saved_wavfiles/'+str(self.galaxy_name)+'-'+str(self.band)+'.mid'   #using our current file conventions to define self.galaxy_name (see relevant line for further details); will save file to saved_wavfile directory
            
            #write file
            with open(midi_savename,"wb") as f:
                self.midi_file.writeFile(f)
            
            wav_savename = self.path_to_repos+'saved_wavfiles/'+str(self.galaxy_name)+'-'+str(self.band)+'.wav'   
            
            #initiate FluidSynth class!
            #gain governs the volume of wavefile. I needed to tweak the source code of midi2audio to 
            #have the gain argument --> I'll give instructions somewhere for how to do so...
            #check my github wiki. :-)
            fs = FluidSynth(sound_font=self.soundfont, gain=3)   
            
            if os.path.isfile(wav_savename):    
                self.namecounter+=1
                wav_savename = self.path_to_repos+'saved_wavfiles/'+str(self.galaxy_name)+'-'+str(self.band)+'-'+str(self.namecounter)+'.wav'                
            else:
                self.namecounter=0
            
            fs.midi_to_audio(midi_savename, wav_savename) 
            
            self.download_success()   #play the jingle
            
            self.time = self.get_wav_length(wav_savename)   #length of soundfile
            
            self.wav_savename = wav_savename   #need for creating .mp4
            
        #if user has not yet clicked "Sonify", then clicking button will activate a popup message
        else:
            self.textbox = 'Do not try to save an empty .wav file! Create a rectangle on the image canvas then click "Sonify" to generate MIDI notes.'
            self.popup()

    def remove_current_bar(self):
        try:
            self.current_bar.remove()
        except:
            pass
    
    def place_vertical_bar(self):
        #if x is within the rectangle bounds, all is well. 
        if (self.x<=self.xmax) & (self.x>=self.xmin):
            pass
        else:
            #if x is beyond the right side of the rectangle, line will be placed at rightmost end
            if (self.x>=self.xmax):
                self.x = self.xmax

            #if x is beyond the left side of the rectangle, line will be placed at leftmost end
            if (self.x<=self.xmin):
                self.x = self.xmin

        n_pixels = int(self.ymax-self.ymin)   #number of pixels between ymin and ymax
        line_x = np.zeros(n_pixels)+int(self.x)
        line_y = np.linspace(self.ymin,self.ymax,n_pixels)       
        self.current_bar, = self.ax.plot(line_x,line_y,linewidth=3,color='red')

        #extract the mean pixel value from this bar
        value_list = np.zeros(n_pixels)
        for index in range(n_pixels):
            y_coord = line_y[index]
            px_value = self.dat[int(y_coord)][int(self.x)]   #x will be the same...again, by design.
            value_list[index] = px_value
        self.mean_px = '{:.2f}'.format(np.mean(value_list[value_list!=0.]))
        self.val.config(text=f'Mean Pixel Value: {self.mean_px}',font='Ariel 18')
        self.canvas.draw()
        
    def place_rotated_bar(self):
        closest_line_index = find_closest_bar(self.all_line_coords, self.x, self.y)
                
        line_mean = self.mean_list[closest_line_index]
        line_coords = self.all_line_coords[closest_line_index]

        line_xvals = np.asarray(line_coords)[:,0]
        line_yvals = np.asarray(line_coords)[:,1]

        self.current_bar, = self.ax.plot([line_xvals[0],line_xvals[-1]],[line_yvals[0],line_yvals[-1]],
                                        linewidth=3,color='red')

        #extract the mean pixel value from this bar
        self.mean_px = '{:.2f}'.format(line_mean)

        self.val.config(text=f'Mean Pixel Value: {self.mean_px}',font='Ariel 16')
        self.canvas.draw()
    
    def load_image_data(self):
        self.dat_for_display, self.dat_header = load_fits(self.path_to_im.get())
        self.galaxy_name, self.band = get_galaxy_info(self.path_to_im.get())
        
    def load_mask_data(self):
        try:
            self.mask_bool = load_mask(self.path_to_mask.get())
        except:
            self.mask_bool = create_default_mask(self.dat_for_display.shape)  
            print('Mask image not found or not same dimensions as image; proceeding with default v1, v2, and normalization values.') 
            self.path_to_mask.delete(0,tk.END)
            self.path_to_mask.insert(0,'No Mask Found!')
    
    def get_display_normalization(self):
        v1 = scoreatpercentile(self.dat,0.5)
        v2 = scoreatpercentile(self.dat,99.9)
        norm_im = simple_norm(self.dat,'asinh', min_percent=0.5, max_percent=99.9,
                              min_cut=v1, max_cut=v2)  #'beautify' the image
        return v1, v2, norm_im
    
    def configure_display_sliders(self, v1, v2):
        self.v1slider.configure(from_=np.min(self.dat), to=np.max(self.dat))
        self.v2slider.configure(from_=np.min(self.dat), to=np.max(self.dat))
        
        #set the slider starting values
        self.v1slider.set(v1)
        self.v2slider.set(v2)
    
    def display_image(self, norm_im):
        self.ax = self.fig.add_subplot()
        if self.showmask.get()>0.:
            self.im = self.ax.imshow(self.dat,origin='lower',norm=norm_im)
        else:
            self.im = self.ax.imshow(self.dat_for_display,origin='lower',norm=norm_im)
        
        self.ax.set_xlim(0,len(self.dat)-1)
        self.ax.set_ylim(0,len(self.dat)-1)
        self.ax.set_title(f'{self.galaxy_name} ({self.band})',fontsize=15)
        
    def initialize_image_bounds(self):
        self.im_length = np.shape(self.dat)[0]
        self.ymin = int(self.im_length/2-(0.20*self.im_length))
        self.ymax = int(self.im_length/2+(0.20*self.im_length))
        self.x=self.im_length/2
        
    def initialize_current_bar(self):
        #initiate self.current_bar (just an invisible line...for now)
        self.current_bar, = self.ax.plot([self.im_length/2,self.im_length/2+1],
                                         [self.im_length/2,self.im_length/2+1],
                                         color='None')
    
    ##################
    # ACTUAL METHODS #
    ##################
    
    def initiate_canvas(self):
        
        self.stop_animation()
        
        #I need to add a try...except statement here, in case a user accidentally clicks "Load/Refresh" without loading a galaxy first. If they do so THEN try to successfully load a galaxy, the GUI will break.
        try:
            #delete any and all miscellany (galaxy image, squares, lines) from the canvas (created using 
            #self.init_display_size())
            self.label.delete('all')
            self.ax.remove()
        except:
            pass
        
        self.load_image_data()
        
        #many cutouts have pesky stars or artifacts which dominate the display of the image stretch. I grab the mask image for the galaxy and create a 'mask bool' of 0s and 1s, then multiply this by the image in order to dictate v1, v2, and the normalization *strictly* on the central galaxy pixel values. 
        self.load_mask_data()

        #apply mask to image...not that this will NOT affect the default display (which excludes the bool mask)
        self.dat = apply_mask(self.dat_for_display, self.mask_bool)
        
        #use masked image to find default display stretch parameters
        v1, v2, norm_im = self.get_display_normalization()
        
        self.configure_display_sliders(v1, v2)

        #if checkbox is already activated, display will default to including the mask
        self.display_image(norm_im)
        self.initialize_image_bounds()
        
        #initiate self.current_bar (just an invisible line...for now)
        self.initialize_current_bar()
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.frame_display)    
        
        #activate the draw square/rectangle/quadrilateral/four-sided polygon event
        self.connect_event=self.canvas.mpl_connect('button_press_event',self.drawSqRec)
        
        #add canvas 'frame'
        self.label = self.canvas.get_tk_widget()
        self.label.grid(row=0,column=0,columnspan=3,rowspan=6)
    
    def enable_rectangle_mode(self):
        self.canvas.mpl_disconnect(self.connect_event)
        if hasattr(self, "connect_event_midi"):
            self.canvas.mpl_disconnect(self.connect_event_midi)  #this is the event that plays audio when clicking
        self.connect_event = self.canvas.mpl_connect('button_press_event',self.drawSqRec)
    
    def enable_line_mode(self):
        self.canvas.mpl_disconnect(self.connect_event)
        self.connect_event = self.canvas.mpl_connect('button_press_event',self.placeBar)
        try:
            self.connect_event_midi = self.canvas.mpl_connect('button_press_event', self.midi_singlenote)
        except:
            pass
    
    #deletes the GUI line + sweep animation to prevent onslaught of error messages in the terminal window...
    def stop_animation(self):

        if hasattr(self, "line_anim"):
            try:
                self.line_anim.event_source.stop()
            except:
                pass
            try:
                del self.line_anim
            except:
                pass
        
        if hasattr(self, "l"):
            try:
                self.l.remove()
            except:
                pass
    
    def change_canvas_event(self):
        
        if int(self.var.get())==0:
            self.enable_rectangle_mode()
        if int(self.var.get())==1:
            self.enable_line_mode()

    #create command function to print info popup message
    def popup(self):
        messagebox.showinfo('Unconventional README.md',self.textbox)
    
    #how silly that I must create a separate popup function for FITS information. sigh.
    def popup_info(self):
        
        try:
            hdu1 = fits.open(str(self.path_to_im.get()))
            self.textbox_info = hdu1[0].header
            hdu1.close()
        except:
            self.textbox_info = 'No header information available.'
        
        popup = tk.Toplevel()     #creates a window on top of the parent frame
        
        vscroll = tk.Scrollbar(popup, orient=tk.VERTICAL)   #generates vertical scrollbar
        vscroll.pack(side=tk.RIGHT, fill=tk.Y)   #PACKS scrollbar -- because this is a new window, we are not bound to using the grid

        text = tk.Text(popup, wrap=None, yscrollcommand=vscroll.set)   #initiate textbox in window; adds vertical and horizontal scrollbars; wrap=None prevents lines from being cut off
        text.pack(expand=True, fill=tk.BOTH)
        
        vscroll.config(command=text.yview)   #not entirely sure what this entails
        
        text.insert(tk.END, self.textbox_info)
    
    def find_closest_mean(self,meanlist):
        
        #from https://stackoverflow.com/questions/12141150/from-list-of-integers-get-number-closest-to-a-given-value
        self.closest_mean_index = np.where(np.asarray(meanlist) == min(meanlist, key=lambda x:abs(x-float(self.mean_px))))[0][0]     
    
    def get_animation_filename(self):
        
        #note that self.namecounter_ani is initialized to be 0; reset for every galaxy
        base_animation_name = self.path_to_repos+'saved_mp4files/'+str(self.galaxy_name)+'-'+str(self.band)+'.mp4'
        ani_savename, self.namecounter_ani = unique_filename(base_animation_name, self.namecounter_ani)  
        
        base_merged_name = f'{self.path_to_repos}saved_mp4files/{self.galaxy_name}-{self.band}-concat.mp4'
        ani_both_savename, self.namecounter_ani_both = unique_filename(base_merged_name, self.namecounter_ani_both)
        
        return ani_savename, ani_both_savename
    
    def create_animation_figure(self):
        #dictionary which includes fig, spec, ax1, ax2,  point1a, and line2 (the sweeping line for self.band image)
        v1_2 = float(self.v1slider.get())
        v2_2 = float(self.v2slider.get())
        
        figure_dict = create_base_figure(self.dat, self.band, self.galaxy_name, self.t_data, 
                                         self.midi_data, self.vel_data, self.xmin, self.xmax, 
                                         self.ymin, self.ymax, v1=v1_2, v2=v2_2)
        return figure_dict
    
    def add_animation_overlay(self, ax1, ax3):

        #concatenate midi lists; find min and max values
        self.ymin_anim = int(np.min(self.midi_data+self.midi_data_alt))
        self.ymax_anim = int(np.max(self.midi_data+self.midi_data_alt))

        overlay_dict = add_overlay_subplot(ax1=ax1, ax3=ax3, dat_alt=self.dat_alt, 
                                           band_alt=self.band_alt, t_data=self.t_data, 
                                           midi_data_alt=self.midi_data_alt, vel_data_alt=self.vel_data_alt, 
                                           xmin=self.xmin, xmax=self.xmax, ymin=self.ymin, ymax=self.ymax)

        point1b = overlay_dict['point1b']
        l3 = overlay_dict['line3']

        return point1b, l3
    
    def build_animation(self, fig, point1a, point1b, l2, l3):
        if int(self.var_w1w3.get())>0:
            return build_overlay_animation(fig=fig, point1a=point1a, point1b=point1b, line2=l2, line3=l3,
                                                xvals_anim=self.xvals_anim, t_data=self.t_data, midi_data=self.midi_data,
                                                midi_data_alt=self.midi_data_alt, all_line_coords=self.all_line_coords)
        else:
            return build_single_band_animation(fig=fig, point1a=point1a, line2=l2, xvals_anim=self.xvals_anim,
                                                    t_data=self.t_data, midi_data=self.midi_data, 
                                                    all_line_coords=self.all_line_coords)
    
    def placeBar(self, event):  
        
        self.x=event.xdata
        self.y=event.ydata
        
        #remove current bar, if applicable
        self.remove_current_bar()
        
        #remove animation bar, if applicable
        self.stop_animation()
        
        #if user clicks outside the image bounds, then problem-o.
        if event.inaxes:
            
            #if no rotation of rectangle, just create some vertical bars.
            #else, create rotated bars
            if (self.angle == 0):
                self.place_vertical_bar()            
            else:
                self.place_rotated_bar()
                                                       
        else:
            print('Keep inside of the bounds of either the rectangle or the image!')
            self.val.config(text='Mean Pixel Value: None', font='Ariel 16')


    def create_rectangle(self,x_one=None,x_two=None,y_one=None,y_two=None):
        
        self.angle = self.parse_angle()
            
        self.remove_rotated_rectangle()  #if applicable
        
        if x_one is not None:    
            self.draw_unrotated_rectangle(x_one, x_two, y_one, y_two)
            return
        
        #a nonzero angle means the user has already created the unaltered rectangle
        #that is, the coordinates already exist in self.event_bounds (x1,x2,y1,y2)
        self.update_rotated_coordinates()
        self.draw_rotated_rectangle()            
            
    def drawSqRec(self, event):
        
        #remove animation line, if applicable
        self.stop_animation()
        
        #remove current bar, if applicable
        self.remove_current_bar()
        
        self.angle = self.parse_angle()
        
        #collect the x and y coordinates of the click event
        #if first click event already done, then just define x2, y2. otherwise, define x1, y1.
        if (self.x1 is not None) & (self.y1 is not None):
            self.x2 = event.xdata
            self.y2 = event.ydata
        else:
            self.x1 = event.xdata
            self.y1 = event.ydata
        
        #the user has clicked only the 'first' rectangle corner...
        if (self.x1 is not None) & (self.x2 is None):
            self.handle_first_rectangle_click(event)
        
        #if the 'first' corner is already set, then plot the rectangle and print the output mean pixel value
        #within this rectangle
        if (self.x2 is not None):
            self.handle_second_rectangle_click(event)

##########
#the sonification-specific functions...
##########
    
    def midi_setup_bar(self):
        
        #remove animation bar, if applicable
        self.stop_animation()
        
        self.load_sonification_settings()
        self.angle = self.parse_angle()
        self.note_names = self.get_note_names()
                
        self.update_rectangle_bounds()
        self.update_xminmax_display()
            
        if int(self.var_w1w3.get()) > 0:
            self.load_overlay_data()  #load either W1 or W3 alternate image data
        
        rec_dict = self.sample_rectangle()
        self.update_sampled_data(rec_dict)  #updates self.mean_list and self.all_line_coords
        
        mean_strip_values_alt = None    
        if int(self.var_w1w3.get()) > 0:
            mean_strip_values_alt = rec_dict['mean_list_alt']
        
        self.build_sonification_data(mean_strip_values_alt)
        
        self.midi_allnotes() 
        
    def midi_allnotes(self):
        
        self.create_rectangle()
                
        if int(self.var_w1w3.get())>0:
            
            self.midi_file = build_overlay_track_midi(self.midi_data, self.midi_data_alt, self.t_data, self.bpm, 
                                                 self.duration, self.program, compare_program=47, threshold=0.10)
        else:
            self.midi_file = build_single_track_midi(self.midi_data, self.vel_data, self.t_data, 
                                                self.bpm, self.duration, self.program)
                
        self.memfile = midi_to_memfile(self.midi_file)
        self.length_of_file = get_midi_length(self.midi_file)
        play_memfile(self.memfile)
        
        self.sweep_line()
        
        
    def midi_singlenote(self,event):
                
        #for the instance where there is no rotation
        if self.angle == 0:

            self.find_closest_mean(self.mean_list_norot)  #determine index at which the mean_list element    
                                                          #is closest to the current bar mean outputs 
                                                          #self.closest_mean_index
        else:
            self.find_closest_mean(self.mean_list)
            
        #extract the midi and velocity notes associated with that index. 
        single_pitch = self.midi_data[self.closest_mean_index]
        single_volume = self.vel_data[self.closest_mean_index]
        
        #isolate the one note corresponding to the click event, build MIDI file.
        midi_file = build_preview_midi(pitch=single_pitch, velocity=single_volume, bpm=self.bpm, program=self.program,
                                       duration=0.5)
                
        self.memfile = midi_to_memfile(midi_file)
        play_memfile(self.memfile)
    
    ###ANIMATION FUNCTIONS###
    
    #when file(s) are finished downloading, there will be a ding sound indicating completion. it's fun.
    def download_success(self):
        path = os.getcwd()+'/success.mp3'
        mixer.init()
        mixer.music.set_volume(0.25)
        mixer.music.load(path)
        mixer.music.play()
    
    def get_wav_length(self,file):
        wav_length = mixer.Sound(file).get_length()
        print(f'File Length (seconds): {mixer.Sound(file).get_length()}')
        return wav_length

    def sweep_line(self):
        
        #remove current bar, if applicable
        self.remove_current_bar()
        
        self.l, self.line_anim = create_gui_sweep_animation(fig=self.fig, ax=self.ax, xmin=self.xmin, ymin=self.ymin, 
                                                      xmax=self.xmax, ymax=self.ymax, length_of_file=self.length_of_file, 
                                                      duration=self.duration, t_data=self.t_data, midi_data=self.midi_data, 
                                                      all_line_coords=self.all_line_coords, update_func=update_gui_sweep)    
    def create_midi_animation(self):
        
        self.save_sound()

        #dictionary which includes fig, spec, ax1, ax2,  point1a, and line2 (the sweeping line for self.band image)
        figure_dict = self.create_animation_figure()
        
        #unpack dictionary...
        fig = figure_dict['fig']

        ax1 = figure_dict['ax1']
        ax3 = figure_dict['ax3']

        point1a = figure_dict['point1a']
        l2 = figure_dict['line2']
        
        self.ymin_anim = int(np.min(self.midi_data))
        self.ymax_anim = int(np.max(self.midi_data))
        
        #add overlay of W1 or W3 "midi_data_alt," if applicable
        if int(self.var_w1w3.get())>0:
            point1b, l3 = self.add_animation_overlay(ax1, ax3)
        else:
            point1b, l3 = None, None

        self.xmin_anim = 0
        self.xmax_anim = np.max(self.t_data)
            
        self.xvals_anim = np.arange(0,len(self.midi_data),1)
        
        line_anim = self.build_animation(fig, point1a, point1b, l2, l3)
             
        ani_savename, ani_both_savename = self.get_animation_filename()
        
        save_animation(line_anim=line_anim, output_path=ani_savename, n_frames=len(self.xvals_anim),
                       length_of_file=self.length_of_file)
        
        del fig     #I am finished with the figure, so I shall delete references to the figure.
        
        merge_audio_video(video_path=ani_savename, audio_path=self.wav_savename, 
                          output_path=ani_both_savename, overwrite=True)
            
        self.download_success()
        
        self.textbox = 'Done! Check the saved_mp4file directory for the final product.'
        self.popup()
        
if __name__ == "__main__":
    
    #parameter.txt file unpacking here

    if '-h' in sys.argv or '--help' in sys.argv:
        print("Usage: %s [-params (name of parameter.txt file, no single or double quotation marks)]")
        sys.exit(1)
        
    if '-params' in sys.argv:
        p = sys.argv.index('-params')
        param_file = str(sys.argv[p+1])
    
    #create dictionary with keyword and values from param textfile...
    param_dict = read_params(param_file)

    #now...extract parameters and assign to relevantly-named variables
    path_to_ffmpeg = param_dict['path_to_ffmpeg']
    path_to_repos = param_dict['path_to_repos']
    initial_browsedir = param_dict['initial_browsedir']
    soundfont = param_dict['soundfont']
    window_geometry = param_dict['window_geometry']
    
    matplotlib.rcParams['animation.ffmpeg_path'] = path_to_ffmpeg   #need for generating animations...
    
    app = App(path_to_repos, initial_browsedir, soundfont, window_geometry)
    app.mainloop()