# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:50:20 2020

@author: MaXentric
"""

import io
import os
import numpy as np
import matplotlib.pyplot as plt
from MIDI import midi2score
import plotly.graph_objects as go
# i

#import MIDI.py as MIDI
#os.system("MIDI.py")

slices = []

midi_paths = ["waldstein_3.mid"]
file = midi_paths[0]
def read_midi_file(file, get_durations=False):
    midi = open(file,'rb').read()
    score = midi2score(midi)
    '''
    The argument is a list: the first item in the list is the "ticks"
    parameter, the others are the tracks. Each track is a list
    of score-events, and each event is itself a list.  A score-event
    is similar to an opus-event (see above), except that in a score:
     1) the times are expressed as an absolute number of ticks
        from the track's start time
     2) the pairs of 'note_on' and 'note_off' events in an "opus"
        are abstracted into a single 'note' event in a "score":
        ['note', start_time, duration, channel, pitch, velocity]
    score2opus() returns a list specifying the equivalent "opus".
    
    my_score = [
        96,
        [   # track 0:
            ['patch_change', 0, 1, 8],
            ['note', 5, 96, 1, 25, 96],
            ['note', 101, 96, 1, 29, 96]
        ],   # end of track 0
    ]
    my_opus = score2opus(my_score)
    '''

    # numTracks = len(score[2][:])

    ticks = score[0]

    #GOING THROUGH A SCORE WITHIN A PYTHON PROGRAM
    channels = {2,3,5,8,13}
    itrack = 1   # skip 1st element which is ticks
    all_notes = []
    all_durations = []
    all_starts = []
    while itrack < len(score):
        for event in score[itrack]:
            if event[0] == 'note':   # for example,
                all_notes.append(event[4])
                all_starts.append(event[1])
                all_durations.append(event[2])# do something to all notes
            # or, to work on events in only particular channels...

        itrack += 1

    all_quarters = np.array(all_durations)/ticks

    x_score = (np.array([np.array(all_starts)/ticks,np.array(all_starts)/ticks+all_quarters]))
    y_score = np.transpose(np.repeat(np.reshape(np.array(all_notes),(-1,1)),2,axis=1))

    if get_durations:
        durations = np.where(all_quarters >= 1 / 16, np.round(all_quarters * 16) / 16, all_quarters)
        durations = np.transpose(np.repeat(np.reshape(np.array(durations), (-1, 1)), 2, axis=1))
        return x_score, y_score, durations
    else:
        return x_score, y_score

def plot_midi(x_score, y_score):
    # Slice by quarter
    max_quarter = np.ceil(np.max(x_score))
    t_slice = np.repeat(np.reshape(np.arange(0,max_quarter,1),(1,-1)),2,axis=0) # slice one quarter note at a time
    y_slice = np.repeat(np.reshape([np.min(y_score),np.max(y_score)],(-1,1)),np.size(t_slice,1),axis=1)

    # numSlices = np.size(t_slice,1)
    # music_vecs = [] # hold note, duration, residual flag
    # s_idx = 0 # slice index
    #for t_ in t_slice[0,:]:
    #    if t_

    plt.figure
    plt.plot(x_score,y_score,'k')
    plt.plot(np.repeat(t_slice,2,1))
    plt.plot(t_slice,y_slice,'r')
    plt.ylabel('Note Pitch (60 = C4)')
    plt.xlabel('t in quarter notes')
    plt.title('Waldstein 3 MIDI score')

# determines if note occurs in the given slice
# def in_slice(note_start, note_end, slice_start, slice_end):
#     # return not (note_start > slice_end or note_end < slice_start)
#     return not (note_start > slice_end or note_end <= slice_start)
#
# x_score: [[note start times], [note end times]]
# each slice will be either just a note set or a ({note set}, {duration set}) pair
# slices0 = []
# for slice in range(int(max_quarter)):
#     note_set = set()
#     for inote in range(x_score.shape[1]):
#         if in_slice(x_score[0][inote], x_score[1][inote], slice, slice+1):
#             note_set.add(y_score[0][inote])
#     slices0.append(note_set)
# str_slices = [str(slice) for slice in slices]
# d = dict([(y,x+1) for x,y in enumerate(str_slices)])
# index_slices = [d[x] for x in str_slices]

# runs much faster than nested for loops with in_slice function
# slices contain list of notes and their durations in the rder they occur in
def slice_midi(x_score, y_score, durations):
    slices = []
    for slice_start in range(int(np.ceil(np.max(x_score)))):
        inote = ~((x_score[0] > slice_start+1) | (x_score[1] <= slice_start))
        # slices.append(set(y_score[0,inote]))

        y = y_score[0, inote]
        d = durations[0, inote]
    # CONTINUE HERE THEN CHANGE COUNT_SLICES TO BE CONSISTENT
        [(y[i], d[i]) for i in range(len(y))]
    # return slices

def slice_midi_note(x_score, y_score):
    slices = []
    for slice_start in range(int(np.ceil(np.max(x_score)))):
        inote = ~((x_score[0] > slice_start+1) | (x_score[1] <= (slice_start)))
        slices.append(set(y_score[0,inote]))
    return slices

def plot_slices_hist(slices):
    str_slices = [str(slice) for slice in slices]
    # d = dict([(y,x+1) for x,y in enumerate(str_slices)])

    fig = go.Figure(data=[go.Histogram(x=str_slices)])
    fig.update_xaxes(categoryorder='total descending')
    fig.show()

def plot_slices_bar(slices, filename=''):
    fig = go.Figure([go.Bar(x=slices.index, y=slices)])
    if filename != '':
        # filename needs to have .html extension
        fig.write_html(filename)
    fig.show()


"""
for file in midi_paths:
    print(file)
    midi = io.open(file, 'rb').read()
    score = midi2score(midi)
    
    
    
    for track in score [1:]:
        # can add filter condition so only notes are kept
        # either append to list or append to file
        
        slices.extend(track)
        with open("./music.txt", "a") as f:
            for event in track:
 
                f.write('{}\n'.format(event))


"""