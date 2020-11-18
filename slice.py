# -*- coding: utf-8 -*-
"""
Created on Tue Nov 17 13:50:20 2020

@author: MaXentric
"""

import io
import os
import numpy as np

i 

#import MIDI.py as MIDI
#os.system("MIDI.py")

slices = []

midi_paths = ["waldstein_3.mid"]

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

numTracks = len(score[2][:])

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

# Slice by quarter
max_quarter = np.ceil(np.max(x_score))
t_slice = np.repeat(np.reshape(np.arange(0,max_quarter,1),(1,-1)),2,axis=0) # slice one quarter note at a time
y_slice = np.repeat(np.reshape([np.min(y_score),np.max(y_score)],(-1,1)),np.size(t_slice,1),axis=1)

numSlices = np.size(t_slice,1)
music_vecs = [] # hold note, duration, residual flag
s_idx = 0 # slice index
#for t_ in t_slice[0,:]:
#    if t_ 
    

plt.figure
plt.plot(x_score,y_score,'k')
plt.plot(np.repeat(t_slice,2,1))
plt.plot(t_slice,y_slice,'r')
plt.ylabel('Note Pitch (60 = C4)')
plt.xlabel('t in quarter notes')
plt.title('Waldstein 3 MIDI score')




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