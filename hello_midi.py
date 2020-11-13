# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 01:57:33 2020
Load MIDI file
@author: Misha Luczkiw & Nick Triano
"""

import mido as md

from mido import MidiFile, MetaMessage 
import os.path



folder = "beeth"

fn = "waldstein_3.mid"
fn1 = "waldstein_2.mid"
fn2 = os.path.join(folder,fn1)

print(fn2)


#f = open(fn)
#fn2 = "C:\Users\MaXentric\Desktop\Misha\Deep Learning\final\beeth"

song = md.MidiFile(fn2)
print(song)

for track in song.tracks:
    print(track)

song.play


