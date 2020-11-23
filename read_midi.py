import os
from zipfile import ZipFile
import io
from slice import read_midi_file, slice_midi_note, plot_slices_bar
import pickle
import sys
import pandas as pd
# import pymongo

def unzip(path, zips):
    for root, dirs, files in os.walk(path):
        for file in files:
            file_name = os.path.join(root, file)
            if file_name.endswith('.zip'):
                currentdir = os.path.join(root, file_name[:-4])
                currentdir = '\\\\?\\' + currentdir
                zips.append(currentdir)
                if not os.path.exists(currentdir):
                    os.makedirs(currentdir)
                with ZipFile('\\\\?\\' + file_name) as zipObj:
                    zipObj.extractall(currentdir)
                os.remove('\\\\?\\' + file_name)
                zips = unzip(currentdir, zips)
    return zips

def get_midi_file_paths(folder):
    midis = []
    num_mids = 0
    num_Mids = 0
    num_MIDs = 0
    num_midis = 0
    num_MIDis = 0
    noOfFiles = 0
    noOfMidis = 0
    for base, dirs, files in os.walk(folder):
        for file in files:
            noOfFiles += 1
            if file.endswith('.mid'):
                midis.append(os.path.join(base, file))
                noOfMidis += 1
                num_mids += 1
            elif file.endswith('.Mid'):
                midis.append(os.path.join(base, file))
                num_Mids += 1
            elif file.endswith('.MID'):
                midis.append(os.path.join(base, file))
                num_MIDs += 1
            elif file.endswith('.midi'):
                midis.append(os.path.join(base, file))
                num_midis += 1
            elif file.endswith('.MIDi'):
                midis.append(os.path.join(base, file))
                num_MIDis += 1
    print(noOfFiles)
    print(noOfMidis)
    print('.mid: %d\t .Mid: %d\t .MID: %d\t .midi: %d\t .MIDi: %d' % (num_mids, num_Mids, num_MIDs, num_midis, num_MIDis))
    print('total MIDI:', len(midis))
    return midis


folder = 'C:\\Users\\User\\Documents\\Deep Learning\\Final\\music_embed\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]'
# folder = 'C:\\Users\\MaXentric\\Desktop\\Misha\\Deep_Learning\\final\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]'

zips = unzip(folder, [])
midi_paths = get_midi_file_paths(folder)


# append slices to pickle file for all midi files in the given range
def get_slices(start, end, file_prefix):
    i_bad_midis = [3403, 7826, 25152, 40659, 40715, 58949, 63606, 67136, 99019, 99034, 106954, 112028, 118286, 122886]
    for i in range(start, end):
    # for i in range(122886+1, end):
        # print('midi #%d of %d...' % (start + i + 1, len(midi_paths)), end='\r')
        # # print('midi #%d of %d...' % (start + i + 1, len(midi_paths)))
        print('midi #%d of %d...' % (i + 1, len(midi_paths)), end='\r')
        file = midi_paths[i]
        if i not in i_bad_midis:
            try:
                x_score, y_score = read_midi_file('\\\\?\\' + file)
                slices = slice_midi_note(x_score, y_score)
                with open("%s_%d-%d" % (file_prefix, start, end), "ab") as f:
                    pickle.dump(slices, f)
            except:
                with open("bad_midis.txt", "a", encoding='utf-8') as f:
                    f.write('{}\n'.format(file))
        else:
            with open("bad_midis.txt", "a", encoding='utf-8') as f:
                f.write('{}\n'.format(file))

# get all slice counts for the given slice file and write counts to another file
def compile_slice_counts(start, file_in, file_out):

    all_slices = []
    iii=start
    # iii=90000
    # threshold = 117976
    # iii=threshold
    # with open("slices_%d-%d" % (start, end), "rb") as f:
    with open(file_in, "rb") as f:
        while 1:
        # while iii<threshold:
            print(iii,end='\r')
            try:
                all_slices.extend(pickle.load(f))
                # s = pickle.load(f)
                # if iii >= threshold:
                #     all_slices.extend(s)
                # print(sys.getsizeof(all_slices), end='\r')
            except EOFError:
                break
            iii+=1

    all_slices = [str(s) for s in all_slices]
    counts = pd.Series(all_slices).value_counts()
    # with open("slice_counts_%d-%d" % (start, end), "wb") as f:
    with open(file_out, "wb") as f:
    # with open("slice_counts_%d-%d" % (start, threshold), "wb") as f:
    # with open("slice_counts_%d-%d" % (threshold, end), "wb") as f:
        pickle.dump(counts, f)


# combine slices from all files
# keep the top 500 slices?
# filter all slices at least threshold (10) occurrences
def compile_all_slice_counts(slice_files, threshold=10):
    slices = pd.Series()
    for i, file in enumerate(slice_files):
        print('loading file %d of %d' % (i+1, len(slice_files)))
        with open('slices_notes/' + file, "rb") as f:
            counts = pickle.load(f)
            counts = counts[counts >= threshold]
            slices = slices.add(counts, fill_value=0)

    # sort all slices in descending order
    # remove empty set
    slices = slices[slices.index != 'set()']
    slices.sort_values(ascending=False, inplace=True)
    return slices


# start, end = 0, 30000
# start, end = 30000, 60000
# start, end = 60000, 90000
start, end = 90000, len(midi_paths)
# start, end = 67137, 90000
prefix='slicesasdfasdsa'
get_slices(start=start, end=end, file_prefix=prefix)

file_in = "%s_%d-%d" % (prefix, start, end)
file_out = "slice_counts_%d-%d" % (start, end)
compile_slice_counts(start, file_in, file_out)


slice_files = ['slice_counts_0-30000', 'slice_counts_30000-60000', 'slice_counts_60000-90000', 'slice_counts_90000-127422']
slices = compile_all_slice_counts(slice_files, threshold=10)
# write all slice counts to a single file
with open("all_slice_counts_notes", "wb") as f:
    pickle.dump(slices, f)

# to load the binary file, use this:
# with open("all_slice_counts_notes", "rb") as f:
#     all_slice_counts = pickle.load(f)

# plot top 1000 slices
plot_slices_bar(slices[:1000])
# plt.bar(range(len(slices)), height=slices)
# plot_slices_hist(all_slices)

# # checks for beginning and end of all_slices
# file = midi_paths[start]
# x_score, y_score = read_midi_file('\\\\?\\' + file)
# slices_start = slice_midi_note(x_score, y_score)
# slices_start[:5]
# file = midi_paths[end-1]
# file = midi_paths[112287]
# x_score, y_score = read_midi_file('\\\\?\\' + file)
# slices_end = slice_midi_note(x_score, y_score)
# slices_end[-5:-1]




"""
midi files that cause issues when reading;
\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\A\\A\\abba-voulez-vous.mid'
    - index 3403 in midi_paths
    - gets stuck in reading the midi, never finishes
    - crashes media player when playing end of the midi

indexes that get hung up: possibly bc files are either too large or have nothing to play
7826, 25152, 40659, 40715 (nothing plays), 58949, 63606, 67136, 99019, 99034, check: 106954, 112028, 118286, 122886

when loading slices_90000-127422:
98905, 101944, 111785, 117976 (takes longest),
also 110637?, 112287?, 126881?
starting at 90000, last index is 127014

"""


"""
zips = 
['\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Beethoven\\Complete Sonata in A flat major Op.110',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Beethoven\\Complete Tempest Sonata',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Beethoven\\Piano and Cello Sonata No.2 Op 5',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Beethoven\\Piano Sonata No.18',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Beethoven\\Piano Sonata No.23',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Beethoven\\Sonata No.1 in D',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Busoni\\Chacona in D, with variations',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Busoni\\Piano Concerto, Mov.1. A beautiful piece',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Busoni\\Piano Concerto, Mov.3. A wonderful piece',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Busoni\\Piano Concerto, Mov.4. Another wonderful piece',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Busoni\\Piano Concerto, Mov.5',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Chabrier\\España, rhapsody for orchestra (1883)',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Chabrier\\Joyeuse marche',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Chabrier\\Souvenirs de Munich',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Chabrier\\Suite pastorale 2 Danse villageoise',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Chabrier\\Suite Pastorale 4 Scherzo-valse',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Copland\\Appalacian Spring',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Copland\\Concerto for Clarinet',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Dvorak\\Slavonic Dance No. 8 Op. 46',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Dvorak\\Suite in A Major for Piano Op.98 (American Suite',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Frescobaldi\\Messa della Domenica (Orbis factor) from the Fiori Musicali',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Ginastera\\Danza Final from his ballet  Estancia',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Griffes\\Piano Sonata',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Mussorgski\\Sonata in C major',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Rachmaninov\\Paganini Variations No.9',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Rachmaninov\\Piano Concerto, 3rd mov',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Ravel\\La Valse',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Schubert\\Piano Sonata in B flat No.4, D960',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Classical Piano Midis\\Strauss, J\\The blue danube Op.314',
 '\\\\?\\C:\\Users\\User\\Documents\\Deep Learning\\Final\\7zip\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\Classical Archives - The Greats (MIDI)\\Scarlatti\\v5']
"""