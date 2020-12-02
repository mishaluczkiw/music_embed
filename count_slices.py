import os
from zipfile import ZipFile
from slice import read_midi_file, slice_midi, plot_slices_bar
import pickle
import sys
import pandas as pd
import numpy as np
import ast

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

# append slices to pickle file for all midi files in the given range
def generate_slices(start, end, file_prefix):
    i_bad_midis = [3403, 7826, 20202, 25152, 40659, 40715, 58949, 63606, 67136, 99019, 99034, 106954, 112028, 118286, 122886] # add 112051?
    for i in range(start, end):
        print('midi #%d of %d...' % (i + 1, len(midi_paths)), end='\r')
        # print('midi #%d of %d...' % (i + 1, len(midi_paths)))
        file = midi_paths[i]
        if i not in i_bad_midis:
            try:
                x_score, y_score, durations = read_midi_file('\\\\?\\' + file, get_durations=True)
                slices = slice_midi(x_score, y_score, durations)
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
    with open(file_in, "rb") as f:
        while 1:
            print(iii,end='\r')
            try:
                all_slices.extend(pickle.load(f))
            except EOFError:
                break
            iii+=1

    all_slices = [str(s) for s in all_slices]
    counts = pd.Series(all_slices).value_counts()
    with open(file_out, "wb") as f:
        pickle.dump(counts, f)
    return counts

# combine slices from all files
# filter all slices with at least threshold (10) occurrences
def combine_all_slice_counts(slice_count_files, threshold=10):
    slices = pd.Series()
    for i, file in enumerate(slice_count_files):
        print('loading file %d of %d' % (i+1, len(slice_count_files)))
        with open(file, "rb") as f:
            counts = pickle.load(f)
            counts = counts[counts >= threshold]
            slices = slices.add(counts, fill_value=0)

    # sort all slices in descending order and remove empty sets
    slices = slices[(slices.index != 'set()') & (slices.index != '[]')]
    slices.sort_values(ascending=False, inplace=True)
    print('%d total slices with at least %d occurences' % (len(slices), threshold))
    return slices


# folder = 'C:\\Users\\User\\Documents\\Deep Learning\\Final\\music_embed\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]'
# folder = 'C:\\Users\\MaXentric\\Desktop\\Misha\\Deep_Learning\\final\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]'

zips = unzip(folder, [])
midi_paths = get_midi_file_paths(folder)

batches = [(0, 15000), (15000, 30000), (30000, 45000), (45000, 60000),
           (60000, 75000), (75000, 90000), (90000, 105000), (105000, len(midi_paths))]
prefix = 'slices'
counts_prefix = 'slice_counts'
slice_count_files = []
for batch in batches:
    start, end = batch
    generate_slices(start=start, end=end, file_prefix=prefix)
    file_in = "%s_%d-%d" % (prefix, start, end)
    file_out = "%s_%d-%d" % (counts_prefix, start, end)
    slice_count_files.append(file_out)
    slice_counts = compile_slice_counts(start, file_in, file_out)

slices = combine_all_slice_counts(slice_count_files, threshold=10)
# write all slice counts to a single file
with open("all_slice_counts_gte_10", "wb") as f:
    pickle.dump(slices, f)

# to load the binary file, use this:
# with open("all_slice_counts_notes", "rb") as f:
#     all_slice_counts = pickle.load(f)

# # plot top 1000 slices and save html widget
# plot_slices_bar(slices[:1000], filename='slices_1000.html')


# filter out all slices with durations > threshold
def filter_durations(threshold=128):
    # max_durs = np.array([max(ast.literal_eval(s))[1] for s in slices.index])
    min_durs = np.array([min(ast.literal_eval(s))[1] for s in slices.index])
    # slices_good = slices[max_durs <= threshold]
    slices_good_min = slices[min_durs <= threshold]
    # return slices_good
    return slices_good_min

# # start, end = 0, 15000
# # start, end = 15000, 30000
# # start, end = 30000, 45000
# # start, end = 45000, 60000
# # start, end = 60000, 75000
# # start, end = 75000, 90000
# # start, end = 90000, 105000
# start, end = 105000, len(midi_paths)
# prefix = 'slices'
# # generate_slices(start=start, end=end, file_prefix=prefix)
#
#
# # ssss = generate_slices2(start=start, end=end, file_prefix=prefix) # 10:13
#
# file_in = "%s_%d-%d" % (prefix, start, end)
# file_out = "slice_counts_%d-%d" % (start, end)
# slice_counts = compile_slice_counts(start, file_in, file_out)
# # slice_counts = compile_slice_counts2(start, file_in, file_out) # 11:57 only at 1422 by 12:06, even slower but didnt get stuck
#
#
#
# slice_files = ['slice_counts_0-15000', 'slice_counts_15000-30000', 'slice_counts_30000-45000',
#                'slice_counts_45000-60000', 'slice_counts_60000-75000', 'slice_counts_75000-90000',
#                'slice_counts_90000-105000', 'slice_counts_105000-127422']
# slices = combine_all_slice_counts(slice_files, threshold=10)
# #
# # write all slice counts to a single file
# with open("all_slice_counts_gte_10", "wb") as f:
#     pickle.dump(slices, f)
#
#
#
#
#
# # diff = []
# # for iii in slices_good_min.index:
# #     if max(ast.literal_eval(iii))[1] > 128:
# #         diff.append(iii)
# # diff_min = []
# # for iii in diff:
# #     diff.append()
# # diff_min = [min(ast.literal_eval(iii))[1] for iii in diff]
# # 1466 slices have a mix of normal durations and durations > 128
#
#
# # #  check for which midis have weird slice (with very large duration)
# # # weird_slice = [(41, 26995.75)]
# # i_weird = []
# # i_bad_midis = [3403, 7826, 20202, 25152, 40659, 40715, 58949, 63606, 67136, 99019, 99034, 106954, 112028, 118286,
# #                122886]
# # i_bad_midis.extend([2547])
# # for i in range(0, 100):
# #     print('midi #%d of %d...' % (i + 1, len(midi_paths)), end='\r')
# #     file = midi_paths[i]
# #     if i not in i_bad_midis:
# #         try:
# #             x_score, y_score, durations = read_midi_file('\\\\?\\' + file, get_durations=True)
# #             slices = slice_midi(x_score, y_score, durations)
# #             durs = np.unique([note[1] for slice in slices for note in slice])
# #             if any(durs > 64):
# #                 i_weird.append(i)
# #                 print(i, file)
# #                 with open("weird_midis.txt", "a", encoding='utf-8') as f:
# #                     f.write('{} {}\n'.format(i, file))
# #             # if weird_slice in slices:
# #             #     print('this file', i)
# #             #     print('this file', i)
# #             #     print(file)
# #             #     break
# #             # with open("%s_%d-%d" % (file_prefix, start, end), "ab") as f:
# #             #     pickle.dump(slices, f)
# #         except:
# #             # with open("bad_midis.txt", "a", encoding='utf-8') as f:
# #             #     f.write('{}\n'.format(file))
# #             pass
# #     else:
# #         # with open("bad_midis.txt", "a", encoding='utf-8') as f:
# #         #     f.write('{}\n'.format(file))
# #         pass
# #
# # with open('slice_counts_0-15000', "rb") as f:
# #     counts = pickle.load(f)
# #
# # """
# # index 2547 contains all 26994 occurences of [(41, 26995.75)]
# # C:\Users\User\Documents\Deep Learning\Final\music_embed\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\130000_Pop_Rock_Classical_Videogame_EDM_MIDI_Archive[6_19_15]\9\92502_08.MID
# #     - silence for like 2 min straight, crashes media player at the end and midi editor when trying to open
# # """
# #
# # # compile list of all notes w/ weird durations, > 16
# # weird_slices = []
