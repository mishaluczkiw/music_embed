import os
from zipfile import ZipFile
import io
from slice import read_midi_file, slice_midi, plot_slices

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

# this takes way too long to run to run for all midi files, not sure if it can completely fit in memory
# midis = []
# for midi in midi_paths:
#     midis.append(io.open('\\\\?\\' + midi, 'rb').read())

all_slices = []
start, end = 0, 100
for i, file in enumerate(midi_paths[start:end]):
    print('midi #%d of %d...' % (start+i+1, len(midi_paths)), end='\r')
    try:
        x_score, y_score = read_midi_file(file)
        slices = slice_midi(x_score, y_score)
        all_slices.extend(slices)
        # also option to write to file instead of storing in variable
        with open("slices_notes_%d-%d.txt" % (start, end), "a") as f:
            for slice in slices:
                f.writelines("%s\n" % slice for slice in slices)
    except:
        with open("bad_midis.txt", "a") as f:
            f.write('{}\n'.format(file))

plot_slices(all_slices)

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