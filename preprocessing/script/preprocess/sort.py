import subprocess
import os
import csv
import glob
# import tar



# for path in glob.glob('*/*.wav'):
    # if path > '373_P/373_AUDIO.wav':
    #     continue
#     subprocess.run(["cp", path, 'audios/'])

# for path in glob.glob('*/features/*_OpenSMILE2.3.0_mfcc.csv'):
#     subprocess.run(["cp", path, 'OpenSMILE2.3.0_mfcc/'])

for path in glob.glob('*/*_Transcript.csv'):
    if path > '373_P/373_Transcript.csv':
        continue
    subprocess.run(["cp", path, 'Transcript/'])