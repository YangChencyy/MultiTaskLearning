import subprocess
import os
import csv
# import tar

paths = os.listdir('.')
tars = []
for path in paths:
    if '.tar' in path and path < '604_p.tar':
        tars.append(path)
for tar in tars:
    print(tar)
    subprocess.run(['tar','-xvf',tar])
