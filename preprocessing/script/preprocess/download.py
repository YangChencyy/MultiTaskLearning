import subprocess
import os
import csv

with open('script/preprocess/links.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=' ', quotechar='|')
    for row in spamreader:
        # subprocess.run(["gdown", "--fuzzy", "'{}'".format(row[0])])
        print("gdown --fuzzy '{}'".format(row[0]))
        os.system("gdown --fuzzy '{}'".format(row[0]))
