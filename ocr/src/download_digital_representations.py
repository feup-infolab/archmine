import sys
import os
import pandas as pd
import requests

NAME_COLUMN = 'Name'
DISSEMINATION_URL_COLUMN = 'DisseminationURL'
DIGITARQ_URL = 'https://digitarq.arquivos.pt/Controls/vaultimage/?id='
DISSEMINATION_URL_PREFIX = 'vault://'

"""Download digital representation files from result of queries made to Digitarq that include filenames and dissemination URLs"""

try:
    src_path = sys.argv[1]
    dst_path = sys.argv[2]
except IndexError:
    print("Invalid command line arguments: 2 arguments are expected. Usage: python download_digital_representations.py <csv source path with dissemination urls> <dataset images dest path>")
    sys.exit()
data = pd.read_csv(src_path)
for idx, row in data.iterrows():
    disseminationURL = row[DISSEMINATION_URL_COLUMN].replace(DISSEMINATION_URL_PREFIX, '')
    url = DIGITARQ_URL + disseminationURL
    r = requests.get(url, allow_redirects=True)
    filename = dst_path + row[NAME_COLUMN]
    if not os.path.exists(dst_path):
        os.makedirs(dst_path)
    file = open(filename, 'wb')
    file.write(r.content)
    file.close()