"""
VoterLabs Inc.
URL Reader
 """
import argparse
import os
import filecmp
import requests

def download_csv(target="tmp.csv", url="https://raw.com"):
    """
    VoterLabs Inc.
    URL File Downloader
    New File
     """
    result = requests.get(url)
    with open(target, 'w') as input_o:
        input_o.write(result.text)

def compare_csvs(source, target="tmp.csv"):
    """
    VoterLabs Inc.
    URL File Downloader
    Compare to Current File
     """
    if not os.path.exists(source):
        os.rename(target, source)
        return
    if not filecmp.cmp(source, target):
        os.rename(target, source)
        return
    os.remove(target)

def parse_command_line():
    """
    VoterLabs Inc.
    URL File Downloader
    Read Command Line
     """
    parser = argparse.ArgumentParser()
    parser.add_argument('--url_in', help='raw URL csv file input to be read')
    parser.add_argument('--file_out', help='new file downloaded')
    args = parser.parse_args()
    return args
def main():
    """
    VoterLabs Inc.
    URL File Downloader
    Get New File
    Compare to Current File
     """
    args = parse_command_line()
    url_in = args.url_in
    file_out = args.file_out
    download_csv(url=url_in)
    compare_csvs(file_out)
    workdir = "covid_data/"
    if not os.path.exists(workdir):
        os.makedirs(workdir)
if __name__ == '__main__':
    main()
