import requests
import filecmp
import logging
import os

log = logging.getLogger("logger")
log.setLevel(logging.INFO)
logging.basicConfig()

class Downloader(object):

    def __init__(self, workdir="covid_data/"):
        log.info("Initializing Downloader")
        self.workdir = workdir
        if not os.path.exists(self.workdir):
            logging.info(f"Creating dir: {self.workdir}")
            os.makedirs(self.workdir)

    def download_csv(self, target="tmp.csv",
        url="https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_US.csv"):


        log.info(f"Downloading {url} to {target}")
        result = requests.get(url)
        with open(target, 'w') as io:
            io.write(result.text)
        log.info(f"{target} written.")


    def compare_csvs(self, source, target="tmp.csv"):

        print("compare_csvs")
        print(source)
        print(target)

        #source = self.workdir + source
        #target = self.workdir + target
        log.info(f"Checking if {target} is different from {source}.")
        if not os.path.exists(source):
            log.info(f"{skuuuuuource} does not exist, renaming {target}")
            os.rename(target, source)
            return

        if not filecmp.cmp(source, target):
            log.info(f"Files are different, overwritting {source}.")
            os.rename(target, source)
            return

        log.info(f"Files are the same, removing {target}.")
        os.remove(target)


downloader = Downloader()