import argparse

import csv

import pandas
import pandas as pd
import numpy
import luigi






class MyTask(luigi.Task):
    x = luigi.IntParameter()
    y = luigi.IntParameter(default=45)

    def run(self):
        print(self.x + self.y)


