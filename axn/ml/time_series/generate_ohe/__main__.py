"""
generates the ohe scripts
"""
# pylint: disable=invalid-name
import argparse
from datetime import datetime, timedelta
import os

description = \
    """
VoterLabs Inc.
    features are encoded using a one-hot ‘one-of-K’ encoding scheme.
    This creates a binary column for each category and returns a sparse matrix or dense array
    the encoder derives the categories based on the unique values in each feature.

     when features are categorical.
     For example a person could have features
     ["male", "female"],
     ["from Europe", "from US", "from Asia"],
     ["uses Firefox", "uses Chrome", "uses Safari", "uses Internet Explorer"].
     Such features can be efficiently coded as integers,
     for instance ["male", "from US", "uses Internet Explorer"] could be expressed as [0, 1, 3]
     while ["female", "from Asia", "uses Chrome"] would be [1, 2, 1].

    READ FILE_IN_RAW.CSV

  LOAD CSV DATA FROM YOUR COMPUTER

Must be a csv file where first row has column header names.
Must include time series date columns - MM/DD/YY (7/3/20)
Must include targeted date or will automatically predict last date in series.
Must include as much data of cause of time series as you can - more data equals better predictions

    GET COLUMN HEADERS
    FOR EACH COLUMN NOT IN IGNORE LIST :
    GET ALL CATEGORIES = UNIQUE COLUMN VALUES
    GENERATE ONE HOT ENCODING HEADER
    ENCODE EACH ROW WITH 1 or 0 FOR EACH HEADER
      """.strip()


def parse_command_line():
    """
    reads the command line args
    """
    # pylint: disable=invalid-name
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        '--file_in',
        help='raw csv file input to be predicted. Must be a csv file where first row has column header names. '
             'Must include time series date columns - like MM/DD/YY (7/3/20) ')
    parser.add_argument(
        '--file_out',
        help='csv file output encoded using one-hot one-of-K encoding scheme')
    parser.add_argument(
        '--ohe_file_script_out',
        help='ohe output script for each time splt directory of data')
    parser.add_argument(
        '--start_date_all',
        help='start of time series window - each step is a day each column must be a date in format MM/DD/YY - like 7/3/20')
    parser.add_argument(
        '--end_date_all',
        help='end of time series window - each step is a day each column must be a date in format MM/DD/YY - like 7/22/20 ')
    parser.add_argument(
        '--window_size',
        help='number of time series increments per window - this is an integet like 4. '
             'This is the sliding window method for framing a time series dataset the increments are days')
    parser.add_argument(
        '--parent_dir',
        help='beginning of docker file system - like /app')

    parser.add_argument(
        '--ignore',
        action='append',
        help='columns of data to NOT be encoded or discretized - remove from processing without removing '
             'from raw data because they might be usseful to know latrer - like first name')
    args = parser.parse_args()
    return args


def main():
    """
    runs the master module
    """
    # pylint: disable=invalid-name
    # pylint: disable=too-many-locals
    # pylint: disable=consider-using-sys-exit
    # pylint: disable=unused-variable
    # pylint: disable=too-many-statements
    args = parse_command_line()
    file_in = args.file_in
    file_out = args.file_out
    ohe_file_script_out = args.ohe_file_script_out
    start_date_all = args.start_date_all
    end_date_all = args.end_date_all
    window_size = args.window_size

    start_date_all_window_f = datetime.strptime(start_date_all, "%m/%d/%Y")
    end_date_all_window_f = datetime.strptime(end_date_all, "%m/%d/%Y")

    start_window_date_next = start_date_all_window_f
    end_window_date_next = start_date_all_window_f + \
        timedelta(days=int(window_size))
    print("start_window_date_next ")
    print(start_window_date_next)
    print("end_window_date_next ")
    print(end_window_date_next)
    print(end_date_all_window_f)

    parent_dir = args.parent_dir
    if parent_dir is None:
        print("Parent dir is not specified.")
        quit()
    print(f"Using parent_dir: {parent_dir}")

    while end_window_date_next < end_date_all_window_f:

        start_window_date = start_window_date_next
        end_window_date = end_window_date_next
        time_series = start_window_date.strftime(
            "%m-%d-%Y") + "_" + end_window_date.strftime("%m-%d-%Y")


        # Directory
        directory = time_series

        # Parent Directory path
        #parent_dir = "/Users/tomlorenc/Sites/VL_standard/ml"
        #parent_dir = "/app"

        # Path
        path = os.path.join(parent_dir, directory)
        try:
            os.mkdir(path)
        except OSError as error:
            print("fick")

        start_date_window_f = start_window_date
        end_date_window_f = end_window_date

        my_ignores = []
        for ignore in args.ignore:
            my_ignores.append(ignore)

        ignore_options = "\\\n".join(f"  --ignore   {i}" for i in my_ignores)

        dates = []
        while end_date_window_f >= start_date_window_f:
            dates.append(start_date_window_f)
            start_date_window_f = start_date_window_f + timedelta(days=1)

        options = "\\\n".join(
            f"  --ignore   {d.strftime('%m/%d/%Y')}_DISCRETE" for d in dates)
        no = options.replace("/", r"\/")
        no2 = no.replace("2020", "20")
        no3 = no2.replace("03", "3")
        no4 = no3.replace("04", "4")
        no5 = no4.replace("05", "5")
        no6 = no5.replace("06", "6")
        no7 = no6.replace("07", "7")
        no8 = no7.replace("01", "1")
        no9 = no8.replace("02", "2")
        no10 = no9.replace("08", "8")
        no11 = no10.replace("09", "9")

        tscsv = "_" + time_series + ".csv"
        file_out_ts = file_out.replace(".csv", tscsv)
        file_out_ts_path = path + "/" + file_out_ts

        file_in_ts = file_in.replace(".csv", tscsv)
        file_in_ts_path = path + "/" + file_in_ts

        template = f"""
        #!/usr/bin/env bash
        python -m ohe  \\
          --file_in {file_in_ts_path} \\
          {no11} \\
        {ignore_options} \\
          --file_out {file_out_ts_path}

        """.strip()

        tssh = "_" + time_series + ".sh"
        discrete_file_script_out_ts = ohe_file_script_out.replace(".sh", tssh)
        discrete_file_script_out_ts_path = path + "/" + discrete_file_script_out_ts

        print("discrete_file_script_out_ts_path ")
        print(discrete_file_script_out_ts_path)
        print(template)
        discrete_text_file = open(discrete_file_script_out_ts_path, "w")
        discrete_text_file.write(template)

        start_window_date_next = start_window_date_next + timedelta(days=1)
        end_window_date_next = start_window_date_next + \
            timedelta(days=int(window_size))


if __name__ == '__main__':
    main()
