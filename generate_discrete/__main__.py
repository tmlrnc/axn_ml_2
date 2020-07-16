import argparse
from datetime import datetime, timedelta
import os

from datetime import date

def diff_dates(date1, date2):
    return abs(date2-date1).days

def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in')
    parser.add_argument('--file_out_discrete')
    parser.add_argument('--file_out')
    parser.add_argument('--discrete_file_script_out')
    parser.add_argument('--start_date_all')
    parser.add_argument('--end_date_all')
    parser.add_argument('--num_bins')
    parser.add_argument('--window_size')
    parser.add_argument('--parent_dir')
    parser.add_argument(
        '--drop_column',
        action='append')

    args = parser.parse_args()
    return args

def main():
    args = parse_command_line()
    file_in = args.file_in
    file_out_discrete = args.file_out_discrete
    file_out = args.file_out

    discrete_file_script_out = args.discrete_file_script_out
    start_date_all = args.start_date_all
    end_date_all = args.end_date_all
    num_bins = args.num_bins
    window_size = args.window_size
    drop_column = args.drop_column

    parent_dir = args.parent_dir
    if parent_dir is None:
        print("Parent dir is not specified.")
        quit()
    print(f"Using parent_dir: {parent_dir}")

    start_date_all_window_f = datetime.strptime(start_date_all, "%m/%d/%Y")
    end_date_all_window_f = datetime.strptime(end_date_all, "%m/%d/%Y")

    start_window_date_next = start_date_all_window_f
    end_window_date_next = start_date_all_window_f + timedelta(days=int(window_size))
    print("start_window_date_next ")
    print(start_window_date_next)
    print("end_window_date_next ")
    print(end_window_date_next)
    print(end_date_all_window_f)


    while (end_window_date_next < end_date_all_window_f ):

        start_window_date = start_window_date_next
        end_window_date = end_window_date_next
        time_series = start_window_date.strftime("%m-%d-%Y")  + "_" + end_window_date.strftime("%m-%d-%Y")

        import os

        # Directory
        directory = time_series

        # Parent Directory path
        #parent_dir = "/Users/tomlorenc/Sites/VL_standard/ml"
        #parent_dir = "/app"


        # Path
        path = os.path.join(parent_dir, directory)
        os.mkdir(path)


        drops = []
        for drop in drop_column:
            drops.append(drop)

        dropsforme = "\\\n".join(f"  --drop_column  {d}" for d in drops)
        start_date_all_f = datetime.strptime(start_date_all, "%m/%d/%Y")
        end_date_all_f = datetime.strptime(end_date_all, "%m/%d/%Y")
        date_drops = []

        while start_window_date > start_date_all_f:
            date_drops.append(start_date_all_f)
            start_date_all_f = start_date_all_f + timedelta(days=1)

        end_date_w_f = end_window_date + timedelta(days=1)
        while end_date_all_f >= end_date_w_f:
            date_drops.append(end_date_w_f)
            end_date_w_f = end_date_w_f + timedelta(days=1)

        dropsdatesforme = "\\\n".join(f"  --drop_column  {d.strftime('%m/%d/%Y')}" for d in date_drops)
        dropsdatesforme2 = dropsdatesforme.replace("2020", "20")
        dropsdatesforme3 = dropsdatesforme2.replace("03", "3")
        dropsdatesforme4 = dropsdatesforme3.replace("04", "4")
        dropsdatesforme5 = dropsdatesforme4.replace("05", "5")
        dropsdatesforme6 = dropsdatesforme5.replace("06", "6")
        dropsdatesforme7 = dropsdatesforme6.replace("07", "7")
        dropsdatesforme8 = dropsdatesforme7.replace("01", "1")
        dropsdatesforme9 = dropsdatesforme8.replace("02", "2")
        dropsdatesforme10 = dropsdatesforme9.replace("08", "8")
        dropsdatesforme11 = dropsdatesforme10.replace("09", "9")

        start_date_window_f = start_window_date
        end_date_window_f = end_window_date

        dates = []
        while end_date_window_f >= start_date_window_f:
            dates.append(start_date_window_f)
            start_date_window_f = start_date_window_f + timedelta(days=1)


        options = "\\\n".join(f"  --dicretize uniform {num_bins} {d.strftime('%m/%d/%Y')} " for d in dates)
        no = options.replace("/", "\/")
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
        file_out_discrete_ts = file_out_discrete.replace(".csv", tscsv)
        file_out_ts = file_out.replace(".csv", tscsv)

        file_out_discrete_ts_path = path + "/" + file_out_discrete_ts
        file_out_ts_path = path + "/" + file_out_ts


        template = f"""
        #!/usr/bin/env bash
        python -m discrete  \\
          --file_in {file_in} \\
        {dropsforme} \\
            {dropsdatesforme11} \\
          {no11} \\
          --file_out_discrete {file_out_discrete_ts_path} \\
          --file_out {file_out_ts_path}
    
        """.strip()


        tssh = "_" + time_series + ".sh"
        discrete_file_script_out_ts = discrete_file_script_out.replace(".sh", tssh)
        discrete_file_script_out_ts_path = path + "/" + discrete_file_script_out_ts


        print("discrete_file_script_out_ts ")
        print(discrete_file_script_out_ts_path)
        print(template)
        discrete_text_file = open(discrete_file_script_out_ts_path, "w")
        discrete_text_file.write(template)

        start_window_date_next = start_window_date_next + timedelta(days=1)
        end_window_date_next = start_window_date_next + timedelta(days=int(window_size))
        print("start_window_date_next ")
        print(start_window_date_next)
        print("end_window_date_next ")
        print(end_window_date_next)


def main_old():
    args = parse_command_line()
    file_in = args.file_in
    file_out_discrete = args.file_out_discrete
    file_out = args.file_out
    discrete_file_script_out = args.discrete_file_script_out
    start_window_date = args.start_window_date
    end_window_date = args.end_window_date
    start_date = args.start_date
    end_date = args.end_date
    num_bins = args.num_bins
    window_size = args.window_size
    drop_column = args.drop_column


    start_date_window_f = datetime.strptime(start_window_date, "%m/%d/%Y")
    end_date_window_f = datetime.strptime(end_window_date, "%m/%d/%Y")


    drops = []
    for drop in drop_column:
        drops.append(drop)

    dropsforme = "\\\n".join(f"  --drop_column  {d}" for d in drops)

    start_date_all_f = datetime.strptime(start_date, "%m/%d/%Y")
    end_date_all_f = datetime.strptime(end_date, "%m/%d/%Y")

    date_drops = []

    while start_date_window_f > start_date_all_f:
        date_drops.append(start_date_all_f)
        start_date_all_f = start_date_all_f + timedelta(days=1)

    end_date_w_f = end_date_window_f + timedelta(days=1)
    while end_date_all_f > end_date_w_f:
        date_drops.append(end_date_w_f)
        end_date_w_f = end_date_w_f + timedelta(days=1)

    dropsdatesforme = "\\\n".join(f"  --drop_column  {d.strftime('%m/%d/%Y')}" for d in date_drops)
    dropsdatesforme2 = dropsdatesforme.replace("2020", "20")
    dropsdatesforme3 = dropsdatesforme2.replace("03", "3")
    dropsdatesforme4 = dropsdatesforme3.replace("04", "4")
    dropsdatesforme5 = dropsdatesforme4.replace("05", "5")
    dropsdatesforme6 = dropsdatesforme5.replace("06", "6")
    dropsdatesforme7 = dropsdatesforme6.replace("07", "7")
    dropsdatesforme8 = dropsdatesforme7.replace("01", "1")
    dropsdatesforme9 = dropsdatesforme8.replace("02", "2")
    dropsdatesforme10 = dropsdatesforme9.replace("08", "8")
    dropsdatesforme11 = dropsdatesforme10.replace("09", "9")

    start_date_window_f = datetime.strptime(start_window_date, "%m/%d/%Y")
    end_date_window_f = datetime.strptime(end_window_date, "%m/%d/%Y")

    dates = []
    while end_date_w_f >= start_date_w_f:
        dates.append(start_date_w_f)
        start_date_w_f = start_date_w_f + timedelta(days=1)


    options = "\\\n".join(f"  --dicretize uniform {num_bins} {d.strftime('%m/%d/%Y')} " for d in dates)
    no = options.replace("/", "\/")
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



    template = f"""
    #!/usr/bin/env bash
    python -m discrete  \\
      --file_in {file_in} \\
    {dropsforme} \\
        {dropsdatesforme11} \\
      {no11} \\
      --file_out_discrete {file_out_discrete} \\
      --file_out {file_out}

    """.strip()

    print(template)
    discrete_text_file = open(discrete_file_script_out, "w")
    discrete_text_file.write(template)


if __name__ == '__main__':
    main()
