from covid import downloader
import datetime as dt
import argparse
from datetime import datetime, timedelta


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_in')
    parser.add_argument('--file_out')
    parser.add_argument('--ohe_file_script_out')
    parser.add_argument('--start_date_all')
    parser.add_argument('--end_date_all')
    parser.add_argument('--window_size')
    parser.add_argument(
        '--ignore',
        action='append')
    args = parser.parse_args()
    return args


def main_old():
    args = parse_command_line()
    file_in = args.file_in
    start_window_date = args.start_window_date
    end_window_date = args.end_window_date
    file_out = args.file_out
    ohe_file_script_out = args.ohe_file_script_out


    start_date = datetime.strptime(start_window_date, "%m_%d_%Y")
    end_date = datetime.strptime(end_window_date, "%m_%d_%Y")

    dates = []
    while end_date >= start_date:
        dates.append(start_date)
        start_date = start_date + timedelta(days=1)

    options = "\\\n".join(f"  --ignore   {d.strftime('%m/%d/%Y')}_DISCRETE" for d in dates)
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

    template = f"""
    #!/usr/bin/env bash


    python -m ohe  \\
      --file_in {file_in} \\
      {no10} \\
      --file_out {file_out}

    """.strip()


    print(template)

    discrete_text_file = open(ohe_file_script_out, "w")

    discrete_text_file.write(template)


def main():
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
    end_window_date_next = start_date_all_window_f + timedelta(days=int(window_size))
    print("start_window_date_next ")
    print(start_window_date_next)
    print("end_window_date_next ")
    print(end_window_date_next)
    print(end_date_all_window_f)

    while (end_window_date_next < end_date_all_window_f):

        start_window_date = start_window_date_next
        end_window_date = end_window_date_next
        time_series = start_window_date.strftime("%m-%d-%Y") + "_" + end_window_date.strftime("%m-%d-%Y")


        import os

        # Directory
        directory = time_series

        # Parent Directory path
        parent_dir = "/Users/tomlorenc/Sites/VL_standard/ml"

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

        options = "\\\n".join(f"  --ignore   {d.strftime('%m/%d/%Y')}_DISCRETE" for d in dates)
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
        end_window_date_next = start_window_date_next + timedelta(days=int(window_size))




if __name__ == '__main__':
    main()


