from covid import downloader
import datetime as dt
import argparse
from datetime import datetime, timedelta


def parse_command_line():
    parser = argparse.ArgumentParser()


    parser.add_argument('--file_in')
    parser.add_argument('--master_file_script_out')
    parser.add_argument('--ohe_file_script_out')
    parser.add_argument('--predict_file_script_out')
    parser.add_argument('--discrete_file_script_out')

    parser.add_argument('--start_date_all')
    parser.add_argument('--end_date_all')
    parser.add_argument('--window_size')
    args = parser.parse_args()
    return args


def main():
    args = parse_command_line()
    file_in = args.file_in
    start_date_all = args.start_date_all
    end_date_all = args.end_date_all
    discrete_file_script_out = args.discrete_file_script_out

    master_file_script_out = args.master_file_script_out
    predict_file_script_out = args.predict_file_script_out

    window_size = args.window_size

    ohe_file_script_out = args.ohe_file_script_out

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

        tssh = "_" + time_series + ".sh"
        discrete_file_script_out_ts = discrete_file_script_out.replace(".sh", tssh)
        discrete_file_script_out_ts_path = path + "/" + discrete_file_script_out_ts

        tssh = "_" + time_series + ".sh"
        ohe_file_script_out_ts = ohe_file_script_out.replace(".sh", tssh)
        ohe_file_script_out_ts_path = path + "/" + ohe_file_script_out_ts

        tssh = "_" + time_series + ".sh"
        predict_file_script_out_ts = predict_file_script_out.replace(".sh", tssh)
        predict_file_script_out_ts_path = path + "/" + predict_file_script_out_ts



        try:
            os.mkdir(path)
        except OSError as error:
            print("fick")

        start_date_window_f = start_window_date
        end_date_window_f = end_window_date
        import time
        start = time.time()
        import os

        comm = "exec bash " + discrete_file_script_out_ts_path
        os.system(comm)
        comm2 = "exec bash " + ohe_file_script_out_ts_path
        os.system(comm2)

        comm3 = "exec bash " + predict_file_script_out_ts_path
        os.system(comm3)

        print('It took {0:0.1f} seconds'.format(time.time() - start))


        start_window_date_next = start_window_date_next + timedelta(days=1)
        end_window_date_next = start_window_date_next + timedelta(days=int(window_size))


if __name__ == '__main__':
    main()


