from covid import downloader
import datetime as dt
import argparse



def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url_in')
    parser.add_argument('--file_out')
    args = parser.parse_args()
    return args


def main():

    print("covid  --- START ")

    args = parse_command_line()
    url_in = args.url_in
    file_out = args.file_out

    print(file_out)

    downloader.download_csv(url=url_in)
    today = str(dt.datetime.date(dt.datetime.now())) +  ".csv"
    downloader.compare_csvs(file_out)


    print("covid  --- END ")


if __name__ == '__main__':
    main()


