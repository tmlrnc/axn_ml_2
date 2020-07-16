from covid import downloader
import datetime as dt
import argparse
from datetime import datetime, timedelta


def parse_command_line():
    parser = argparse.ArgumentParser()
    parser.add_argument('--target')
    parser.add_argument('--file_out_scores')
    parser.add_argument('--file_out_predict')
    parser.add_argument('--ignore')
    parser.add_argument('--file_out_tableau')
    parser.add_argument('--file_in_master')

    parser.add_argument('--file_in')
    parser.add_argument('--predict_file_script_out')
    parser.add_argument('--start_date_all')
    parser.add_argument('--end_date_all')
    parser.add_argument('--window_size')
    parser.add_argument('--parent_dir')
    parser.add_argument(
        '--add_model',
        action='append')
    args = parser.parse_args()
    return args


def main():
    args = parse_command_line()
    file_in = args.file_in
    target = args.target
    file_out_scores = args.file_out_scores
    file_in_master = args.file_in_master

    file_out_predict = args.file_out_predict
    start_date_all = args.start_date_all
    end_date_all = args.end_date_all
    predict_file_script_out = args.predict_file_script_out
    add_model = args.add_model
    ignore = args.ignore
    file_out_tableau = args.file_out_tableau

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


    parent_dir = args.parent_dir
    if parent_dir is None:
        print("Parent dir is not specified.")
        quit()
    print(f"Using parent_dir: {parent_dir}")


    while (end_window_date_next < end_date_all_window_f):
        start_window_date = start_window_date_next
        end_window_date = end_window_date_next
        time_series = start_window_date.strftime("%m-%d-%Y") + "_" + end_window_date.strftime("%m-%d-%Y")

        import os

        # Directory
        directory = time_series

        # Parent Directory path
        #parent_dir = "/Users/tomlorenc/Sites/VL_standard/ml"
        #parent_dir = "/app"


        # Path
        path = os.path.join(parent_dir, directory)

        tscsv = "_" + time_series + ".csv"
        file_out_predict_ts = file_out_predict.replace(".csv", tscsv)
        file_out_predict_ts_path = path + "/" + file_out_predict_ts

        file_out_scores_ts = file_out_scores.replace(".csv", tscsv)
        file_out_scores_path = path + "/" + file_out_scores_ts

        file_in_ts = file_in.replace(".csv", tscsv)
        file_in_ts_path = path + "/" + file_in_ts
        try:
            os.mkdir(path)
        except OSError as error:
            print("fick")

        start_date_window_f = start_window_date
        end_date_window_f = end_window_date

        models = []
        for model in add_model:
            models.append(model)

        model_options = "\\\n".join(f"  --predictor   {m}" for m in models)


        dates = [end_window_date]
        options = "\\\n".join(f"  --target   {d.strftime('%m/%d/%Y')}_DISCRETE" for d in dates)
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
        python -m predict  \\
          --file_in {file_in_ts_path} \\
                    --file_in_master {file_in_master} \\
          --strategy none \\
          {no11} \\
         --ignore  {ignore} \\
          --training_test_split_percent 70 \\
             {model_options} \\
          --score f1_score \\
          --score classification_accuracy \\
          --score recall  \\
          --file_in_config config/ohe_config.yaml \\
          --file_out_scores {file_out_scores_path} \\
                    --file_out_scores {file_out_scores_path} \\
        --file_out_tableau {file_out_tableau} \\
          --file_out_predict {file_out_predict_ts_path}
    
    
        """.strip()


        print(template)

        discrete_text_file = open(predict_file_script_out, "w")

        discrete_text_file.write(template)

        tssh = "_" + time_series + ".sh"
        predict_file_script_out_ts = predict_file_script_out.replace(".sh", tssh)
        predict_file_script_out_ts_path = path + "/" + predict_file_script_out_ts


        print("predict_file_script_out_ts_path ")
        print(predict_file_script_out_ts_path)
        print(template)
        discrete_text_file = open(predict_file_script_out_ts_path, "w")
        discrete_text_file.write(template)


        start_window_date_next = start_window_date_next + timedelta(days=1)
        end_window_date_next = start_window_date_next + timedelta(days=int(window_size))


if __name__ == '__main__':
    main()


