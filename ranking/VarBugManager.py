import csv
import os

from FileManager import get_model_configs_report_path


def is_var_bug_by_config(mutated_project_dir, bases):
    config_report_path = get_model_configs_report_path(mutated_project_dir)

    with open(config_report_path) as f:
        reader = csv.reader(f, delimiter=',')
        header = next(reader)
        feature_names = header[1:]
        for row in reader:
            if row[-1] == "__FAILED__":
                flag = True
                for i in range(0, len(feature_names)):

                    if row[i + 1].strip() == "T":
                        if feature_names[i] not in bases:
                            flag = False

                if flag:
                    return 0
    return 1
