import csv
import os
import random
from platform import system_alias

import numpy as np
import pandas
from numba.np.arrayobj import np_array

from FileManager import list_dir, join_path
from PassingVariants_Classification import write_classified_result
from consistent_testing_manager.FileName import attribute_file, classified_testing_file
from fp_detection.BaseClassifier import overall_performance_measurement
from fp_detection.IsolationForestClassifier import IsolationForestFPClassifier
from fp_detection.KMeansClassifier import KMeansClassifier
from fp_detection.KNNClassifier import KNNClassifier
from fp_detection.OneClassSVMClassifier import OneClassSVMPFClassifier
from fp_detection.PUBaggingLearning import PUBaggingLearning
from fp_detection.TunedKNNClassifier import TunedKNNClassifier
from fp_detection.TwoStepPULearning import TwoStepPULearningClassifier

os.environ['OMP_NUM_THREADS'] = '1'

LABEL = 'LABEL'
VARIANT_NAME = 'model'
TRUE_PASSING = 'TP'
FALSE_PASSING = 'FP'

def ablation_analysis(system_paths, log_path):
    logfile = open(log_path, "w")
    group_size = 10
    for i in range(5):
        total_pred = np.array([])
        total_target = np.array([])
        for s in system_paths:
            print(s)
            attribute_temp =[]
            all_versions = []
            for b in system_paths[s]:
                mutated_projects = list_dir(system_paths[s][b])
                for project in mutated_projects:
                    project_dir = join_path(system_paths[s][b], project)
                    attribute_file_path = join_path(project_dir, attribute_file)
                    if not os.path.isfile(attribute_file_path):
                        continue
                    all_versions.append(project_dir)
                for version_dir in all_versions:
                    attribute_file_path = join_path(version_dir, attribute_file)
                    df = pandas.read_csv(attribute_file_path)
                    attribute_temp.append(df)
            attributes = pandas.concat(attribute_temp)
            target = attributes.iloc[:,1].to_numpy()
            data = attributes.iloc[:,2:].to_numpy()
            start = i * group_size
            end = (i + 1) * group_size
            keep_cols = np.concatenate([
                np.arange(start,end)
            ])
            data = data[:, keep_cols]
            pl = PUBaggingLearning()
            pl.prepare_data(data, target, logfile)
            pred = pl.predict()
            target = pl.ground_truth_passing_target
            total_pred = np.concatenate((total_pred, pred)) if total_pred.size > 0 else pred
            total_target = np.concatenate((total_target, target)) if total_target.size > 0 else target
        overall_performance_measurement(total_target, total_pred, logfile, i)

def product_based_classification2(system_paths, log_path):
    logfile = open(log_path, "w")
    total_pred = np.array([])  # 初始化为空ndarray
    total_target = np.array([])
    for s in system_paths:
        attribute_temp = []
        all_versions = []
        system_set = []
        for b in system_paths[s]:
            system_set.append(system_paths[s][b])
            mutated_projects = list_dir(system_paths[s][b])
            for project in mutated_projects:
                project_dir = join_path(system_paths[s][b], project)
                attribute_file_path = join_path(project_dir, attribute_file)
                if not os.path.isfile(attribute_file_path):
                    continue
                all_versions.append(project_dir)
            # random.shuffle(all_versions)
            for version_dir in all_versions:
                attribute_file_path = join_path(version_dir, attribute_file)
                df = pandas.read_csv(attribute_file_path)
                attribute_temp.append(df)
        systems = load_systems(system_set)
        attributes = pandas.concat(attribute_temp)
        target = attributes.iloc[:, 1].to_numpy()
        data = attributes.iloc[:, 2:].to_numpy()
        keep_cols = np.concatenate([
            np.arange(0, 50),
        ])
        data = data[:, keep_cols]
        
        pl = KMeansClassifier()
        # pl = KNNClassifier()
        # pl = IsolationForestFPClassifier()
        # pl = OneClassSVMPFClassifier()
        # pl = PUBaggingLearning()
        # pl = TwoStepPULearningClassifier()
        pl.prepare_data(data, target, logfile)
        pred = pl.predict()
        target = pl.ground_truth_passing_target
        total_pred = np.concatenate((total_pred, pred)) if total_pred.size > 0 else pred
        total_target = np.concatenate((total_target, target)) if total_target.size > 0 else target
        overall_performance_measurement(target, pred,logfile, s)
        write_classified_result2(pred, systems, 0, classified_testing_file)
    overall_performance_measurement(total_target, total_pred, logfile, "total")
    # classify_by_different_classifiers(logfile, classified_testing_file, X_train, X_test, y_train,
    #                                   y_test)
    # classify_by_svm(logfile, classified_result_file, X_train, X_test, y_train, y_test, test_samples)
    # write_classified_result(total_pred, [], 0, classified_testing_file)

def within_system_classification2(system_paths, log_path):
    logfile = open(log_path, "w")
    total_pred = np.array([])  # 初始化为空ndarray
    total_target = np.array([])
    for s in system_paths:
        all_versions = []
        for b in system_paths[s]:
            mutated_projects = list_dir(system_paths[s][b])
            for project in mutated_projects:
                project_dir = join_path(system_paths[s][b], project)
                attribute_file_path = join_path(project_dir, attribute_file)
                if not os.path.isfile(attribute_file_path):
                    continue
                all_versions.append(project_dir)
            for version_dir in all_versions:
                attribute_file_path = join_path(version_dir, attribute_file)
                df = pandas.read_csv(attribute_file_path)
                target = df.iloc[:, 1].to_numpy()
                data = df.iloc[:, 2:].to_numpy()
                pl = PUBaggingLearning()
                pl.prepare_data(data, target, logfile)
                pred = pl.predict()
                target = pl.ground_truth_passing_target
                total_pred = np.concatenate((total_pred, pred)) if total_pred.size > 0 else pred
                total_target = np.concatenate((total_target, target)) if total_target.size > 0 else target
    overall_performance_measurement(total_target, total_pred, logfile, "total")

def dataset_based_classification(system_paths, log_path):
    logfile = open(log_path, "w")
    attribute_temp = []
    for s in system_paths:
        all_versions = []
        for b in system_paths[s]:
            mutated_projects = list_dir(system_paths[s][b])
            for project in mutated_projects:
                project_dir = join_path(system_paths[s][b], project)
                attribute_file_path = join_path(project_dir, attribute_file)
                if not os.path.isfile(attribute_file_path):
                    continue
                all_versions.append(project_dir)
            for version_dir in all_versions:
                attribute_file_path = join_path(version_dir, attribute_file)
                df = pandas.read_csv(attribute_file_path)
                attribute_temp.append(df)
    attributes = pandas.concat(attribute_temp)
    target = attributes.iloc[:, 1].to_numpy()
    data = attributes.iloc[:, 2:].to_numpy()
    pl = PUBaggingLearning()
    pl.prepare_data(data, target, logfile)
    pred = pl.predict()
    target = pl.ground_truth_passing_target
    overall_performance_measurement(target, pred,logfile, "total")

def load_systems(system_set):
    systems = {}
    for system in system_set:
        systems[system] = []
        mutated_projects = list_dir(system)
        for project in mutated_projects:
            project_dir = join_path(system, project)
            attribute_file_path = join_path(project_dir, attribute_file)
            if not os.path.isfile(attribute_file_path):
                continue
            systems[system].append(project)
    return systems

def write_classified_result2(y_pred, test_samples, variant_ratio, file_name):
    predict_index = 0
    for system in test_samples:
        testing_projects = test_samples[system]
        if len(testing_projects) == 0:
            continue
        for project in testing_projects:
            data = {}
            project_dir = join_path(system, project)
            attribute_file_path = join_path(project_dir, attribute_file)
            df = pandas.read_csv(attribute_file_path)
            df = df[df[LABEL] != 'F']
            if variant_ratio == 0:
                variants = df[VARIANT_NAME]
                labels = df[LABEL]
            else:
                num_train_items = int(variant_ratio * df.shape[0])
                variants = df.iloc[num_train_items + 1:, :][VARIANT_NAME]
                labels = df.iloc[num_train_items + 1:, :][LABEL]

            for idx, v in enumerate(variants):
                if y_pred[predict_index] == 0:
                    data[v] = {LABEL: labels.iloc[idx], "Classified": TRUE_PASSING}
                else:
                    data[v] = {LABEL: labels.iloc[idx], "Classified": FALSE_PASSING}
                predict_index+=1
            classified_file = join_path(project_dir, file_name)
            write_dict_to_file(classified_file, data, [VARIANT_NAME, LABEL, "Classified"])

def write_dict_to_file(file_name, data, fieldnames):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for item in data:
            tmp = {}
            for f in fieldnames:
                if fieldnames.index(f) == 0:
                    tmp[f] = item
                else:
                    tmp[f] = data[item][f]
            writer.writerow(tmp)
