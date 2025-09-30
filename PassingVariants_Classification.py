import os
import random
from collections import Counter

import pandas
from sklearn import svm
from sklearn.cluster import KMeans, DBSCAN
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from xlsxwriter import Workbook
import matplotlib.pyplot as plt

from consistent_testing_manager.FPMatricsCaculation import *
import numpy as np
from consistent_testing_manager.FileName import *

from ranking.MultipleBugsManager import get_failing_variants_by_labels

CLASSIFY_ATTRIBUTES = [DDU, code_coverage,
                       incorrectness_verifiability,
                       correctness_reflectability,
                       buggy_statement_containing_possibility,
                       bug_involving_statements]

FIELDS = [VARIANT_NAME, LABEL, DDU,
          code_coverage,
          incorrectness_verifiability,
          correctness_reflectability,
          buggy_statement_containing_possibility,
          bug_involving_statements]


def labeled_cases_count(system_dir):
    mutated_projects = list_dir(system_dir)
    total_labeled_cases = 0
    for project in mutated_projects:
        project_dir = join_path(system_dir, project)
        if os.path.isfile(join_path(project_dir, variant_labels)) and not only_one_failed_tests(
                project_dir):
            total_labeled_cases += 1
    return total_labeled_cases


def only_one_failed_tests(project_dir):
    label_file = join_path(project_dir, variant_labels)
    failing_variants = get_failing_variants_by_labels(label_file, LABEL)
    if len(failing_variants) > 1:
        return False
    for v in failing_variants:
        variant_dir = join_path(join_path(project_dir, "variants"), v)
        failed_test_files = get_failed_test_coverage_dir(variant_dir)
        if len(os.listdir(failed_test_files)) > 1:
            return False
    return True


def load_data_for_system_based_classification(training_systems, testing_systems, attributes):
    all_systems = []
    for s in training_systems:
        all_systems.append(s)
    for s in testing_systems:
        all_systems.append(s)

    train_temp = []
    test_temp = []
    test_sample = {}
    for system in all_systems:
        test_sample[system] = []
        mutated_projects = list_dir(system)
        for project in mutated_projects:
            project_dir = join_path(system, project)
            attribute_file_path = join_path(project_dir, attribute_file)
            if not os.path.isfile(attribute_file_path):
                continue
            df = pandas.read_csv(attribute_file_path)
            if system in training_systems:
                train_temp.append(df)
            else:
                test_temp.append(df)
                test_sample[system].append(project)
    train_result = pandas.concat(train_temp)
    train_data = train_result[attributes].to_numpy()
    train_target = train_result[LABEL].to_numpy()
    train_target[train_target == TRUE_PASSING] = 0
    train_target[train_target == FALSE_PASSING] = 1

    test_result = pandas.concat(test_temp)
    test_data = test_result[attributes].to_numpy()
    test_target = test_result[LABEL].to_numpy()
    test_target[test_target == TRUE_PASSING] = 0
    test_target[test_target == FALSE_PASSING] = 1
    return train_data, test_data, train_target.astype('int'), test_target.astype('int'), test_sample


def write_classified_result(y_pred, test_samples, variant_ratio, file_name):
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
            if variant_ratio == 0:
                variants = df[VARIANT_NAME]
                labels = df[LABEL]
                label_index = 0
            else:
                num_train_items = int(variant_ratio * df.shape[0])
                variants = df.iloc[num_train_items + 1:, :][VARIANT_NAME]
                labels = df.iloc[num_train_items + 1:, :][LABEL]
                label_index = num_train_items + 1

            for v in variants:
                if y_pred[predict_index] == 0:
                    data[v] = {LABEL: labels[label_index], "Classified": TRUE_PASSING}
                else:
                    data[v] = {LABEL: labels[label_index], "Classified": FALSE_PASSING}
                label_index += 1
                predict_index += 1
            classified_file = join_path(project_dir, file_name)
            write_dict_to_file(classified_file, data, [VARIANT_NAME, LABEL, "Classified"])


def overall_performance_measurement(y_test, y_pred, logfile):
    TP_count = 0
    TP_correct = 0
    FP_count = 0
    FP_correct = 0
    predict_index = 0
    for l in y_test:
        if l == 1:
            FP_count += 1
            if y_pred[predict_index] == 1:
                FP_correct += 1
        else:
            TP_count += 1
            if y_pred[predict_index] == 0:
                TP_correct += 1
        predict_index += 1
    logfile.write("tp_precision:" + str(TP_correct / (TP_correct + FP_count - FP_correct)) + "\n")
    logfile.write("fp_precision: " + str(FP_correct / (FP_correct + TP_count - TP_correct)) + "\n")
    logfile.write("tp_recall: " + str(TP_correct / TP_count) + "\n")
    logfile.write("fp_recall: " + str(FP_correct / FP_count) + "\n")
    logfile.write("-------------\n")


def classify_by_svm(logfile, classified_result_file, X_train, X_test, y_train, y_test, test_samples=[]):
    clf = svm.SVC(kernel='linear')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    logfile.write("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)) + "\n")
    overall_performance_measurement(y_test, y_pred, logfile)
    if len(test_samples) > 0:
        write_classified_result(y_pred, test_samples, 0, classified_result_file)


def classify_by_knn(logfile, X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logfile.write("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)) + "\n")
    overall_performance_measurement(y_test, y_pred, logfile)


def classify_by_naive_bayes(logfile, X_train, X_test, y_train, y_test):
    model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logfile.write("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)) + "\n")
    overall_performance_measurement(y_test, y_pred, logfile)


def classify_by_logistic_regression(logfile, X_train, X_test, y_train, y_test):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logfile.write("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)) + "\n")
    overall_performance_measurement(y_test, y_pred, logfile)


def classify_by_decisiontree(logfile, X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier(max_depth=10, random_state=101, max_features=None, min_samples_leaf=15)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    logfile.write("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)) + "\n")
    overall_performance_measurement(y_test, y_pred, logfile)


def classify_by_different_classifiers(logfile, classified_result_file, X_train, X_test, y_train, y_test,
                                      test_samples=[]):
    logfile.write("SVM\n")
    classify_by_svm(logfile, classified_result_file, X_train, X_test, y_train, y_test, test_samples)
    logfile.write("-----------------\n")
    logfile.write("KNN\n")
    classify_by_knn(logfile, X_train, X_test, y_train, y_test)
    logfile.write("-----------------\n")
    logfile.write("Naive bayes\n")
    classify_by_naive_bayes(logfile, X_train, X_test, y_train, y_test)
    logfile.write("-----------------\n")
    logfile.write("Logistic regression\n")
    classify_by_logistic_regression(logfile, X_train, X_test, y_train, y_test)
    logfile.write("-----------------\n")
    logfile.write("Decision tree\n")
    classify_by_decisiontree(logfile, X_train, X_test, y_train, y_test)


def load_data_for_version_based_classification2(logfile, attributes, all_versions):
    pass

def load_data_for_version_based_classification(logfile, attributes, all_versions):
    kfold = KFold(n_splits=5, shuffle=True)
    fold_index = 0
    for train_index, test_index in kfold.split(all_versions):
        logfile.write("fold: " + str(fold_index) + "\n")
        fold_index += 1
        train_temp = []
        test_temp = []
        train_count = 0
        test_count = 0
        for i in range(0, len(all_versions)):
            attribute_file_path = join_path(all_versions[i], attribute_file)
            if not os.path.isfile(attribute_file_path) and only_one_failed_tests(all_versions[i]):
                continue
            df = pandas.read_csv(attribute_file_path)
            if i in train_index:
                train_temp.append(df)
                train_count += 1
            else:
                test_count += 1
                test_temp.append(df)

        train_result = pandas.concat(train_temp)
        train_data = train_result[attributes].to_numpy()
        train_target = train_result[LABEL].to_numpy()
        train_target[train_target == TRUE_PASSING] = 0
        train_target[train_target == FALSE_PASSING] = 1

        test_result = pandas.concat(test_temp)
        test_data = test_result[attributes].to_numpy()
        test_target = test_result[LABEL].to_numpy()
        test_target[test_target == TRUE_PASSING] = 0
        test_target[test_target == FALSE_PASSING] = 1
        X_train, X_test, y_train, y_test = train_data, test_data, train_target.astype('int'), test_target.astype('int')
        logfile.write("training: " + str(train_count))
        logfile.write("testing: " + str(test_count))
        classify_by_different_classifiers(logfile, classified_testing_file, X_train, X_test, y_train,
                                          y_test)


def load_data_for_version_based_one_classifier(logfile, attributes, all_versions):
    kfold = KFold(n_splits=5)
    fold_index = 0
    for train_index, test_index in kfold.split(all_versions):
        logfile.write("fold: " + str(fold_index) + "\n")
        fold_index += 1
        train_temp = []
        test_temp = []
        for i in range(0, len(all_versions)):
            attribute_file_path = join_path(all_versions[i], attribute_file)
            if not os.path.isfile(attribute_file_path):
                continue
            df = pandas.read_csv(attribute_file_path)
            if i in train_index:
                train_temp.append(df)
            else:
                test_temp.append(df)
        train_result = pandas.concat(train_temp)
        train_data = train_result[attributes].to_numpy()
        train_target = train_result[LABEL].to_numpy()
        train_target[train_target == TRUE_PASSING] = 0
        train_target[train_target == FALSE_PASSING] = 1
        test_result = pandas.concat(test_temp)
        test_data = test_result[attributes].to_numpy()
        test_target = test_result[LABEL].to_numpy()
        test_target[test_target == TRUE_PASSING] = 0
        test_target[test_target == FALSE_PASSING] = 1
        X_train, X_test, y_train, y_test = train_data, test_data, train_target.astype('int'), test_target.astype('int')
        classify_by_svm(logfile, classified_testing_file, X_train, X_test, y_train,
                        y_test)


def version_based_classification(system_paths):
    logfile = open("statistics/version_based.log", "w")
    all_versions = []
    for s in system_paths:
        for b in system_paths[s]:
            mutated_projects = list_dir(system_paths[s][b])
            for project in mutated_projects:
                project_dir = join_path(system_paths[s][b], project)
                attribute_file_path = join_path(project_dir, attribute_file)
                if not os.path.isfile(attribute_file_path):
                    continue
                all_versions.append(project_dir)
    load_data_for_version_based_classification(logfile, CLASSIFY_ATTRIBUTES, all_versions)
    logfile.close()

def product_based_classification2(system_paths):
    logfile = open("statistics/product_based_kmeans.log", "w")
    all_versions = []
    for s in system_paths:
        attribute_temp = []
        for b in system_paths[s]:
            mutated_projects = list_dir(system_paths[s][b])
            for project in mutated_projects:
                project_dir = join_path(system_paths[s][b], project)
                attribute_file_path = join_path(project_dir, attribute_file)
                if not os.path.isfile(attribute_file_path):
                    continue
                all_versions.append(project_dir)
            random.shuffle(all_versions)
            for version_dir in all_versions:
                attribute_file_path = join_path(version_dir, attribute_file)
                df = pandas.read_csv(attribute_file_path)
                attribute_temp.append(df)
        attributes = pandas.concat(attribute_temp)
        target = attributes.iloc[:, 1].to_numpy()
        data = attributes.iloc[:, 2:].to_numpy()
        kmeans_clustering(data, target, logfile)

def kmeans_clustering(data, target, logfile):
    cluster_labels = simple_clustering(data, n_clusters=2)
    f_count = {0: 0, 1: 0}
    for label, true_lbl in zip(cluster_labels, target):
        if true_lbl == "F":
            f_count[label] += 1
    target_cluster = 0 if f_count[0] >= f_count[1] else 1
    print(f"F较多的目标簇: {target_cluster}（簇0F数量: {f_count[0]}, 簇1F数量: {f_count[1]}）")
    pred = np.where(cluster_labels == target_cluster, "FP", "TP")
    non_f_indices = np.where(target != "F")[0]
    target = target[non_f_indices]
    pred = pred[non_f_indices]
    target[target == TRUE_PASSING] = 0
    target[target == FALSE_PASSING] = 1
    pred[pred == TRUE_PASSING] = 0
    pred[pred == FALSE_PASSING] = 1
    pred = pred.astype(int)
    overall_performance_measurement(target, pred, logfile)

def oneClass_clustering(data, target,logfile):
    pass

def isolationForest(data, target, logfile):
    pass



def simple_clustering(data, n_clusters = 2, random_state = None):
    kmeans = KMeans(
        n_clusters=2,
        random_state = random_state,
        n_init = 10
    )
    cluster_labels = kmeans.fit_predict(data)
    return cluster_labels

def product_based_classification(system_paths):
    logfile = open("statistics/product_based.log", "w")
    all_versions = []
    for s in system_paths:
        for b in system_paths[s]:
            mutated_projects = list_dir(system_paths[s][b])
            for project in mutated_projects:
                project_dir = join_path(system_paths[s][b], project)
                attribute_file_path = join_path(project_dir, attribute_file)
                if not os.path.isfile(attribute_file_path):
                    continue
                all_versions.append(project_dir)

    random.shuffle(all_versions)
    attribute_temp = []
    for version_dir in all_versions:
        attribute_file_path = join_path(version_dir, attribute_file)
        df = pandas.read_csv(attribute_file_path)
        attribute_temp.append(df)

    attributes = pandas.concat(attribute_temp)
    data = attributes[CLASSIFY_ATTRIBUTES].to_numpy()
    target = attributes[LABEL].to_numpy()
    target[target == TRUE_PASSING] = 0
    target[target == FALSE_PASSING] = 1
    target = target.astype('int')
    kf = KFold(n_splits=5)
    fold_index = 0
    for train_index, test_index in kf.split(data):
        logfile.write("fold: " + str(fold_index) + "\n")
        X_train, X_test = data[train_index], data[test_index]
        y_train, y_test = target[train_index], target[test_index]
        classify_by_svm(logfile, classified_testing_file, X_train, X_test, y_train,
                        y_test)
        fold_index += 1
    logfile.close()


def system_based_classification(system_paths):
    logfile = open("statistics/system_based.log", "w")
    for s1 in system_paths:
        logfile.write("testing system: " + s1 + "\n")
        logfile.write("---------\n")
        training_set = []
        testing_set = []
        for s2 in system_paths:
            for bug in system_paths[s2]:
                if s1 == s2:
                    testing_set.append(system_paths[s2][bug])
                else:
                    training_set.append(system_paths[s2][bug])
        X_train, X_test, y_train, y_test, test_samples = load_data_for_system_based_classification(training_set,
                                                                                                   testing_set,
                                                                                                   CLASSIFY_ATTRIBUTES)
        classify_by_svm(logfile, classified_testing_file, X_train, X_test, y_train,
                        y_test,
                        test_samples)
    logfile.close()


def within_system_classification(system_paths):
    logfile = open("statistics/within_system.log", "w")
    for s in system_paths:
        logfile.write("testing system: " + s + "\n")
        all_versions = []
        for b in system_paths[s]:
            mutated_projects = list_dir(system_paths[s][b])
            for project in mutated_projects:
                project_dir = join_path(system_paths[s][b], project)
                attribute_file_path = join_path(project_dir, attribute_file)
                if not os.path.isfile(attribute_file_path):
                    continue
                all_versions.append(project_dir)

        logfile.write("---------\n")
        load_data_for_version_based_one_classifier(logfile, CLASSIFY_ATTRIBUTES, all_versions)
    logfile.close()


def intrinsic_analysis(system_paths, system_name):
    logfile = open("statistics/intrinsic_analysis.log", "w")
    training_set = []
    testing_set = []
    for s in system_paths:
        for b in system_paths[s]:
            if s != system_name:
                training_set.append(system_paths[s][b])
            else:
                testing_set.append(system_paths[s][b])

    logfile.write("enable only product implementation attributes\n")
    logfile.write("-------------------\n")
    X_train, X_test, y_train, y_test, test_samples = load_data_for_system_based_classification(training_set,
                                                                                               testing_set, [
                                                                                                   buggy_statement_containing_possibility,
                                                                                                   bug_involving_statements])
    classify_by_svm(logfile, classified_testing_file, X_train, X_test, y_train,
                    y_test,
                    test_samples)
    logfile.write("enable only test adequacy attributes\n")
    logfile.write("-------------------\n")
    X_train, X_test, y_train, y_test, test_samples = load_data_for_system_based_classification(training_set,
                                                                                               testing_set,
                                                                                               [DDU, code_coverage])
    classify_by_svm(logfile, classified_testing_file, X_train, X_test, y_train,
                    y_test,
                    test_samples)
    logfile.write("enable only test effectiveness attributes\n")
    logfile.write("-------------------\n")
    X_train, X_test, y_train, y_test, test_samples = load_data_for_system_based_classification(training_set,
                                                                                               testing_set, [
                                                                                                   incorrectness_verifiability,
                                                                                                   correctness_reflectability])
    classify_by_svm(logfile, classified_testing_file, X_train, X_test, y_train,
                    y_test,
                    test_samples)
    logfile.write("enable all attributes\n")
    logfile.write("-------------------\n")
    X_train, X_test, y_train, y_test, test_samples = load_data_for_system_based_classification(training_set,
                                                                                               testing_set,
                                                                                               CLASSIFY_ATTRIBUTES)
    classify_by_svm(logfile, classified_testing_file, X_train, X_test, y_train,
                    y_test,
                    test_samples)
    logfile.close()
