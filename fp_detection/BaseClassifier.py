from abc import abstractmethod

import numpy as np
from sklearn.preprocessing import LabelEncoder

class BaseClassifier:
    def __init__(self):
        self.model = None
        self.logfile = None
        self.classes = None
        self.ground_truth_target = None
        self.fp_target = {}
        self.feature = None
        self.label_mapping = {'FP': 1, 'TP': 0, 'F': 2}
        self.reverse_mapping = {1: 'FP', 0: 'TP', 2: 'F'}
        self.failing_idx = None
        self.passing_idx = None
        self.failing_feature = None
        self.passing_feature = None
        self.ground_truth_passing_target = None

    def prepare_data(self, data, ground_truth_target, logfile):
        self.logfile = logfile
        self.ground_truth_target = np.array([self.label_mapping[label] for label in ground_truth_target])
        self.feature = data
        self.separate_data()

    def separate_data(self):
        self.failing_idx = np.where(self.ground_truth_target == 2)[0]
        self.passing_idx = np.where(self.ground_truth_target != 2)[0]

        self.failing_feature = self.feature[self.failing_idx]
        self.passing_feature = self.feature[self.passing_idx]
        self.ground_truth_passing_target = self.ground_truth_target[self.passing_idx]


    @abstractmethod
    def predict(self):
        pass

    def evaluate(self, target, pred):
        overall_performance_measurement(target, pred, self.logfile)


def overall_performance_measurement(y_test, y_pred, logfile, mark = ""):
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
        # 计算核心指标（无需处理除零）
    total_samples = len(y_test)
    accuracy = (TP_correct + FP_correct) / total_samples

    tp_precision = TP_correct / (TP_correct + FP_count - FP_correct)
    fp_precision = FP_correct / (FP_correct + TP_count - TP_correct)

    tp_recall = TP_correct / TP_count
    fp_recall = FP_correct / FP_count

    # 计算F1-score
    tp_f1 = 2 * (tp_precision * tp_recall) / (tp_precision + tp_recall)
    fp_f1 = 2 * (fp_precision * fp_recall) / (fp_precision + fp_recall)

    # 写入日志
    logfile.write(f"------{mark}-------\n")
    logfile.write(f"tp_precision: {tp_precision:.4f}\n")
    logfile.write(f"tp_recall: {tp_recall:.4f}\n")
    logfile.write(f"tp_f1: {tp_f1:.4f}\n")
    logfile.write(f"fp_precision: {fp_precision:.4f}\n")
    logfile.write(f"fp_recall: {fp_recall:.4f}\n")
    logfile.write(f"fp_f1: {fp_f1:.4f}\n")
    logfile.write(f"accuracy: {accuracy:.4f}\n")
    logfile.write("-------------\n")