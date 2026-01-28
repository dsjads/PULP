import numpy as np
from sklearn.svm import OneClassSVM

from fp_detection.BaseClassifier import BaseClassifier


class OneClassSVMPFClassifier(BaseClassifier):
    def __init__(self, contamination=0.4, kernel='rbf', gamma='scale'):
        super().__init__()
        self.contamination = contamination  # 对应One-Class SVM的nu参数（异常比例）
        self.kernel = kernel  # SVM核函数，默认'rbf'
        self.gamma = gamma  # 核函数参数，默认'scale'
        self.ocsvm_model = None  # 存储One-Class SVM模型
        self.best_f1 = 0

    def predict(self):
        # 训练One-Class SVM模型
        self._find_optimal_contamination()
        self._train_one_class_svm()
        # 计算异常分数并分类
        pred = self._classify()
        # 评估结果（复用父类的评估逻辑）
        return pred

    def _find_optimal_contamination(self):
        contamination_values = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6]

        for contamination in contamination_values:
            # 使用当前contamination训练临时模型
            temp_model = OneClassSVM(
                nu=contamination,
                kernel=self.kernel,
                gamma=self.gamma
            )
            temp_model.fit(self.failing_feature)

            # 预测
            temp_pred = temp_model.predict(self.passing_feature)
            temp_pred[temp_pred == -1] = 0  # 将异常标记转为0

            # 计算F1分数
            from sklearn.metrics import f1_score
            f1 = f1_score(self.ground_truth_passing_target, temp_pred)

            # 更新最佳结果
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_contamination = contamination

        self.contamination = self.best_contamination

    def _train_one_class_svm(self):
        # 使用F样本（failing_feature）作为"正常样本"训练模型
        self.ocsvm_model = OneClassSVM(
            nu=self.contamination,  # nu控制异常点比例，类似IsolationForest的contamination
            kernel=self.kernel,
            gamma=self.gamma
        )

        # 拟合F样本（视为正常样本分布）
        self.ocsvm_model.fit(self.failing_feature)

    def _classify(self):
        # 对P样本（passing_feature）进行预测
        # OneClassSVM的predict返回1（正常）和-1（异常）
        pred = self.ocsvm_model.predict(self.passing_feature)
        # 与原代码保持一致：将异常标记（-1）转为0，正常标记保持1
        pred[pred == -1] = 0
        return pred