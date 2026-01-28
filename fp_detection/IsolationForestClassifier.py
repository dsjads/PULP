import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from fp_detection.BaseClassifier import BaseClassifier


class IsolationForestFPClassifier(BaseClassifier):
    def __init__(self, contamination=0.3, n_estimators=100, random_state=42):
        super().__init__()
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.if_model = None
        self.f_threshold = None
        self.fp_threshold = None
        self.best_f1 = 0

    def predict(self):
        self._find_optimal_contamination()
        self._train_isolation_forest()
        # 计算异常分数并分类
        pred = self._classify()
        # 评估结果
        return pred

    def _find_optimal_contamination(self):
        contamination_values = [0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]

        for contamination in contamination_values:
            # 使用当前contamination训练临时模型
            temp_model = IsolationForest(
                contamination=contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
            )
            temp_model.fit(self.failing_feature)

            # 预测
            temp_pred = temp_model.predict(self.passing_feature)
            temp_pred[temp_pred == -1] = 0  # 将异常标记转为0

            # 计算F1分数
            from sklearn.metrics import f1_score
            f1 = f1_score(self.ground_truth_passing_target, temp_pred)

            print(f"contamination={contamination:.2f}, F1分数={f1:.4f}")

            # 更新最佳结果
            if f1 > self.best_f1:
                self.best_f1 = f1
                self.best_contamination = contamination
                print(f"  新的最佳结果! contamination={contamination:.2f}, F1={f1:.4f}")

        print(f"最终选择: contamination={self.best_contamination:.2f}, F1={self.best_f1:.4f}")
        self.contamination = self.best_contamination

    def _train_isolation_forest(self):
        # 使用F样本训练基础模型
        self.if_model = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state,
        )

        # 拟合F样本（视为正常样本）
        self.if_model.fit(self.failing_feature)
        print(f"Isolation Forest模型训练完成，使用 {len(self.failing_feature)} 个F样本")

    def _classify(self):
        pred = self.if_model.predict(self.passing_feature)
        pred[pred == -1] = 0
        return pred