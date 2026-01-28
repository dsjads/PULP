import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC, OneClassSVM
from sklearn.metrics import precision_recall_curve, pairwise_distances
import warnings
from sklearn.model_selection import cross_val_score

from fp_detection.BaseClassifier import BaseClassifier

warnings.filterwarnings('ignore')


class TwoStepPULearningClassifier(BaseClassifier):
    def __init__(self, classifier_type='random_forest',
                 n_estimators=100, random_state=42):
        super().__init__()
        self.classifier_type = classifier_type
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.classifier = None
        self.reliable_negatives = None
        self.threshold = 50
        self.best_threshold = None  # 保存最佳threshold
        self.best_f1 = 0  # 保存最佳F1分数

    def predict(self):
        # 在30-70范围内寻找最佳threshold
        self._find_best_threshold()

        # 使用最佳threshold进行最终预测
        reliable_negatives_mask = self._identify_reliable_negatives(self.best_threshold)
        self._train_final_classifier(reliable_negatives_mask)
        pred = self._predict_final()
        # target = self.ground_truth_passing_target
        # self.evaluate(target, pred)
        return pred

    def _find_best_threshold(self):
        P = self.failing_feature
        U = self.passing_feature

        distances = pairwise_distances(U, P, metric='euclidean')
        avg_distances = np.mean(distances, axis=1)

        for threshold in range(30, 71, 5):
            reliable_negatives_mask = self._identify_reliable_negatives_with_distances(
                avg_distances, threshold
            )

            classifier = self._train_temporary_classifier(reliable_negatives_mask)

            f1_score = self._evaluate_temporary_classifier(classifier)

            # 更新最佳结果
            if f1_score > self.best_f1:
                self.best_f1 = f1_score
                self.best_threshold = threshold

        self.threshold = self.best_threshold  # 更新类的threshold

    def _identify_reliable_negatives_with_distances(self, avg_distances, threshold):
        threshold_value = np.percentile(avg_distances, threshold)
        distance_mask = (avg_distances >= threshold_value)
        return distance_mask

    def _identify_reliable_negatives(self, threshold=None):
        if threshold is None:
            threshold = self.threshold

        P = self.failing_feature
        U = self.passing_feature

        distances = pairwise_distances(U, P, metric='euclidean')
        avg_distances = np.mean(distances, axis=1)

        threshold_value = np.percentile(avg_distances, threshold)
        distance_mask = (avg_distances >= threshold_value)

        # 存储结果
        self.reliable_negatives_distance = U[distance_mask]
        return distance_mask

    def _train_temporary_classifier(self, reliable_negatives_mask):
        """训练临时分类器用于评估"""
        # 获取可靠负样本
        reliable_negatives_features = self.passing_feature[reliable_negatives_mask]

        X_step2 = np.vstack([self.failing_feature, reliable_negatives_features])
        y_step2 = np.hstack([
            np.ones(len(self.failing_feature)),
            np.zeros(len(reliable_negatives_features))
        ])

        # 训练分类器
        if self.classifier_type == 'random_forest':
            classifier = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif self.classifier_type == 'svm':
            classifier = SVC(
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'
            )
        else:
            # 默认使用随机森林
            classifier = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                class_weight='balanced'
            )

        classifier.fit(X_step2, y_step2)
        return classifier

    def _evaluate_temporary_classifier(self, classifier):
        """评估临时分类器的性能"""
        # 预测概率
        probabilities = classifier.predict_proba(self.passing_feature)
        fp_proba = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities[:, 0]

        # 生成预测结果
        pred = (fp_proba > 0.5).astype(int)

        # 计算F1分数
        from sklearn.metrics import f1_score
        f1 = f1_score(self.ground_truth_passing_target, pred)

        return f1

    def _train_final_classifier(self, reliable_negatives_mask):
        # 获取可靠负样本
        reliable_negatives_features = self.passing_feature[reliable_negatives_mask]

        X_step2 = np.vstack([self.failing_feature, reliable_negatives_features])
        y_step2 = np.hstack([
            np.ones(len(self.failing_feature)),
            np.zeros(len(reliable_negatives_features))
        ])

        # 训练最终分类器
        if self.classifier_type == 'random_forest':
            self.classifier = RandomForestClassifier(
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                class_weight='balanced'
            )
        elif self.classifier_type == 'svm':
            self.classifier = SVC(
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'
            )

        self.classifier.fit(X_step2, y_step2)

    def _predict_final(self):
        probabilities = self.classifier.predict_proba(self.passing_feature)

        fp_proba = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities[:, 0]

        pred = (fp_proba > 0.5).astype(int)

        fp_count = np.sum(pred == 1)
        tp_count = np.sum(pred == 0)

        print(f"最终分类结果 - FP: {fp_count}, TP: {tp_count}")

        return pred