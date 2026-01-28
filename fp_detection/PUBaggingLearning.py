import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from fp_detection.BaseClassifier import BaseClassifier


class PUBaggingLearning(BaseClassifier):
    def __init__(self, n_estimators=30, base_estimator='svm',
                 sampling_ratio=0.05, random_state=42):
        super().__init__()
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        self.sampling_ratio = sampling_ratio
        self.random_state = random_state
        self.estimators = []
        self.estimator_weights = []

    def predict(self):
        self._train_pu_bagging()
        pred = self._ensemble_predict()
        return pred

    def _train_pu_bagging(self):
        rng = np.random.RandomState(self.random_state)
        for i in range(self.n_estimators):
            # 从非F样本中有放回抽样作为负样本
            n_negative_samples = int(len(self.passing_feature) * self.sampling_ratio)
            negative_indices = rng.choice(
                len(self.passing_feature),
                size=n_negative_samples,
                replace=True
            )

            negative_features = self.passing_feature[negative_indices]

            X_train = np.vstack([self.failing_feature, negative_features])
            y_train = np.hstack([
                np.ones(len(self.failing_feature)),  # 正样本
                np.zeros(len(negative_features))  # 负样本
            ])

            estimator = self._create_base_estimator()
            estimator.fit(X_train, y_train)

            negative_pred = estimator.predict(negative_features)
            negative_accuracy = np.mean(negative_pred == 0)

            self.estimators.append(estimator)
            self.estimator_weights.append(negative_accuracy)

            if (i + 1) % 10 == 0:
                print(f"已完成 {i + 1}/{self.n_estimators} 个基分类器训练")

        # 归一化权重
        self.estimator_weights = np.array(self.estimator_weights)
        if np.sum(self.estimator_weights) > 0:
            self.estimator_weights = self.estimator_weights / np.sum(self.estimator_weights)
        else:
            self.estimator_weights = np.ones(self.n_estimators) / self.n_estimators

    def _create_base_estimator(self):
        if self.base_estimator == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=self.random_state,
                class_weight='balanced',
            )
        elif self.base_estimator == 'svm':
            return SVC(
                probability=True,
                random_state=self.random_state,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"不支持的基分类器: {self.base_estimator}")

    def _ensemble_predict(self):
        all_predictions = []
        all_probabilities = []

        for i, (estimator, weight) in enumerate(zip(self.estimators, self.estimator_weights)):
            # 预测概率
            probabilities = estimator.predict_proba(self.passing_feature)
            fp_proba = probabilities[:, 1] if probabilities.shape[1] == 2 else probabilities[:, 0]
            all_probabilities.append(fp_proba * weight)
            # 单个分类器的预测（用于多样性分析）
            pred = (fp_proba > 0.5).astype(int)
            all_predictions.append(pred)

        # 加权平均概率
        weighted_probabilities = np.sum(all_probabilities, axis=0)

        # 最终预测
        final_predictions = (weighted_probabilities > 0.5).astype(int)
        return final_predictions

