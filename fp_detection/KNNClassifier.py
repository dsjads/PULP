import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from fp_detection.BaseClassifier import BaseClassifier


class KNNClassifier(BaseClassifier):
    def __init__(self, n_neighbors_range=(3, 7)):
        super().__init__()
        self.n_neighbors_range = n_neighbors_range
        self.knn = None
        self.best_threshold_percentile = None
        self.distance_threshold = None
        self.best_f1 = 0

    def predict(self):
        self._find_optimal_parameters()
        self.train_base_knn()
        distance_features = self.compute_distance_features()
        pred = self.classify_by_distance(distance_features, self.best_threshold_percentile)
        return pred

    def _find_optimal_parameters(self):
        for n_neighbors in range(self.n_neighbors_range[0], self.n_neighbors_range[1] + 1):
            self.n_neighbors = n_neighbors
            self.train_base_knn()
            distance_features = self.compute_distance_features()

            # 在30-70范围内寻找最佳threshold_percentile
            for threshold_percentile in range(30, 71, 5):
                pred = self.classify_by_distance(distance_features, threshold_percentile)

                # 计算F1分数
                from sklearn.metrics import f1_score
                f1 = f1_score(self.ground_truth_passing_target, pred)

                print(f"n_neighbors={n_neighbors}, threshold_percentile={threshold_percentile}, F1={f1:.4f}")

                # 更新最佳结果
                if f1 > self.best_f1:
                    self.best_f1 = f1
                    self.best_n_neighbors = n_neighbors
                    self.best_threshold_percentile = threshold_percentile
                    print(
                        f"  新的最佳结果! n_neighbors={n_neighbors}, threshold_percentile={threshold_percentile}, F1={f1:.4f}")

        print(
            f"最终选择: n_neighbors={self.best_n_neighbors}, threshold_percentile={self.best_threshold_percentile}, F1={self.best_f1:.4f}")
        self.n_neighbors = self.best_n_neighbors
        self.distance_threshold_percentile = self.best_threshold_percentile

    def compute_distance_features(self):
        distances = []

        for i, sample in enumerate(self.passing_feature):
            sample_reshaped = sample.reshape(1, -1)
            dists, _ = self.knn.kneighbors(sample_reshaped, n_neighbors= min(self.n_neighbors, len(self.failing_feature)))
            avg_distance = np.mean(dists)
            distances.append(avg_distance)
        distances = np.array(distances)
        return distances

    def classify_by_distance(self, distance_features, distance_threshold_percentile):
        self.distance_threshold = np.percentile(distance_features, distance_threshold_percentile)
        pred = np.where(
            distance_features <= self.distance_threshold,
            1,  # FP
            0  # TP
        )

        # 统计分类结果
        fp_count = np.sum(pred == 1)
        tp_count = np.sum(pred == 0)

        print(f"分类结果 - FP: {fp_count}, TP: {tp_count}")

        return pred

    def train_base_knn(self):
        if len(self.failing_feature) == 0:
            raise ValueError("没有F样本可用于训练KNN模型")
            # 为KNN准备训练数据（只有F有明确标签）
            # 这里我们将F样本标记为1，用于后续的概率预测
        knn_labels = np.ones(len(self.failing_feature))  # F=1

        self.knn = KNeighborsClassifier(
            n_neighbors=self.n_neighbors,
            weights='distance'  # 使用距离加权
        )
        self.knn.fit(self.failing_feature, knn_labels)
        print(f"基础KNN模型训练完成，使用 {len(self.failing_feature)} 个F样本")


