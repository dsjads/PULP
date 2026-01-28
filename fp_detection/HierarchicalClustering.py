import numpy as np
from sklearn.cluster import AgglomerativeClustering

from fp_detection.BaseClassifier import BaseClassifier

class HierarchicalClusteringClassifier(BaseClassifier):
    def __init__(self):
        super(HierarchicalClusteringClassifier, self).__init__()

    def predict(self):
        hierarchical = AgglomerativeClustering(
            n_clusters=2,
        )
        cluster_labels = hierarchical.fit_predict(self.feature)

        # 以下逻辑与原有KMeansClassifier完全一致，无需修改
        f_count = {0: 0, 1: 0}
        for label, true_lbl in zip(cluster_labels, self.ground_truth_target):
            if true_lbl == 2:
                f_count[label] += 1
        target_cluster = 0 if f_count[0] >= f_count[1] else 1
        pred = np.where(cluster_labels == target_cluster, "FP", "TP")

        non_f_indices = np.where(self.ground_truth_target != 2)[0]
        target = self.ground_truth_target[non_f_indices]
        pred = pred[non_f_indices]
        pred[pred == "TP"] = 0
        pred[pred == "FP"] = 1
        pred = pred.astype(int)
        # self.evaluate(target, pred)
        return pred