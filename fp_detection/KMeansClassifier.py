import numpy as np
from sklearn.cluster import KMeans

from consistent_testing_manager.FPMatricsCaculation import TRUE_PASSING
from fp_detection.BaseClassifier import BaseClassifier


class KMeansClassifier(BaseClassifier):
    def __init__(self):
        super(KMeansClassifier, self).__init__()

    def predict(self):
        kmeans = KMeans(
            n_clusters=2,
            max_iter=1,
            random_state=42,
        )
        cluster_labels = kmeans.fit_predict(self.feature)
        f_count = {0:0, 1:0}
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

