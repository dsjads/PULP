from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from fp_detection.KNNClassifier import KNNClassifier


class TunedKNNClassifier(KNNClassifier):
    def __init__(self, param_grid=None, cv=5):
        super().__init__()
        self.param_grid = param_grid or {
            'n_neighbors': [3, 5, 7, 9, 11, 15],
            'distance_threshold_percentile': [25, 30, 35, 40, 45, 50, 55, 60],
            'weights': ['uniform', 'distance']
        }
        self.cv = cv
        self.best_params_ = None

    def tune_parameters(self):
        """使用网格搜索调参"""
        print("开始参数调优...")

        # 准备数据
        X = self.passing_feature
        y = self.ground_truth_passing_target

        # 自定义评分函数
        def custom_scorer(estimator, X, y):
            # 这里需要根据您的KNN结构调整
            distances = estimator.compute_distance_features(X)
            pred = estimator.classify_by_distance(distances)
            return f1_score(y, pred, average='weighted')

        # 由于我们的KNN结构特殊，需要手动实现网格搜索
        best_score = -1
        best_params = {}

        for n_neighbors in self.param_grid['n_neighbors']:
            for percentile in self.param_grid['distance_threshold_percentile']:
                for weight in self.param_grid['weights']:

                    # 创建分类器实例
                    knn_temp = KNNClassifier(
                        n_neighbors=n_neighbors,
                        distance_threshold_percentile=percentile
                    )
                    knn_temp.failing_feature = self.failing_feature
                    knn_temp.passing_feature = self.passing_feature
                    knn_temp.ground_truth_passing_target = self.ground_truth_passing_target

                    # 训练和评估
                    try:
                        knn_temp.train_base_knn()
                        distances = knn_temp.compute_distance_features()
                        pred = knn_temp.classify_by_distance(distances)

                        score = f1_score(y, pred, average='weighted')

                        if score > best_score:
                            best_score = score
                            best_params = {
                                'n_neighbors': n_neighbors,
                                'distance_threshold_percentile': percentile,
                                'weights': weight
                            }
                    except Exception as e:
                        continue

        print(f"最佳参数: {best_params}")
        print(f"最佳分数: {best_score:.4f}")

        # 更新当前实例的参数
        self.n_neighbors = best_params['n_neighbors']
        self.distance_threshold_percentile = best_params['distance_threshold_percentile']
        self.knn = KNeighborsClassifier(
            n_neighbors=best_params['n_neighbors'],
            weights=best_params['weights']
        )
        self.best_params_ = best_params

        return best_params