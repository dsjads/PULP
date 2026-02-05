The main function to execute PULP is in the file PULP.py. 
In order to execute PULP, you need to specify paths to folders containing buggy SPL systems. In this work, to conduct different experiments, we clearly separate buggy versions of each system.

Each invoked function is corresponding to a designed experiment.
Specifically:

label_data: prepare data and label the ground truth false-passing products labels

calculate_attributes_from_system_paths: feature extraction

ablation_analysis: designed for RQ2, ablation analysis

product_based_classification: product level false-passing products classification

within_system_classification: system level false-passing products classification (default)

dataset_based_classification: cross-system level false-passing products classification

fl_with_fp: fault localization after mitigation the effect of false-passing products

Dataset can be found here: https://tuanngokien.github.io/splc2021/
