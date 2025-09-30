The main function to execute CLAP is in the file CLAP.py. 
In order to execute CLAP, you need to specify paths to folders containing buggy SPL systems. In this work, to conduct different experiments, we clearly separate buggy versions of each system.

Each invoked function is corresponding to a designed experiment.
Specifically:
1. system_based_classification: CLAP is trained with the products in buggy versions of five systems, and the products in the buggy versions of the remaining system used for testing.
2. version_based_classification: All the buggy versions in the six systems are shuffled and then these buggy versions are split into training and testing set by the ratio 8:2.
3. product_based_classification: All the products in all the buggy versions of the six systems are shuffled and then these products are split into a training and testing by ratio 8:2.
4. within_system_classification: The buggy versions of a systems are split into a training and testing set by the ratio 8:2.
5. intrinsic_analysis: This experiment studies the impact of the proposed attributes on CLAP's performance: product implementation, test adequacy, and test effectiveness. This experiment builds different variants of CLAP, which use attributes only in one of these aspects to detect false-passing products, and measure their performance.

Dataset can be found here: https://tuanngokien.github.io/splc2021/
