from collections import defaultdict

from PassingVariants_Classification import *
from consistent_testing_manager.LabelData import label_data, do_label_statistics
from fl.fl_evaluation import fl_with_fp, calculate_average_rank
from fp_detection.core import product_based_classification, dataset_based_classification, \
    within_system_classification, ablation_analysis

if __name__ == "__main__":
    system_paths = defaultdict(dict)
    system_paths["BankAccountTP"]["1Bug"] = "/home/yuanxixing/BuggyVersions/BankAccountTP/4wise-BankAccountTP-1BUG-Full"
    system_paths["BankAccountTP"]["2Bug"] = "/home/yuanxixing/BuggyVersions/BankAccountTP/4wise-BankAccountTP-2BUG-Full"
    system_paths["BankAccountTP"]["3Bug"] = "/home/yuanxixing/BuggyVersions/BankAccountTP/4wise-BankAccountTP-3BUG-Full"
    system_paths["Elevator"]["1Bug"] = "/home/yuanxixing/BuggyVersions/Elevator-FH-JML/4wise-Elevator-FH-JML-1BUG-Full"
    system_paths["Elevator"]["2Bug"] = "/home/yuanxixing/BuggyVersions/Elevator-FH-JML/4wise-Elevator-FH-JML-2BUG-Full"
    system_paths["Elevator"]["3Bug"] = "/home/yuanxixing/BuggyVersions/Elevator-FH-JML/4wise-Elevator-FH-JML-3BUG-Full"
    system_paths["Email"]["1Bug"] = "/home/yuanxixing/BuggyVersions/Email-FH-JML/4wise-Email-FH-JML-1BUG-Full"
    system_paths["Email"]["2Bug"] = "/home/yuanxixing/BuggyVersions/Email-FH-JML/4wise-Email-FH-JML-2BUG-Full"
    system_paths["Email"]["3Bug"] = "/home/yuanxixing/BuggyVersions/Email-FH-JML/4wise-Email-FH-JML-3BUG-Full"
    system_paths["ExamDB"]["1Bug"] = "/home/yuanxixing/BuggyVersions/ExamDB/4wise-ExamDB-1BUG-Full"
    system_paths["ExamDB"]["2Bug"] = "/home/yuanxixing/BuggyVersions/ExamDB/4wise-ExamDB-2BUG-Full"
    system_paths["ExamDB"]["3Bug"] = "/home/yuanxixing/BuggyVersions/ExamDB/4wise-ExamDB-3BUG-Full"
    system_paths["GPL"]["1Bug"] = "/home/yuanxixing/BuggyVersions/GPL/4wise-GPL-1BUG-Full"
    system_paths["GPL"]["2Bug"] = "/home/yuanxixing/BuggyVersions/GPL/4wise-GPL-2BUG-Full"
    system_paths["GPL"]["3Bug"] = "/home/yuanxixing/BuggyVersions/GPL/4wise-GPL-3BUG-Full"
    system_paths["ZipMe"]["1Bug"] = "/home/yuanxixing/BuggyVersions/ZipMe/4wise-ZipMe-1BUG-Full"
    system_paths["ZipMe"]["2Bug"] = "/home/yuanxixing/BuggyVersions/ZipMe/4wise-ZipMe-2BUG-Full"
    system_paths["ZipMe"]["3Bug"] = "/home/yuanxixing/BuggyVersions/ZipMe/4wise-ZipMe-3BUG-Full"
    # system_paths["BankAccountTP"]["1Bug"] = "D:/BuggyVersions/BankAccountTP/4wise-BankAccountTP-1BUG-Full"
    # system_paths["BankAccountTP"]["2Bug"] = "D:/BuggyVersions/BankAccountTP/4wise-BankAccountTP-2BUG-Full"
    # system_paths["BankAccountTP"]["3Bug"] = "D:/BuggyVersions/BankAccountTP/4wise-BankAccountTP-3BUG-Full"
    # system_paths["Elevator"]["1Bug"] = "D:/BuggyVersions/Elevator-FH-JML/4wise-Elevator-FH-JML-1BUG-Full"
    # system_paths["Elevator"]["2Bug"] = "D:/BuggyVersions/Elevator-FH-JML/4wise-Elevator-FH-JML-2BUG-Full"
    # system_paths["Elevator"]["3Bug"] = "D:/BuggyVersions/Elevator-FH-JML/4wise-Elevator-FH-JML-3BUG-Full"
    # system_paths["Email"]["1Bug"] = "D:/BuggyVersions/Email-FH-JML/4wise-Email-FH-JML-1BUG-Full"
    # system_paths["Email"]["2Bug"] = "D:/BuggyVersions/Email-FH-JML/4wise-Email-FH-JML-2BUG-Full"
    # system_paths["Email"]["3Bug"] = "D:/BuggyVersions/Email-FH-JML/4wise-Email-FH-JML-3BUG-Full"
    # system_paths["ExamDB"]["1Bug"] = "D:/BuggyVersions/ExamDB/4wise-ExamDB-1BUG-Full"
    # system_paths["ExamDB"]["2Bug"] = "D:/BuggyVersions/ExamDB/4wise-ExamDB-2BUG-Full"
    # system_paths["ExamDB"]["3Bug"] = "D:/BuggyVersions/ExamDB/4wise-ExamDB-3BUG-Full"
    # system_paths["GPL"]["1Bug"] = "D:/BuggyVersions/GPL/4wise-GPL-1BUG-Full"
    # system_paths["GPL"]["2Bug"] = "D:/BuggyVersions/GPL/4wise-GPL-2BUG-Full"
    # system_paths["GPL"]["3Bug"] = "D:/BuggyVersions/GPL/4wise-GPL-3BUG-Full"
    # system_paths["ZipMe"]["1Bug"] = "D:/BuggyVersions/ZipMe/4wise-ZipMe-1BUG-Full"
    # system_paths["ZipMe"]["2Bug"] = "D:/BuggyVersions/ZipMe/4wise-ZipMe-2BUG-Full"
    # system_paths["ZipMe"]["3Bug"] = "D:/BuggyVersions/ZipMe/4wise-ZipMe-3BUG-Full"
    # label_data(system_paths)
    fl_with_fp("/results/splfl/",system_paths)
    calculate_attributes_from_system_paths(system_paths)
    # calculate_average_rank(system_paths, "/results/splfl/")
    # do_generate_fl_results(system_paths)
    product_based_classification(system_paths, "statistics/product_hierarchicalClustering.log")
    # calculate_attributes_from_system_paths(system_paths)
    # product_based_classification2(system_paths, "statistics/cr2.log")
    within_system_classification(system_paths, "statistics/within_system_bagging.log")
    # version_based_classification2(system_paths)
    # do_slicing_statements(system_paths)
    # product_based_classification(system_paths)
    # within_system_classification(system_paths)
    # intrinsic_analysis(system_paths, system_name="BankAccountTP")
    ablation_analysis(system_paths, "statistics/ablation3.log")
    # system_based_classification(system_paths)

