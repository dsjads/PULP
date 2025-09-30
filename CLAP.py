from collections import defaultdict

from PassingVariants_Classification import *
from consistent_testing_manager.LabelData import label_data, do_label_statistics


if __name__ == "__main__":
    system_paths = defaultdict(dict)
    system_paths["BankAccountTP"]["1Bug"] = "C:/Users/zhangt/Desktop/SPLC2021_Full_Dataset/BuggyVersions/BankAccountTP/4wise-BankAccountTP-1BUG-Full"
    system_paths["BankAccountTP"]["2Bug"] = "C:/Users/zhangt/Desktop/SPLC2021_Full_Dataset/BuggyVersions/BankAccountTP/4wise-BankAccountTP-2BUG-Full"
    system_paths["BankAccountTP"]["3Bug"] = "C:/Users/zhangt/Desktop/SPLC2021_Full_Dataset/BuggyVersions/BankAccountTP/4wise-BankAccountTP-3BUG-Full"
    # system_paths["Elevator"]["1Bug"] = "C:/Users/zhangt/Desktop/SPLC2021_Full_Dataset/BuggyVersions/Elevator-FH-JML/4wise-Elevator-FH-JML-1BUG-Full"
    # system_paths["Elevator"]["2Bug"] = "C:/Users/zhangt\Desktop\SPLC2021_Full_Dataset\BuggyVersions\Elevator-FH-JML/4wise-Elevator-FH-JML-2BUG-Full"
    # system_paths["Elevator"]["3Bug"] = "C:/Users/zhangt\Desktop\SPLC2021_Full_Dataset\BuggyVersions\Elevator-FH-JML/4wise-Elevator-FH-JML-3BUG-Full"
    # system_paths["Email"]["1Bug"] = "C:/Users/zhangt/Desktop/SPLC2021_Full_Dataset/BuggyVersions/Email-FH-JML/4wise-Email-FH-JML-1BUG-Full"
    # system_paths["Email"]["2Bug"] = "C:/Users/zhangt/Desktop/SPLC2021_Full_Dataset/BuggyVersions/Email-FH-JML/4wise-Email-FH-JML-2BUG-Full"
    # system_paths["Email"]["3Bug"] = "C:/Users/zhangt/Desktop/SPLC2021_Full_Dataset/BuggyVersions/Email-FH-JML/4wise-Email-FH-JML-3BUG-Full"
    # system_paths["ExamDB"]["1Bug"] = "C:/Users/zhangt/Desktop/SPLC2021_Full_Dataset/BuggyVersions/ExamDB/4wise-ExamDB-1BUG-Full"
    # system_paths["ExamDB"]["2Bug"] = "C:/Users/zhangt/Desktop/SPLC2021_Full_Dataset/BuggyVersions/ExamDB/4wise-ExamDB-2BUG-Full"
    # system_paths["ExamDB"]["3Bug"] = "C:/Users/zhangt/Desktop/SPLC2021_Full_Dataset/BuggyVersions/ExamDB/4wise-ExamDB-3BUG-Full"
    # system_paths["GPL"]["1Bug"] = "C:/Users/zhangt/Desktop/SPLC2021_Full_Dataset/BuggyVersions/GPL/4wise-GPL-1BUG-Full"
    # system_paths["GPL"]["2Bug"] = "C:/Users/zhangt/Desktop/SPLC2021_Full_Dataset/BuggyVersions/GPL/4wise-GPL-2BUG-Full"
    # system_paths["GPL"]["3Bug"] = "C:/Users/zhangt/Desktop/SPLC2021_Full_Dataset/BuggyVersions/GPL/4wise-GPL-3BUG-Full"
    # system_paths["ZipMe"]["1Bug"] = "C:/Users/zhangt/Desktop/SPLC2021_Full_Dataset/BuggyVersions/ZipMe/4wise-ZipMe-1BUG-Full"
    # system_paths["ZipMe"]["2Bug"] = "C:/Users/zhangt/Desktop/SPLC2021_Full_Dataset/BuggyVersions/ZipMe/4wise-ZipMe-2BUG-Full"
    # system_paths["ZipMe"]["3Bug"] = "C:/Users/zhangt/Desktop/SPLC2021_Full_Dataset/BuggyVersions/ZipMe/4wise-ZipMe-3BUG-Full"
    # do_slicing_statements(system_paths)
    # label_data(system_paths)
    # do_label_statistics(system_paths)
    # remove_src_code_file(system_paths)
    # label_data(system_paths)
    calculate_attributes_from_system_paths(system_paths)
    # version_based_classification(system_paths)
    # system_based_classification(system_paths)
    # product_based_classification2(system_paths)
    # product_based_classification(system_paths)
    # within_system_classification(system_paths)
    # intrinsic_analysis(system_paths, system_name="BankAccountTP")

