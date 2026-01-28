import os
from collections import defaultdict

import numpy as np
import pandas as pd
from multiprocessing import Pool
from FileManager import join_path, list_dir, list_files
from consistent_testing_manager.FileName import classified_testing_file
from ranking import RankingManager
from ranking.MultipleBugsManager import multiple_bugs_ranking, multiple_slicing




def fl_with_fp(result_folder, system_paths):
    for system in system_paths:
        for bug in system_paths[system]:
            sys_path = system_paths[system][bug]
            spectrum_expressions = [
                "Tarantula", "Ochiai", "Op2", "Barinel", "Dstar"
            ]

            multiple_bugs_ranking(result_folder,
                                  system, sys_path,
                                  bug, "remove_fps",
                                  spectrum_expressions, True,
                                  classified_testing_file, keep_useful_tests= True, add_more_tests=False)

            # multiple_slicing(sys_path, True, True)


def calculate_average_rank(system_paths, result_folder, strategy = "remove_fps"):
    # spectrum_expressions = [
    #     "Jaccard", "Tarantula", "Ochiai", "Op2", "Dstar", "Barinel", "Russell_rao", "GP02", "GP03", "GP19"
    # ]
    sheet_summary = defaultdict(list)
    output_excel = "summary-remove-fps-varcop-best.xlsx"

    for system in system_paths:
        aggregations = RankingManager.AGGREGATION_ARITHMETIC_MEAN
        normalizations = RankingManager.NORMALIZATION_ALPHA_BETA
        system_result_dir = join_path(result_folder, system, normalizations, aggregations)
        result_dir = join_path(system_result_dir, strategy)
        result_files = list_files(result_dir)
        for file in result_files:
            all_sheets = pd.read_excel(file, sheet_name=None)
            for sheet_name, df in all_sheets.items():
                dataframe = df.values
                sliced_data = dataframe[:, [0,2,3]]

                col_names = df.columns[[0,2,3]].tolist()

                sliced_data = filter_lower_rank_stmts(sliced_data)

                sliced_data[:, 0] = sliced_data[:, 0].astype(np.object_)
                # 再执行前缀拼接（此时不会截断，因为object dtype支持任意长度字符串）
                sliced_data[:, 0] = np.char.add(system, sliced_data[:, 0].astype(str))

                sliced_df = pd.DataFrame(sliced_data, columns=col_names)
                sheet_summary[sheet_name].append(sliced_df)
    with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
        for sheet_name, df in sheet_summary.items():
            merged_df = pd.concat(df, ignore_index=True)
            print(f"工作表 {sheet_name} 汇总后形状：{merged_df.shape}")

            # 将拼接后的数据写入总Excel的对应工作表
            merged_df.to_excel(writer, sheet_name=sheet_name, index=False)
    print(f"\n✅ 总Excel生成完成！保存路径：{os.path.abspath(output_excel)}")
    return output_excel  # 返回总Excel路径，方便后续使用


def filter_lower_rank_stmts(data):
    data_copy = data.copy()
    id_col = data_copy[:, 0]
    for i in range(1, len(id_col)):
        if pd.isna(id_col[i]) and not pd.isna(id_col[i - 1]):
            id_col[i] = id_col[i - 1]
    data_copy[:, 0] = id_col

    valid_mask = ~pd.isna(data_copy[:, 0])  # 非NaN的行掩码
    data_filled = data_copy[valid_mask]

    unique_ids = np.unique(data_filled[:, 0])
    filtered_rows = []

    for uid in unique_ids:
        # 5.2 筛选当前标识的所有行
        group_mask = (data_filled[:, 0] == uid)
        group_data = data_filled[group_mask]

        # 5.3 找到组内比较列最小值对应的行（若有多个最小值，保留第一行）
        min_rank = np.min(group_data[:, 1])
        min_row = group_data[group_data[:, 1] == min_rank][0]  # 取第一行最小值行

        filtered_rows.append(min_row)

    # 5.4 转换为ndarray并返回
    filtered_arr = np.array(filtered_rows, dtype=data.dtype)  # 保持原数据类型
    return filtered_arr



if __name__ == "__main__":
    system_paths = defaultdict(dict)
    system_paths["BankAccountTP"]["1Bug"] = "D:/BuggyVersions/BankAccountTP/4wise-BankAccountTP-1BUG-Full"
    system_paths["BankAccountTP"]["2Bug"] = "D:/BuggyVersions/BankAccountTP/4wise-BankAccountTP-2BUG-Full"
    system_paths["BankAccountTP"]["3Bug"] = "D:/BuggyVersions/BankAccountTP/4wise-BankAccountTP-3BUG-Full"
    system_paths["Elevator"]["1Bug"] = "D:/BuggyVersions/Elevator-FH-JML/4wise-Elevator-FH-JML-1BUG-Full"
    system_paths["Elevator"]["2Bug"] = "D:/BuggyVersions/Elevator-FH-JML/4wise-Elevator-FH-JML-2BUG-Full"
    system_paths["Elevator"]["3Bug"] = "D:/BuggyVersions/Elevator-FH-JML/4wise-Elevator-FH-JML-3BUG-Full"
    system_paths["Email"]["1Bug"] = "D:/BuggyVersions/Email-FH-JML/4wise-Email-FH-JML-1BUG-Full"
    system_paths["Email"]["2Bug"] = "D:/BuggyVersions/Email-FH-JML/4wise-Email-FH-JML-2BUG-Full"
    system_paths["Email"]["3Bug"] = "D:/BuggyVersions/Email-FH-JML/4wise-Email-FH-JML-3BUG-Full"
    system_paths["ExamDB"]["1Bug"] = "D:/BuggyVersions/ExamDB/4wise-ExamDB-1BUG-Full"
    system_paths["ExamDB"]["2Bug"] = "D:/BuggyVersions/ExamDB/4wise-ExamDB-2BUG-Full"
    system_paths["ExamDB"]["3Bug"] = "D:/BuggyVersions/ExamDB/4wise-ExamDB-3BUG-Full"
    system_paths["GPL"]["1Bug"] = "D:/BuggyVersions/GPL/4wise-GPL-1BUG-Full"
    system_paths["GPL"]["2Bug"] = "D:/BuggyVersions/GPL/4wise-GPL-2BUG-Full"
    system_paths["GPL"]["3Bug"] = "D:/BuggyVersions/GPL/4wise-GPL-3BUG-Full"
    system_paths["ZipMe"]["1Bug"] = "D:/BuggyVersions/ZipMe/4wise-ZipMe-1BUG-Full"
    system_paths["ZipMe"]["2Bug"] = "D:/BuggyVersions/ZipMe/4wise-ZipMe-2BUG-Full"
    system_paths["ZipMe"]["3Bug"] = "D:/BuggyVersions/ZipMe/4wise-ZipMe-3BUG-Full"
    # label_data(system_paths)
    fl_with_fp("D:/splfl/",system_paths)
