import os

import pandas

from FileManager import join_path, EXPERIMENT_RESULT_FOLDER
from experimental_results_analyzer.ImprovementComparisonAnalyzer import init_comparison_data
from ranking.Keywords import *
from spectrum_manager.Spectrum_Expression import JACCARD, SORENSEN_DICE, TARANTULA, OCHIAI, OP2, BARINEL, DSTAR, ROGERS_TANIMOTO, \
    AMPLE, \
    SIMPLE_MATCHING, RUSSELL_RAO, COHEN, SCOTT, ROGOT1, GEOMETRIC_MEAN, M2, WONG1, SOKAL, DICE, HUMANN, ZOLTAR, \
    WONG2, ROGOT2, EUCLID, HAMMING, FLEISS, ANDERBERG, KULCZYNSKI2, HARMONIC_MEAN, GOODMAN

from xlsxwriter import Workbook

# data_column = [VARCOP_RANK, VARCOP_EXAM, VARCOP_SPACE, VARCOP_TC_RANK, VARCOP_TC_EXAM, SBFL_TC_RANK, SBFL_TC_EXAM,
#                FB_TC_RANK, FB_TC_EXAM, TC_SPACE, VARCOP_DISABLE_BPC_RANK, VARCOP_DISABLE_BPC_EXAM,
#                SBFL_RANK, SBFL_EXAM, FB_RANK, FB_EXAM, SPACE]
data_column = [VARCOP_RANK, VARCOP_SPACE, VARCOP_DISABLE_BPC_RANK, SBFL_RANK, SPACE]
rank_column = [VARCOP_RANK, VARCOP_TC_RANK, SBFL_TC_RANK, FB_TC_RANK, VARCOP_DISABLE_BPC_RANK, SBFL_RANK]
SBFL_METRIC_COL = 0
NUM_CASES_COL = 1
NUM_BUGS_COL = 2
VARCOP_RANK_COL = 3
VARCOP_EXAM_COL = 4
VARCOP_SPACE_COL = 5
VARCOP_DISABLE_BPC_RANK_COL = 6
VARCOP_DISABLE_BPC_EXAM_COL = 7
SBFL_RANK_COL = 8
SBFL_EXAM_COL = 9
FB_RANK_COL = 10
FB_EXAM_COL = 11
SPACE_COL = 12

# SPECTRUM_EXPRESSIONS_LIST = [TARANTULA, OCHIAI, OP2, BARINEL, DSTAR,
#                              RUSSELL_RAO, SIMPLE_MATCHING, ROGERS_TANIMOTO, AMPLE, JACCARD,
#                              COHEN, SCOTT, ROGOT1, GEOMETRIC_MEAN, M2,
#                              WONG1, SOKAL, SORENSEN_DICE, DICE, HUMANN, WONG2, ZOLTAR,
#                              EUCLID, ROGOT2, HAMMING, FLEISS, ANDERBERG,
#                              GOODMAN, HARMONIC_MEAN, KULCZYNSKI2]


SPECTRUM_EXPRESSIONS_LIST = [TARANTULA, OCHIAI, OP2, BARINEL, DSTAR]

def write_all_bugs_to_a_file(summary_file_dir, file_lists, num_of_bugs, base_path):
    writer = pandas.ExcelWriter(summary_file_dir, engine='openpyxl')

    row = 0
    num_of_file = 0
    for file in file_lists:
        for b in num_of_bugs:
            file_name = join_path(file, b + ".xlsx")
            file_path = join_path(base_path, file_name)
            print(file_path)
            if os.path.exists(file_path):
                excel_data_df = pandas.read_excel(file_path, sheet_name=None, engine='openpyxl')
                num_of_file += 1
                if num_of_file > 1:
                    for spectrum_expression_type in SPECTRUM_EXPRESSIONS_LIST:
                        excel_data_df[spectrum_expression_type].to_excel(writer, sheet_name=spectrum_expression_type,
                                                                         startrow=row,
                                                                         index=False, header=False)
                    row += len(excel_data_df[TARANTULA])
                else:
                    for spectrum_expression_type in SPECTRUM_EXPRESSIONS_LIST:
                        excel_data_df[spectrum_expression_type].to_excel(writer, sheet_name=spectrum_expression_type,
                                                                         startrow=row,
                                                                         index=False)
                    row += len(excel_data_df[TARANTULA]) + 1

    writer.save()


def summary_hitx(hitx_file_dir, all_bugs_file_dir, hitn):
    wb = Workbook(hitx_file_dir)
    sheet = wb.add_worksheet("sheet1")

    row = 0
    for hit_index in range(1, hitn + 1):
        sheet.write(row, hit_index * 3 - 2, HIT + str(hit_index))

    row += 1
    sheet.write(row, 0, SBFL_METRIC)
    for hit_index in range(1, hitn + 1):
        col = hit_index * 3 - 2
        sheet.write(row, col, HIT_VARCOP)
        sheet.write(row, col + 1, HIT_TC_SBFL)
        sheet.write(row, col + 2, HIT_SBFL)
    row += 1

    excel_data_df = pandas.read_excel(all_bugs_file_dir, sheet_name=None)

    for spectrum_expression_type in SPECTRUM_EXPRESSIONS_LIST:
        sheet.write(row, 0, spectrum_expression_type)

        col = 0;
        for hit in range(1, hitn + 1):
            col += 1
            sheet.write(row, col, count_hit_x(excel_data_df[spectrum_expression_type][VARCOP_RANK], hit))
            col += 1
            sheet.write(row, col,
                        count_hit_x(excel_data_df[spectrum_expression_type][SBFL_TC_RANK], hit))
            col += 1
            sheet.write(row, col,
                        count_hit_x(excel_data_df[spectrum_expression_type][SBFL_RANK], hit))

        row += 1

    wb.close()


def count_hit_x(value_list, x):
    count = 0
    for value in value_list:
        if type(value) != str and value != -1 and value <= x:
            count += 1
    return count


def summary_result(all_bugs_file, summary_file, prefix):
    summary_file_dir = join_path(EXPERIMENT_RESULT_FOLDER,
                                 summary_file)
    wb = Workbook(summary_file_dir)
    sheet = wb.add_worksheet("sheet1")

    row = 0
    write_header_in_sumary_file(row, sheet)
    row += 1
    comparison_data = calculate_average_in_a_file(all_bugs_file, row, sheet)

    wb.close()
    return comparison_data


def write_header_in_sumary_file(row, sheet):
    sheet.write(row, SBFL_METRIC_COL, SBFL_METRIC)
    sheet.write(row, NUM_CASES_COL, NUM_CASES)
    sheet.write(row, NUM_BUGS_COL, NUM_BUGS)
    col = NUM_BUGS_COL + 1
    for item in data_column:
        sheet.write(row, col, item)
        col += 1
    # sheet.write(row, VARCOP_RANK_COL, VARCOP_RANK)
    # sheet.write(row, VARCOP_EXAM_COL, VARCOP_EXAM)
    # sheet.write(row, VARCOP_SPACE_COL, VARCOP_SPACE)
    # sheet.write(row, VARCOP_DISABLE_BPC_RANK_COL, VARCOP_DISABLE_BPC_RANK)
    # sheet.write(row, VARCOP_DISABLE_BPC_EXAM_COL, VARCOP_DISABLE_BPC_EXAM)
    # sheet.write(row, SBFL_RANK_COL, SBFL_RANK)
    # sheet.write(row, SBFL_EXAM_COL, SBFL_EXAM)
    # sheet.write(row, FB_RANK_COL, FB_RANK)
    # sheet.write(row, FB_EXAM_COL, FB_EXAM)
    # sheet.write(row, SPACE_COL, SPACE)


def num_of_element(data_list):
    element_count = 0
    for r in data_list:
        if not pandas.isnull(r):
            element_count += 1
    return element_count


def calculate_average_in_a_file(experimental_file_dir, row, sheet):
    excel_data_df = pandas.read_excel(experimental_file_dir, sheet_name=None, engine='openpyxl')
    comparison_data = init_comparison_data()

    for spectrum_expression_type in SPECTRUM_EXPRESSIONS_LIST:
        num_of_cases = num_of_element(excel_data_df[spectrum_expression_type][BUG_ID])
        sheet.write(row, NUM_CASES_COL, num_of_cases)

        num_of_bugs = num_of_element(excel_data_df[spectrum_expression_type][BUGGY_STM])
        sheet.write(row, NUM_BUGS_COL, num_of_bugs)

        sheet.write(row, SBFL_METRIC_COL, spectrum_expression_type)

        average_value_list = average_best_rank_exam(excel_data_df, spectrum_expression_type)
        # average_value_list = percentage_of_cases_found_bugs(experimental_file_dir, spectrum_expression_type, 3)
        col = NUM_BUGS_COL + 1
        for metric in data_column:

            sheet.write(row, col, average_value_list[metric])
            col += 1
        row += 1

        #comparison_data = comparison(comparison_data, average_value_list, spectrum_expression_type)

    #return comparison_data


def summary_pbl(all_bugs_file, summary_file, prefix):
    summary_file_dir = join_path(EXPERIMENT_RESULT_FOLDER,
                                 summary_file)
    wb = Workbook(summary_file_dir)
    sheets = []
    num_sheet =0
    for spectrum_expression_type in [TARANTULA, OP2, OCHIAI, BARINEL, DSTAR]:
        sheets.append(wb.add_worksheet(spectrum_expression_type))
        sheet = sheets[num_sheet]
        num_sheet += 1
        row = 0
        col = 0
        sheet.write(row, col, "NUM OF EXAMED STMS")
        col += 1
        for item in rank_column:
            sheet.write(row, col, item)
            col += 1
        row += 1
        for num_stm in range(1, 11):
            col = 0
            sheet.write(row, col, num_stm)
            col = +1
            average_value_list = percentage_of_bugs_found_per_case(all_bugs_file, spectrum_expression_type, num_stm)
            for metric in rank_column:
                 sheet.write(row, col, average_value_list[metric])
                 col += 1
            row += 1

    wb.close()

def summary_percentage_of_cases_found_bugs(all_bugs_file, summary_file, prefix):
    summary_file_dir = join_path(EXPERIMENT_RESULT_FOLDER,
                                 summary_file)
    wb = Workbook(summary_file_dir)
    sheets = []
    num_sheet =0
    for spectrum_expression_type in [TARANTULA, OP2, OCHIAI, BARINEL, DSTAR]:
        sheets.append(wb.add_worksheet(spectrum_expression_type))
        sheet = sheets[num_sheet]
        num_sheet += 1
        row = 0
        col = 0
        sheet.write(row, col, "NUM OF EXAMED STMS")
        col += 1
        for item in rank_column:
            sheet.write(row, col, item)
            col += 1
        row += 1
        for num_stm in range(1, 11):
            col = 0
            sheet.write(row, col, num_stm)
            col = +1
            average_value_list = percentage_of_cases_found_bugs(all_bugs_file, spectrum_expression_type, num_stm)

            for metric in rank_column:
                sheet.write(row, col, average_value_list[metric])
                col += 1
            row += 1

    wb.close()


MAX = 100000
MIN = -100000


def percentage_of_bugs_found_per_case(experimental_file_dir, sbfl_metric, num_examined_stm):
    excel_data_df = pandas.read_excel(experimental_file_dir, sheet_name=None)
    index = 0
    percentage_list = {}
    for metric in rank_column:
        percentage_list[metric] = []
    while index < len(excel_data_df[sbfl_metric][BUG_ID]):
        data, index = get_values_of_a_case(excel_data_df, sbfl_metric, index, rank_column)
        for metric in rank_column:
            temp = 0
            for item in data[metric]:
                if item <= num_examined_stm:
                    temp += 1
            percentage_list[metric].append(temp / len(data[metric]))

    average_percentage_list = {}
    for metric in rank_column:
        average_percentage_list[metric] = calculate_average(percentage_list[metric])
    return average_percentage_list


def percentage_of_cases_found_bugs(experimental_file_dir, sbfl_metric, num_examined_stm):
    excel_data_df = pandas.read_excel(experimental_file_dir, sheet_name=None)
    index = 0
    percentage_list = {}
    for metric in rank_column:
        percentage_list[metric] = 0
    while index < len(excel_data_df[sbfl_metric][BUG_ID]):
        data, index = get_values_of_a_case(excel_data_df, sbfl_metric, index, rank_column)
        for metric in rank_column:
            for item in data[metric]:
                if item <= num_examined_stm:
                    percentage_list[metric] += 1
                    break

    average_percentage_list = {}
    num_of_cases = num_of_element(excel_data_df[sbfl_metric][BUG_ID])
    for metric in rank_column:
        average_percentage_list[metric] = percentage_list[metric] / num_of_cases
    return average_percentage_list


def calculate_average(data):
    sum = 0
    for item in data:
        sum += item
    return sum / len(data)


def average_best_rank_exam(excel_data_df, sbfl_metric):
    best_value_list = get_best_rank_exam(excel_data_df, sbfl_metric)
    average_value_list = {}
    for metric in data_column:
        average_value_list[metric] = calculate_average(best_value_list[metric])
    return average_value_list


def average_worst_rank_exam(excel_data_df, sbfl_metric):
    best_value_list = get_worst_rank_exam(excel_data_df, sbfl_metric)
    average_value_list = {}
    for metric in data_column:
        average_value_list[metric] = calculate_average(best_value_list[metric])
    return average_value_list


def get_best_rank_exam(excel_data_df, sbfl_metric):
    index = 0
    best_value_list = {}
    for metric in data_column:
        best_value_list[metric] = []

    while index < len(excel_data_df[sbfl_metric][BUG_ID]):
        data, index = get_values_of_a_case(excel_data_df, sbfl_metric, index, data_column)
        for metric in data_column:
            best_value_list[metric].append(min(data[metric]))
    return best_value_list


def get_worst_rank_exam(excel_data_df, sbfl_metric):
    index = 0
    best_value_list = {}
    for metric in data_column:
        best_value_list[metric] = []

    while index < len(excel_data_df[sbfl_metric][BUG_ID]):
        data, index = get_values_of_a_case(excel_data_df, sbfl_metric, index, data_column)
        for metric in data_column:
            best_value_list[metric].append(max(data[metric]))
    return best_value_list


def get_values_of_a_case(excel_data_df, sbfl_metric, index, column_list):
    list_of_values = {}
    for metric in column_list:
        list_of_values[metric] = []
        list_of_values[metric].append(excel_data_df[sbfl_metric][metric][index])
    index += 1
    while index < len(excel_data_df[sbfl_metric][BUG_ID]) and pandas.isnull(excel_data_df[sbfl_metric][BUG_ID][index]):
        for metric in column_list:
            list_of_values[metric].append(excel_data_df[sbfl_metric][metric][index])
        index += 1
    return list_of_values, index
