import os

from xlsxwriter import Workbook

from consistent_testing_manager.FPMatricsCaculation import FALSE_PASSING, LABEL
from consistent_testing_manager.FileName import *
from ranking import RankingManager
from ranking.FeaturesRankingManager import features_ranking_multiple_bugs
from ranking.Keywords import *
from ranking.RankingManager import VARCOP_RANK, SBFL_RANK, \
    ranking_multiple_bugs, VARCOP_SPACE, SPACE, get_set_of_stms
from FileManager import join_path, EXPERIMENT_RESULT_FOLDER, list_dir, get_spc_log_file_path, get_slicing_log_file_path, \
    get_outer_dir
from ranking.VarBugManager import is_var_bug_by_config
from spc import SPCsManager
from spectrum_manager.SpectrumReader import get_executed_stms_of_the_system
from suspicious_statements_manager import SlicingManager
from suspicious_statements_manager.SuspiciousStatementManager import get_multiple_buggy_statements, \
    get_suspicious_statement_varcop, get_suspicious_statement_tc_based
from csv import reader

BUG_ID_COL = 0
VARCOP_BUGGY_STM_COL = 1
VARCOP_RANK_COL = 2
VARCOP_EXAM_COL = 3
VARCOP_SPACE_COL = 4
# VARCOP_TC_SLICED_RANK_COL = 5
# VARCOP_TC_SLICED_EXAM_COL = 6
# SBFL_TC_SLICED_RANK_COL = 7
# SBFL_TC_SLICED_EXAM_COL = 8
# FB_TC_SLICED_RANK_COL = 9
# FB_TC_SLICED_EXAM_COL = 10
# TC_SLICED_SPACE_COL = 11
VARCOP_DISABLE_BPC_RANK_COL = 5
VARCOP_DISABLE_BPC_EXAM_COL = 6
SBFL_RANK_COL = 7
SBFL_EXAM_COL = 8
FB_RANK_COL = 9
FB_EXAM_COL = 10
SPACE_COL = 11
IS_VAR_BUG_COL = 12


def write_header_in_result_file(row, sheet):
    sheet.write(row, BUG_ID_COL, BUG_ID)
    sheet.write(row, VARCOP_BUGGY_STM_COL, BUGGY_STM)

    sheet.write(row, VARCOP_RANK_COL, VARCOP_RANK)
    sheet.write(row, VARCOP_EXAM_COL, VARCOP_EXAM)
    sheet.write(row, VARCOP_SPACE_COL, VARCOP_SPACE)

    # sheet.write(row, VARCOP_TC_SLICED_RANK_COL, VARCOP_TC_RANK)
    # sheet.write(row, VARCOP_TC_SLICED_EXAM_COL, VARCOP_TC_EXAM)
    #
    # sheet.write(row, SBFL_TC_SLICED_RANK_COL, SBFL_TC_RANK)
    # sheet.write(row, SBFL_TC_SLICED_EXAM_COL, SBFL_TC_EXAM)
    #
    # sheet.write(row, FB_TC_SLICED_RANK_COL, FB_TC_RANK)
    # sheet.write(row, FB_TC_SLICED_EXAM_COL, FB_TC_EXAM)
    #
    # sheet.write(row, TC_SLICED_SPACE_COL, TC_SPACE)

    sheet.write(row, VARCOP_DISABLE_BPC_RANK_COL, VARCOP_DISABLE_BPC_RANK)
    sheet.write(row, VARCOP_DISABLE_BPC_EXAM_COL, VARCOP_DISABLE_BPC_EXAM)

    sheet.write(row, SBFL_RANK_COL, SBFL_RANK)
    sheet.write(row, SBFL_EXAM_COL, SBFL_EXAM)

    sheet.write(row, FB_RANK_COL, FB_RANK)
    sheet.write(row, FB_EXAM_COL, FB_EXAM)

    sheet.write(row, SPACE_COL, SPACE)

    # sheet.write(row, IS_VAR_BUG_COL, "IS_VAR_BUG")


def write_result_to_file(row, sheet, ranking_results, fb_results, search_spaces, is_var_bug):
    varcop_space = len(get_set_of_stms(search_spaces[SS_VARCOP]))
    sliced_space = len(get_set_of_stms(search_spaces[SS_SLICING]))
    all_space = len(get_set_of_stms(search_spaces[SS_STMS_IN_F_PRODUCTS]))
    all_stms = len(get_set_of_stms(search_spaces[SS_ALL_STMS]))

    for stm in ranking_results[SBFL_RANK].keys():
        sheet.write(row, VARCOP_BUGGY_STM_COL, stm)

        if is_var_bug:
            sheet.write(row, VARCOP_RANK_COL, ranking_results[VARCOP_RANK][stm][RANK])
            sheet.write(row, VARCOP_EXAM_COL, (ranking_results[VARCOP_RANK][stm][RANK] / all_stms) * 100)
        else:
            sheet.write(row, VARCOP_RANK_COL, ranking_results[VARCOP_DISABLE_BPC_RANK][stm][RANK])
            sheet.write(row, VARCOP_EXAM_COL, (ranking_results[VARCOP_DISABLE_BPC_RANK][stm][RANK] / all_stms) * 100)
            sheet.write(row, IS_VAR_BUG_COL, 0)

        sheet.write(row, VARCOP_SPACE_COL, varcop_space)

        # sheet.write(row, VARCOP_TC_SLICED_RANK_COL, ranking_results[VARCOP_TC_RANK][stm][RANK])
        # sheet.write(row, VARCOP_TC_SLICED_EXAM_COL, (ranking_results[VARCOP_TC_RANK][stm][RANK] / all_stms) * 100)
        # sheet.write(row, SBFL_TC_SLICED_RANK_COL, ranking_results[SBFL_TC_RANK][stm][RANK])
        # sheet.write(row, SBFL_TC_SLICED_EXAM_COL, (ranking_results[SBFL_TC_RANK][stm][RANK] / all_stms) * 100)
        # sheet.write(row, FB_TC_SLICED_RANK_COL, fb_results[FB_TC_RANK][stm][RANK])
        # sheet.write(row, FB_TC_SLICED_EXAM_COL, (fb_results[FB_TC_RANK][stm][RANK] / all_stms) * 100)
        #
        # sheet.write(row, TC_SLICED_SPACE_COL, sliced_space)

        sheet.write(row, VARCOP_DISABLE_BPC_RANK_COL, ranking_results[VARCOP_DISABLE_BPC_RANK][stm][RANK])
        sheet.write(row, VARCOP_DISABLE_BPC_EXAM_COL,
                    (ranking_results[VARCOP_DISABLE_BPC_RANK][stm][RANK] / all_stms) * 100)
        sheet.write(row, SBFL_RANK_COL, ranking_results[SBFL_RANK][stm][RANK])
        sheet.write(row, SBFL_EXAM_COL, (ranking_results[SBFL_RANK][stm][RANK] / all_stms) * 100)
        # sheet.write(row, FB_RANK_COL, fb_results[FB_RANK][stm][RANK])
        # sheet.write(row, FB_EXAM_COL, (fb_results[FB_RANK][stm][RANK] / all_stms) * 100)
        sheet.write(row, SPACE_COL, all_space)

        row += 1
    return row


def suspicious_isolation(mutated_project_dir, failing_variants, FP_variants, add_more_tests, filtering_coverage_rate,
                         coverage_version,
                         spc_postfix=""):
    SPCsManager.find_SPCs(mutated_project_dir, failing_variants, FP_variants, add_more_tests, filtering_coverage_rate,
                          spc_postfix)
    spc_log_file_path = get_spc_log_file_path(mutated_project_dir, str(filtering_coverage_rate) + spc_postfix)
    SlicingManager.do_slice_spc(spc_log_file_path, filtering_coverage_rate, coverage_version, spc_postfix)


def get_suspicious_space(mutated_project_dir, failing_variants, FP_variants, add_more_tests, filtering_coverage_rate,
                         coverage_version, spc_postfix=""):
    spc_log_file_path = get_spc_log_file_path(mutated_project_dir, str(filtering_coverage_rate) + spc_postfix)
    if coverage_version != "":
        slicing_postfix = str(filtering_coverage_rate) + "_" + coverage_version + "_"
    else:
        slicing_postfix = filtering_coverage_rate

    slicing_output_path = get_slicing_log_file_path(get_outer_dir(spc_log_file_path),
                                                    str(slicing_postfix) + spc_postfix)
    if not os.path.isfile(slicing_output_path):
        slicing_output_path = join_path(mutated_project_dir, "slicing_0.log")
    stms_isolated_by_varcop = get_suspicious_statement_varcop(mutated_project_dir,
                                                              slicing_output_path)

    tc_sliced_based_isolation = get_suspicious_statement_tc_based(mutated_project_dir)
    all_stms_of_the_system, all_stms_in_failing_products = get_executed_stms_of_the_system(
        mutated_project_dir, failing_variants, FP_variants, add_more_tests, coverage_version, filtering_coverage_rate)
    search_spaces = {SS_VARCOP: stms_isolated_by_varcop, SS_SLICING: tc_sliced_based_isolation,
                     SS_ALL_STMS: all_stms_of_the_system,
                     SS_STMS_IN_F_PRODUCTS: all_stms_in_failing_products}
    return search_spaces


def create_exp_result_folder(result_folder, system_name):
    result_folder_dir = join_path(EXPERIMENT_RESULT_FOLDER, result_folder)
    if not os.path.exists(result_folder_dir):
        os.makedirs(result_folder_dir)

    system_result_dir = join_path(result_folder_dir, system_name)
    if not os.path.exists(system_result_dir):
        os.makedirs(system_result_dir)
    return system_result_dir


def multiple_bugs_ranking(result_folder, system_name, system_dir, num_of_bugs, kwise, spectrum_expressions,
                          FP_detection, classified_file_name, alpha=0.5, add_more_tests=False, keep_useful_tests=False,
                          filtering_coverage_rate=0.0, coverage_version=""):
    aggregations = [RankingManager.AGGREGATION_ARITHMETIC_MEAN]
    normalizations = [RankingManager.NORMALIZATION_ALPHA_BETA]

    if not os.path.exists(system_dir):
        return

    mutated_projects = list_dir(system_dir)
    system_result_dir = create_exp_result_folder(result_folder, system_name)

    for normalization_type in normalizations:
        normalization_result_dir = join_path(system_result_dir, normalization_type)
        if not os.path.exists(normalization_result_dir):
            os.makedirs(normalization_result_dir)

        for aggregation_type in aggregations:
            aggregation_result_dir = join_path(normalization_result_dir, aggregation_type)
            if not os.path.exists(aggregation_result_dir):
                os.makedirs(aggregation_result_dir)

            kwise_result_dir = join_path(aggregation_result_dir, kwise)
            if not os.path.exists(kwise_result_dir):
                os.makedirs(kwise_result_dir)

            sheet = []

            row = 0

            experiment_file_name = join_path(kwise_result_dir,
                                             num_of_bugs + coverage_version + ".xlsx")

            wb = Workbook(experiment_file_name)

            for i in range(0, len(spectrum_expressions)):
                sheet.append(wb.add_worksheet(spectrum_expressions[i]))
                write_header_in_result_file(row, sheet[i])
            row += 1
            num_of_bugs = 0

            for mutated_project_name in mutated_projects:
                mutated_project_dir = join_path(system_dir, mutated_project_name)
                classified_file_path = join_path(mutated_project_dir, classified_file_name)
                label_file = join_path(mutated_project_dir, variant_labels)
                if not os.path.isfile(classified_file_path) or not os.path.isfile(label_file):
                    continue

                print(mutated_project_name)
                num_of_bugs += 1

                failing_variants = get_failing_variants_by_labels(label_file, LABEL)
                if FP_detection and not add_more_tests:
                    FP_variants = get_fp_variants(classified_file_path)
                    spc_postfix = "remove_fps"
                elif FP_detection and add_more_tests:
                    FP_variants = get_fp_variants(classified_file_path)
                    spc_postfix = "add_more_tests"
                else:
                    FP_variants = []
                    spc_postfix = "original"

                # suspicious_isolation(mutated_project_dir, failing_variants, FP_variants, add_more_tests, filtering_coverage_rate,
                #                      coverage_version, spc_postfix)
                search_spaces = get_suspicious_space(mutated_project_dir, failing_variants, FP_variants, add_more_tests,
                                                     filtering_coverage_rate, coverage_version, spc_postfix)
                buggy_statements = get_multiple_buggy_statements(mutated_project_name, mutated_project_dir)

                row_temp = row

                if system_name == "ZipMe":
                    is_a_var_bug = is_var_bug_by_config(mutated_project_dir, ["Base", "Compress"])
                elif system_name == "BankAccountTP":
                    is_a_var_bug = is_var_bug_by_config(mutated_project_dir, ["BankAccount"])
                else:
                    is_a_var_bug = is_var_bug_by_config(mutated_project_dir, ["Base"])

                ranking_results, varcop_ranking_time = ranking_multiple_bugs(buggy_statements,
                                                                             mutated_project_dir, failing_variants,
                                                                             FP_variants, add_more_tests,
                                                                             keep_useful_tests,
                                                                             search_spaces,
                                                                             spectrum_expressions,
                                                                             aggregation_type,
                                                                             normalization_type,
                                                                             coverage_version,
                                                                             filtering_coverage_rate, alpha)
                fb_ranking_results = features_ranking_multiple_bugs(buggy_statements, mutated_project_dir,
                                                                    failing_variants,
                                                                    FP_variants, add_more_tests,
                                                                    search_spaces,
                                                                    filtering_coverage_rate, spectrum_expressions)

                for metric in range(0, len(spectrum_expressions)):
                    sheet[metric].write(row_temp, BUG_ID_COL, mutated_project_name)
                    row = write_result_to_file(row_temp, sheet[metric],
                                               ranking_results[spectrum_expressions[metric]],
                                               "", search_spaces,
                                               is_a_var_bug)
            wb.close()


def write_runtime_to_file(system_result_dir, run_time, file_name):
    experiment_file_name = join_path(system_result_dir,
                                     file_name)
    if os.path.exists(experiment_file_name):
        return
    wb = Workbook(experiment_file_name)
    sheet = wb.add_worksheet("run_time")
    row = 0
    for item in run_time.keys():
        sheet.write(row, 0, item)
        col = 1
        for time in run_time[item]:
            sheet.write(row, col, time)
            col += 1
        row += 1
    wb.close()


def get_failing_variants_by_labels(label_file, column):
    f_variants = []
    count = 0
    index = 0
    with open(label_file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            if count == 0:
                index = row.index(column)
                count += 1
            else:
                if row[index] == "F":
                    f_variants.append(row[0])
    return f_variants


def get_fp_variants(classified_file):
    fp_variants = []

    with open(classified_file, 'r') as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:

            if row[-1] == FALSE_PASSING:
                fp_variants.append(row[0])
    return fp_variants


def get_not_used_variants_for_classification_by_variant(normalized_info_file,
                                                        classified_file):
    nu_variants = []
    with open(classified_file, 'r') as c_read_obj:
        c_csv_reader = reader(c_read_obj)
        for row in c_csv_reader:
            if row[-1] == FALSE_PASSING:
                nu_variants.append(row[0])

    with open(normalized_info_file, 'r') as n_read_obj:
        n_csv_reader = reader(n_read_obj)
        for row in n_csv_reader:

            if row[1] not in nu_variants and row[2] == FALSE_PASSING:
                nu_variants.append(row[1])
    return nu_variants
