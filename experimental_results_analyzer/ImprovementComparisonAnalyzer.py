from ranking.Keywords import *
from spectrum_manager.Spectrum_Expression import JACCARD, SORENSEN_DICE, TARANTULA, OCHIAI, OP2, BARINEL, DSTAR, ROGERS_TANIMOTO, \
    AMPLE, \
    SIMPLE_MATCHING, RUSSELL_RAO, COHEN, SCOTT, ROGOT1, GEOMETRIC_MEAN, M2, WONG1, SOKAL, DICE, HUMANN, ZOLTAR, \
    WONG2, ROGOT2, EUCLID, HAMMING, FLEISS, ANDERBERG, KULCZYNSKI2, HARMONIC_MEAN, GOODMAN

from xlsxwriter import Workbook

comparison_list = [VARCOP_WIN_RANK, VARCOP_WIN_EXAM, VARCOP_DISABLE_BPC_WIN_RANK, VARCOP_DISABLE_BPC_WIN_EXAM,
                   SBFL_WIN_VARCOP_RANK, SBFL_WIN_VARCOP_EXAM, SBFL_WIN_VARCOP_DISABLED_BPC_RANK,
                   SBFL_WIN_VARCOP_DISABLED_BPC_EXAM]

comparison_metrics = [
    [VARCOP_RANK, SBFL_RANK], [VARCOP_TC_RANK, SBFL_TC_RANK], [VARCOP_RANK, SBFL_TC_RANK],
    [VARCOP_RANK, VARCOP_TC_RANK], [VARCOP_DISABLE_BPC_RANK, SBFL_TC_RANK], [VARCOP_DISABLE_BPC_RANK, SBFL_RANK]
]

SBFL_METRIC_COL = 0
VARCOP_VS_SBFL_RANK_COL = 1
VARCOP_VS_SBFL_EXAM_COL = 2

SPECTRUM_EXPRESSIONS_LIST = [TARANTULA, OCHIAI, OP2, BARINEL, DSTAR,
                             RUSSELL_RAO, SIMPLE_MATCHING, ROGERS_TANIMOTO, AMPLE, JACCARD,
                             COHEN, SCOTT, ROGOT1, GEOMETRIC_MEAN, M2,
                             WONG1, SOKAL, SORENSEN_DICE, DICE, HUMANN, WONG2, ZOLTAR,
                             EUCLID, ROGOT2, HAMMING, FLEISS, ANDERBERG,
                             GOODMAN, HARMONIC_MEAN, KULCZYNSKI2]



def init_comparison_data():
    data = {}
    for i in range(0, len(comparison_metrics)):
        metric = comparison_metrics[i][0] + "_vs_" + comparison_metrics[i][1]
        if metric not in data:
            data[metric] = 0
    return data


def comparison(data, average_value_list, spectrum_expression):
    data[spectrum_expression] = {}
    for metrics in comparison_metrics:
        tmp = varcop_win(average_value_list[metrics[0]], average_value_list[metrics[1]])
        metric_tmp = metrics[0] + "_vs_" + metrics[1]
        data[spectrum_expression][metric_tmp] = tmp

        if tmp > 0:
            data[metric_tmp] += 1

    return data




def varcop_win(varcop, sbfl):
    return (sbfl - varcop) / sbfl




def write_comparison(sheet, comparison_data, metrics):
    systems = list(comparison_data.keys())
    row = 1
    sheet.write(row, 0, "SPFL Metric")
    row += 1
    for spectrum_expression in SPECTRUM_EXPRESSIONS_LIST:
        sheet.write(row, 0, spectrum_expression)
        row += 1
    sheet.write(row, 0, metrics[0] + "_vs_" + metrics[1])
    # row += 1
    # sheet.write(row, 0, "SBFL win")
    for index in range(0, len(systems)):
        row = 0
        sheet.write(row, 2 * index + 1, systems[index])
        row += 1
        sheet.write(row, 2 * index + 1, "RANK")
        sheet.write(row, 2 * index + 2, "EXAM")
        row += 1
        for spectrum_expression in SPECTRUM_EXPRESSIONS_LIST:
            sheet.write(row, 2 * index + 1,
                        comparison_data[systems[index]][spectrum_expression][metrics[0] + "_vs_" + metrics[1]])
            # sheet.write(row, 2 * index + 2,
            #             comparison_data[systems[index]][spectrum_expression][VARCOP_DISABLE_BPC_VS_SBFL_IN_EXAM])
            row += 1
        sheet.write(row, 2 * index + 1, comparison_data[systems[index]][metrics[0] + "_vs_" + metrics[1]])
        #sheet.write(row, 2 * index + 2, comparison_data[systems[index]][VARCOP_DISABLE_BPC_WIN_EXAM])
        # row += 1
        # sheet.write(row, 2 * index + 1, comparison_data[systems[index]][SBFL_WIN_VARCOP_DISABLED_BPC_RANK])
        # sheet.write(row, 2 * index + 2, comparison_data[systems[index]][SBFL_WIN_VARCOP_DISABLED_BPC_EXAM])


def write_comparison_data_to_file(file_path, comparison_data):
    wb = Workbook(file_path)
    sheets = []
    index = 0
    for metrics in comparison_metrics:
        sheet_name = ""
        if metrics[0] == VARCOP_RANK:
            sheet_name = "VARCOP"
        if metrics[0] == VARCOP_TC_RANK:
            sheet_name = "VARCOP_SLICING"
        if metrics[0] == VARCOP_DISABLE_BPC_RANK:
            sheet_name = "VARCOP_ALL"
        if metrics[1] == SBFL_RANK:
            sheet_name += "vs SBFL"
        if metrics[1] == SBFL_TC_RANK:
            sheet_name += "vs SBFL_SLICING"

        sheets.append(wb.add_worksheet(sheet_name))
        write_comparison(sheets[index], comparison_data, metrics)
        index += 1


    wb.close()
