import logging
import os
import xml.etree.ElementTree as ET

from FileManager import get_all_variant_dirs, get_test_coverage_dir, \
    get_spectrum_failed_coverage_file_name_with_version, get_spectrum_passed_coverage_file_name_with_version, join_path
from ranking.Keywords import VARIANT_NUM_OF_PASSES, VARIANT_NUM_OF_FAILS

from spectrum_manager.Spectrum_Expression import suspicious_score_by_sbfl_metric

VARIANT_LEVEL_SUSPICIOUSNESS_SCORE = "variant_level_suspiciousness_score"

def init_num_of_failing_passing_variants(list_of_stms):
    data = {}
    for stm in list_of_stms:
        data[stm] = {}
        data[stm][VARIANT_NUM_OF_PASSES] = 0
        data[stm][VARIANT_NUM_OF_FAILS] = 0
    return data


def read_data_from_coverage_file(data, coverage_file, variant_type):
    try:
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        project = root.find("project")
        check_duplicate = []
        for package in project:
            for file in package:
                for line in file:
                    id = line.get('featureClass') + "." + line.get('featureLineNum')
                    if id not in data:
                        data[id] = {}
                        data[id][VARIANT_NUM_OF_FAILS] = 0
                        data[id][VARIANT_NUM_OF_PASSES] = 0

                    if int(line.get('count')) > 0:
                        if id not in check_duplicate:
                            data[id][variant_type] += 1
                            check_duplicate.append(id)
        return data
    except:
        logging.info("Exception when parsing %s", coverage_file)


def get_num_passing_failing_variants(mutated_project_dir, failing_variants, fp_variants, add_more_tests, list_of_stms, spectrum_coverage_prefix):
    failing_passing_variants_of_stms = init_num_of_failing_passing_variants(list_of_stms)
    variants_list = get_all_variant_dirs(mutated_project_dir)
    total_fails = 0
    total_passes = 0
    for variant_dir in variants_list:
        test_coverage_dir = get_test_coverage_dir(variant_dir)
        spectrum_failed_file = get_spectrum_failed_coverage_file_name_with_version(spectrum_coverage_prefix)
        spectrum_failed_coverage_file_dir = join_path(test_coverage_dir, spectrum_failed_file)
        spectrum_passed_file = get_spectrum_passed_coverage_file_name_with_version(spectrum_coverage_prefix)
        spectrum_passed_coverage_file_dir = join_path(test_coverage_dir, spectrum_passed_file)
        if variant_dir.split("/")[-1] not in fp_variants:
            if variant_dir.split("/")[-1] in failing_variants:
                failing_passing_variants_of_stms = read_data_from_coverage_file(failing_passing_variants_of_stms,
                                                                        spectrum_failed_coverage_file_dir,
                                                                        VARIANT_NUM_OF_FAILS)
                total_fails += 1
            elif os.path.isfile(spectrum_passed_coverage_file_dir):
                failing_passing_variants_of_stms = read_data_from_coverage_file(failing_passing_variants_of_stms,
                                                                        spectrum_passed_coverage_file_dir,
                                                                        VARIANT_NUM_OF_PASSES)
                total_passes += 1
        elif add_more_tests and variant_dir.split("/")[-1] in fp_variants and os.path.isfile(spectrum_failed_coverage_file_dir):
            failing_passing_variants_of_stms = read_data_from_coverage_file(failing_passing_variants_of_stms,
                                                                            spectrum_failed_coverage_file_dir,
                                                                            VARIANT_NUM_OF_FAILS)
            total_fails += 1

    print(total_passes, total_fails)
    return failing_passing_variants_of_stms, total_fails, total_passes


def calculate_suspiciousness_variant_level(failing_passing_variants_of_stms, total_fails, total_passes, spectrum_expression):
    score = spectrum_expression + VARIANT_LEVEL_SUSPICIOUSNESS_SCORE
    for stm in failing_passing_variants_of_stms.keys():
        failing_passing_variants_of_stms[stm][score] = suspicious_score_by_sbfl_metric(spectrum_expression, failing_passing_variants_of_stms[stm][VARIANT_NUM_OF_FAILS],
                                                                                       failing_passing_variants_of_stms[stm][VARIANT_NUM_OF_PASSES],
                                                                                       total_fails, total_passes)

    return failing_passing_variants_of_stms