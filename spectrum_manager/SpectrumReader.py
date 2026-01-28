import logging
import os

from FileManager import join_path, list_dir, get_failed_test_coverage_dir, get_variant_dir, get_test_coverage_dir, \
    SPECTRUM_FAILED_COVERAGE_FILE_NAME, SPECTRUM_PASSED_COVERAGE_FILE_NAME, get_all_variant_dirs, \
    get_passed_test_coverage_dir, get_spectrum_failed_coverage_file_name_with_version, \
    get_spectrum_passed_coverage_file_name_with_version, get_variants_dir, PASSED_TEST_COVERAGE_FOLDER_NAME, \
    FAILED_TEST_COVERAGE_FOLDER_NAME
import xml.etree.ElementTree as ET

# from TestingCoverageManager import statement_coverage
from consistent_testing_manager.DDU import create_activity_matrix_system_level, ddu_system_level
from ranking.Keywords import FAILED_TEST_COUNT, PASSED_TEST_COUNT


def get_all_stm_ids(project_dir):
    system_stm_ids = {}
    variants = get_all_variant_dirs(project_dir)
    for variant in variants:
        # variant_name = variant.split("/")[-1]
        variant_name = variant.split("\\")[-1]
        system_stm_ids[variant_name] = get_stm_ids_per_variant(variant)
    return system_stm_ids


def get_stm_ids_per_variant(variant_dir, fp_variant = False):
    coverage_dir = get_test_coverage_dir(variant_dir)
    if fp_variant:
        spectrum_dirs = [join_path(coverage_dir, SPECTRUM_PASSED_COVERAGE_FILE_NAME)]
    else:
        spectrum_dirs = [join_path(coverage_dir, SPECTRUM_FAILED_COVERAGE_FILE_NAME),
                     join_path(coverage_dir, SPECTRUM_PASSED_COVERAGE_FILE_NAME)]
    data = {}
    for cfile in spectrum_dirs:
        if os.path.isfile(cfile):
            try:
                tree = ET.parse(cfile)
                root = tree.getroot()
                project = root.find("project")
                for package in project:
                    for file in package:
                        file_name = file.get("path")
                        for line in file:
                            id = line.get('featureClass') + "." + line.get('featureLineNum')
                            key = file_name + "." + line.get('num')
                            if key not in data:
                                data[key] = {}
                                data[key]['id'] = id
                                data[key]['tested'] = 0
                                if int(line.get('count')) != 0:
                                    data[key]['tested'] = 1
                            else:
                                if data[key]['tested'] == 0 and int(line.get('count')) != 0:
                                    data[key]['tested'] = 1
            except:
                logging.info("Exception when parsing %s", cfile)
    return data


def get_failings_executions(project_dir, system_stm_ids, failing_variants):
    failed_test_executions = {}
    for variant in failing_variants:
        failed_test_executions[variant] = {}
        variant_dir = get_variant_dir(project_dir, variant)
        failed_tests_coverage_dir = get_failed_test_coverage_dir(variant_dir)
        all_failed_spectrums = list_dir(failed_tests_coverage_dir)
        for file in all_failed_spectrums:
            file_dir = join_path(failed_tests_coverage_dir, file)
            failed_test_executions[variant][file] = read_coverage_file(system_stm_ids[variant], file_dir)
    return failed_test_executions


def read_coverage_file(system_stm_ids, coverage_file_dir):
    data = []
    if os.path.isfile(coverage_file_dir):
        try:
            tree = ET.parse(coverage_file_dir)
            root = tree.getroot()
            project = root.find("project")
            packages = project.findall("package")
            for package in packages:
                files = package.findall("file")
                for file in files:
                    file_path = file.get("path").split("/")
                    src_id = file_path.index("src")
                    file_name = "/".join(file_path[src_id + 1:])
                    lines = file.findall("line")
                    for line in lines:
                        id = file_name + "." + line.get('num')
                        if "count" in line.attrib and int(line.get("count")) > 0:
                            if id in system_stm_ids.keys():
                                if system_stm_ids[id] not in data:
                                    data.append(system_stm_ids[id])
        except:
            logging.info("Exception when parsing %s", coverage_file_dir)
    return data


def similar_path(path1, path2, threshold):
    count = 0
    for item in path1:
        if item in path2:
            count += 1
    if len(path1) != 0 and count / len(path1) > threshold:
        return True
    return False


def get_infor_for_sbfl(mutated_project_dir, failing_variants, fp_variants, spectrum_coverage_prefix,
                       coverage_rate, add_more_tests=False):

    total_failed_tests = 0
    total_passed_tests = 0
    stm_info_for_spectrum = {}
    variants_list = get_all_variant_dirs(mutated_project_dir)
    for variant_dir in variants_list:
        test_coverage_dir = get_test_coverage_dir(variant_dir)
        spectrum_failed_file = get_spectrum_failed_coverage_file_name_with_version(spectrum_coverage_prefix)
        spectrum_failed_coverage_file_dir = join_path(test_coverage_dir, spectrum_failed_file)
        spectrum_passed_file = get_spectrum_passed_coverage_file_name_with_version(spectrum_coverage_prefix)
        spectrum_passed_coverage_file_dir = join_path(test_coverage_dir, spectrum_passed_file)

        # if variant is a passing variant and stm_coverage < coverage_rate
        if not os.path.isfile(
                spectrum_failed_coverage_file_dir) and coverage_rate != 0:
            continue

        variant_dir = variant_dir.replace('\\', '/')
        # 当前配置系统是失败系统
        if variant_dir.split("/")[-1] not in fp_variants:

            if variant_dir.split("/")[-1] in failing_variants and os.path.isfile(spectrum_failed_coverage_file_dir):
                stm_info_for_spectrum = read_coverage_info_for_spectrum(stm_info_for_spectrum,
                                                                        spectrum_failed_coverage_file_dir,
                                                                        FAILED_TEST_COUNT)

            if os.path.isfile(spectrum_passed_coverage_file_dir):
                stm_info_for_spectrum = read_coverage_info_for_spectrum(stm_info_for_spectrum,
                                                                        spectrum_passed_coverage_file_dir,
                                                                        PASSED_TEST_COUNT)
            test_coverage_dir = test_coverage_dir.replace('\\', '/')
            ftests, ptests = count_tests_by_labeled_variants(test_coverage_dir, failing_variants,
                                                             spectrum_coverage_prefix)
            total_failed_tests += ftests
            total_passed_tests += ptests
        elif add_more_tests and variant_dir.split("/")[-1] in fp_variants and os.path.isfile(
                spectrum_failed_coverage_file_dir):

            stm_info_for_spectrum = read_coverage_info_for_spectrum(stm_info_for_spectrum,
                                                                    spectrum_failed_coverage_file_dir,
                                                                    FAILED_TEST_COUNT)
            if os.path.isfile(spectrum_passed_coverage_file_dir):
                stm_info_for_spectrum = read_coverage_info_for_spectrum(stm_info_for_spectrum,
                                                                        spectrum_passed_coverage_file_dir,
                                                                        PASSED_TEST_COUNT)

            ftests, ptests = count_tests_by_labeled_variants(test_coverage_dir, [variant_dir.split("/")[-1]],
                                                             spectrum_coverage_prefix)
            total_failed_tests += ftests
            total_passed_tests += ptests

    return stm_info_for_spectrum, total_passed_tests, total_failed_tests

# fp detection的结果
def get_infor_for_sbfl_with_FP_detection(mutated_project_dir, failing_variants, fp_variants, sups_in_variants,
                                         add_more_tests,
                                         keep_useful_tests,
                                         spectrum_coverage_prefix, coverage_rate):

    stm_info_for_spectrum, total_passed_tests, total_failed_tests = get_infor_for_sbfl(mutated_project_dir,
                                                                                       failing_variants,
                                                                                       fp_variants,
                                                                                       spectrum_coverage_prefix,
                                                                                       coverage_rate, add_more_tests)

    if keep_useful_tests:
        stm_info_for_spectrum, total_passed_tests = get_useful_tests_in_FP_variants(mutated_project_dir,
                                                                                    fp_variants, sups_in_variants,
                                                                                    stm_info_for_spectrum,
                                                                                    total_passed_tests)
    return stm_info_for_spectrum, total_passed_tests, total_failed_tests


def init_activity_matrix_for_ddu(mutated_project_dir, failing_variants, fp_variants):
    variants_list = get_all_variant_dirs(mutated_project_dir)
    A = {}
    for variant_dir in variants_list:
        if variant_dir.split("/")[-1] not in fp_variants:
            coverage_dir = get_test_coverage_dir(variant_dir)
            if variant_dir.split("/")[-1] in failing_variants:
                pass_fail_dirs = [join_path(coverage_dir, PASSED_TEST_COVERAGE_FOLDER_NAME),
                                  join_path(coverage_dir, FAILED_TEST_COVERAGE_FOLDER_NAME)]
            else:
                pass_fail_dirs = [join_path(coverage_dir, PASSED_TEST_COVERAGE_FOLDER_NAME)]
            for path in pass_fail_dirs:
                if os.path.isdir(path):
                    all_coverage_files = list_dir(path)
                    for file in all_coverage_files:
                        file_path = join_path(path, file)
                        A = create_activity_matrix_system_level(A, file_path)
    return A


def get_passing_tests_by_local_suspiciousness(passing_executions, sups_in_variants, stm_info_for_spectrum,
                                              total_passed_tests):
    score_list = []
    most_suspicious_stmt = []
    for stm in sups_in_variants:
        score_list.append(sups_in_variants[stm])
    score_list.sort(reverse=True)
    tscore = score_list[int(len(score_list) * 0.1)]

    for stm in sups_in_variants:
        if sups_in_variants[stm] > tscore:
            most_suspicious_stmt.append(stm)

    for v in passing_executions:
        for test in passing_executions[v]:
            flag = False
            for item in passing_executions[v][test]:
                if item["id"] in most_suspicious_stmt:
                    flag = True
                    break
            if not flag:
                for item in passing_executions[v][test]:
                    if int(item["tested"]) == 1:
                        if item["id"] in stm_info_for_spectrum:
                            stm_info_for_spectrum[item["id"]]['passed_test_count'] += 1
                total_passed_tests += 1
    return stm_info_for_spectrum, total_passed_tests


def get_useful_tests_in_FP_variants(mutated_project_dir, fp_variants, sups_in_variants,
                                    stm_info_for_spectrum,
                                    total_passed_tests):
    system_stm_ids = get_all_stm_ids(mutated_project_dir)
    passing_executions = get_passing_executions(mutated_project_dir, system_stm_ids, fp_variants)

    return get_passing_tests_by_local_suspiciousness(passing_executions, sups_in_variants, stm_info_for_spectrum,
                                                     total_passed_tests)


def contain_suspcious_stmts(path, failed_executions):
    passing_execution_stmt = []
    for item in path:
        if item["id"] not in passing_execution_stmt:
            passing_execution_stmt.append(item["id"])
    for v in failed_executions:
        for t in failed_executions[v]:
            for item in failed_executions[v][t]:
                if item["id"] in passing_execution_stmt:
                    return True
    return False


def get_passing_executions(project_dir, system_stm_ids, variants):
    passed_test_executions = {}
    for variant in variants:
        passed_test_executions[variant] = {}
        variant_dir = get_variant_dir(project_dir, variant)
        passed_tests_coverage_dir = get_passed_test_coverage_dir(variant_dir)
        all_passed_spectrums = list_dir(passed_tests_coverage_dir)
        for file in all_passed_spectrums:
            file_dir = join_path(passed_tests_coverage_dir, file)
            passed_test_executions[variant][file] = read_coverage_file(system_stm_ids[variant], file_dir)
    return passed_test_executions


def get_passing_executions_in_a_variant(project_dir, system_stm_ids, variant):
    passed_test_executions = {}
    variant_dir = get_variant_dir(project_dir, variant)
    variant_dir = get_variant_dir(project_dir, variant_dir)
    passed_tests_coverage_dir = get_passed_test_coverage_dir(variant_dir)
    all_passed_spectrums = list_dir(passed_tests_coverage_dir)
    for file in all_passed_spectrums:
        file_dir = join_path(passed_tests_coverage_dir, file)
        passed_test_executions[file] = read_coverage_file(system_stm_ids[variant], file_dir)
    return passed_test_executions


def get_suspicious_space_consistent_version(mutated_project_dir, failing_variants, filtering_coverage_rate,
                                            coverage_version):
    all_stms_of_the_system, all_stms_in_failing_products = get_executed_stms_of_the_system_consistent_testing_version(
        mutated_project_dir, failing_variants, coverage_version, filtering_coverage_rate)

    return all_stms_in_failing_products


def get_executed_stms_of_the_system_consistent_testing_version(mutated_project_dir, failing_variants,
                                                               spectrum_coverage_prefix, coverage_rate):
    variants_dir = get_variants_dir(mutated_project_dir)
    variants_list = list_dir(variants_dir)
    all_stms_in_system = {}
    all_stms_in_failing_product = {}
    for variant in variants_list:
        variant_dir = get_variant_dir(mutated_project_dir, variant)
        test_coverage_dir = get_test_coverage_dir(variant_dir)
        spectrum_failed_file = get_spectrum_failed_coverage_file_name_with_version(spectrum_coverage_prefix)
        if variant in failing_variants:
            failed_file = join_path(test_coverage_dir, spectrum_failed_file)
        else:
            failed_file = ""
        spectrum_passed_file = get_spectrum_passed_coverage_file_name_with_version(spectrum_coverage_prefix)
        passed_file = join_path(test_coverage_dir, spectrum_passed_file)

        # if variant is a passing coverage and statement coverage is less than the coverage rate
        if not os.path.isfile(failed_file) and coverage_rate != 0:
            continue

        coverage_files = [failed_file, passed_file]
        all_stms_in_system, all_stms_in_failing_product = get_executed_stms_by_coverage_files(variant, coverage_files,
                                                                                              all_stms_in_system,
                                                                                              all_stms_in_failing_product)
    return all_stms_in_system, all_stms_in_failing_product


def get_executed_stms_by_coverage_files(variant, coverage_files, all_stms_in_system, all_stms_in_failing_product):
    for c_file in coverage_files:
        if os.path.isfile(c_file):
            data = {}
            try:
                tree = ET.parse(c_file)
                root = tree.getroot()
                project = root.find("project")

                for package in project:
                    for file in package:
                        for line in file:
                            id = line.get('featureClass') + "." + line.get('featureLineNum')
                            if id not in data:
                                data[id] = {'num_interactions': 0}
            except:
                logging.info("Exception when parsing %s", c_file)
            all_stms_in_system[variant] = data
            if coverage_files.index(c_file) == 0:
                all_stms_in_failing_product[variant] = data

    return all_stms_in_system, all_stms_in_failing_product


def get_executed_stms_of_the_system(mutated_project_dir, failing_variants, FP_variants, add_more_tests,
                                    spectrum_coverage_prefix, coverage_rate):
    variants_dir = get_variants_dir(mutated_project_dir)
    variants_list = list_dir(variants_dir)
    all_stms_in_system = {}
    all_stms_in_failing_product = {}
    for variant in variants_list:
        variant_dir = get_variant_dir(mutated_project_dir, variant)
        test_coverage_dir = get_test_coverage_dir(variant_dir)
        spectrum_failed_file = get_spectrum_failed_coverage_file_name_with_version(spectrum_coverage_prefix)

        if variant not in FP_variants:
            if variant in failing_variants:
                failed_file = join_path(test_coverage_dir, spectrum_failed_file)
            else:
                failed_file = ""
            spectrum_passed_file = get_spectrum_passed_coverage_file_name_with_version(spectrum_coverage_prefix)
            passed_file = join_path(test_coverage_dir, spectrum_passed_file)

            # if variant is a passing coverage and statement coverage is less than the coverage rate
            if not os.path.isfile(failed_file) and coverage_rate != 0:
                continue

            coverage_files = [failed_file, passed_file]

            all_stms_in_system, all_stms_in_failing_product = get_executed_stms_by_coverage_files(variant, coverage_files,
                                                                                                  all_stms_in_system,
                                                                                                  all_stms_in_failing_product)
        elif add_more_tests and variant in FP_variants:
            failed_file = join_path(test_coverage_dir, spectrum_failed_file)
            if os.path.isfile(failed_file):
                spectrum_passed_file = get_spectrum_passed_coverage_file_name_with_version(spectrum_coverage_prefix)
                passed_file = join_path(test_coverage_dir, spectrum_passed_file)
                coverage_files = [failed_file, passed_file]

                all_stms_in_system, all_stms_in_failing_product = get_executed_stms_by_coverage_files(variant, coverage_files,
                                                                                                  all_stms_in_system,
                                                                                                  all_stms_in_failing_product)
    return all_stms_in_system, all_stms_in_failing_product


def read_coverage_info_for_spectrum(statement_infor, coverage_file, kind_of_test_count):
    data = {}
    try:
        tree = ET.parse(coverage_file)
        root = tree.getroot()
        project = root.find("project")

        for package in project:
            for file in package:
                for line in file:
                    id = line.get('featureClass') + "." + line.get('featureLineNum')
                    if id not in data:
                        data[id] = {}
                        data[id][FAILED_TEST_COUNT] = 0
                        data[id][PASSED_TEST_COUNT] = 0
                    data[id][kind_of_test_count] = max(int(line.get('count')),
                                                       data[id][kind_of_test_count])

        for id in data.keys():
            if id not in statement_infor:
                statement_infor[id] = data[id]
            else:
                statement_infor[id][kind_of_test_count] += data[id][kind_of_test_count]

        return statement_infor
    except:
        logging.info("Exception when parsing %s", coverage_file)


def count_tests(dir, spectrum_coverage_prefix):
    spectrum_failed_file = get_spectrum_failed_coverage_file_name_with_version(spectrum_coverage_prefix)
    spectrum_failed_coverage_file_dir = join_path(dir, spectrum_failed_file)
    spectrum_passed_file = get_spectrum_passed_coverage_file_name_with_version(spectrum_coverage_prefix)
    spectrum_passed_coverage_file_dir = join_path(dir, spectrum_passed_file)

    num_of_failed_tests = 0
    num_of_passed_tests = 0
    if os.path.isfile(spectrum_failed_coverage_file_dir):
        num_of_failed_tests = count_test_in_file(spectrum_failed_coverage_file_dir)

    if os.path.isfile(spectrum_passed_coverage_file_dir):
        num_of_passed_tests = count_test_in_file(spectrum_passed_coverage_file_dir)

    return num_of_failed_tests, num_of_passed_tests


def count_tests_by_labeled_variants(dir, failing_variants, spectrum_coverage_prefix):
    spectrum_failed_file = get_spectrum_failed_coverage_file_name_with_version(spectrum_coverage_prefix)
    spectrum_failed_coverage_file_dir = join_path(dir, spectrum_failed_file)
    spectrum_passed_file = get_spectrum_passed_coverage_file_name_with_version(spectrum_coverage_prefix)
    spectrum_passed_coverage_file_dir = join_path(dir, spectrum_passed_file)

    num_of_failed_tests = 0
    num_of_passed_tests = 0
    if dir.split("/")[-2] in failing_variants and os.path.isfile(spectrum_failed_coverage_file_dir):
        num_of_failed_tests = count_test_in_file(spectrum_failed_coverage_file_dir)

    if os.path.isfile(spectrum_passed_coverage_file_dir):
        num_of_passed_tests = count_test_in_file(spectrum_passed_coverage_file_dir)

    return num_of_failed_tests, num_of_passed_tests


def count_test_in_file(file_dir):
    try:
        tree = ET.parse(file_dir)
        root = tree.getroot()
        project = root.find("tests")
        return int(project.get("count"))
    except:
        logging.info("Exception when parsing %s", file_dir)
