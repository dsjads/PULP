import logging

from FileManager import *
import xml.etree.ElementTree as ET


def coverage_per_test(file_path):
    data = {}
    if os.path.isfile(file_path):
        try:
            tree = ET.parse(file_path)
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
                        if id not in data:
                            data[id] = 0
                        if "count" in line.attrib:
                            if int(line.get("count")) > 0:
                                data[id] = 1
                        elif "falsecount" in line.attrib:
                            if int(line.get("falsecount")) > 0:
                                data[id] = 1
                        elif "truecount" in line.attrib:
                            if int(line.get("truecount")) > 0:
                                data[id] = 1
        except:
            logging.info("Exception when parsing %s", file_path)

    return data

# 构造语句级覆盖矩阵
def create_activity_matrix_variant_level(variant_dir, variant_label):
    A = {}
    coverage_dir = get_test_coverage_dir(variant_dir)
    if variant_label == "F":
        pass_fail_dirs = [join_path(coverage_dir, PASSED_TEST_COVERAGE_FOLDER_NAME),
                          join_path(coverage_dir, FAILED_TEST_COVERAGE_FOLDER_NAME)]
    else:
        pass_fail_dirs = [join_path(coverage_dir, PASSED_TEST_COVERAGE_FOLDER_NAME)]
    for path in pass_fail_dirs:
        if os.path.isdir(path):
            all_coverage_files = list_dir(path)
            for file in all_coverage_files:
                file_path = join_path(path, file)
                A[file] = coverage_per_test(file_path)
    return A


def create_activity_matrix_system_level(A, file_path):
    A[file_path] = coverage_per_test(file_path)
    return A


def matrix_density(matrix):
    N = len(matrix)
    M = 0
    sum = 0
    for key in matrix:
        M = len(matrix[key])
        for a in matrix[key]:
            sum += matrix[key][a]
    return sum / (N * M)


def density(matrix):
    return 1 - abs(1 - 2 * matrix_density(matrix))


def same_pattern(consider_matrix):
    same_p = {}
    for key in consider_matrix:
        same_p[key] = 0
    for key in consider_matrix:
        count = 1
        if same_p[key] == 0:
            for k in consider_matrix:
                if key != k and same_p[k] == 0:
                    test_value = 0
                    for a in consider_matrix[key]:
                        if a not in consider_matrix[k]:
                            consider_matrix[k][a] = 0
                        if consider_matrix[key][a] != consider_matrix[k][a]:
                            test_value = -1
                            break
                    if test_value == 0:
                        count += 1
                        same_p[k] = -1
            if count > 1:
                same_p[key] = count
            else:
                same_p[key] = 1
    return same_p


def diversity(matrix):
    N = len(matrix)
    same_activity_tests = same_pattern(matrix)
    sum = 0
    count = 0
    for key in same_activity_tests:
        if same_activity_tests[key] > 0:
            count += 1
            sum += same_activity_tests[key] * (same_activity_tests[key] - 1)
    diversity_metric = 1 - (sum / (N * (N - 1)))
    return diversity_metric


def uniqueness(matrix):
    component_matrix = {}
    for key in matrix:
        for a in matrix[key]:
            if a not in component_matrix:
                component_matrix[a] = {}
            component_matrix[a][key] = matrix[key][a]
    identical_components = same_pattern(component_matrix)
    count = 0
    for key in identical_components:
        if identical_components[key] > 0:
            count += 1
    return count / len(identical_components)


def check_test_status(variant_dir):
    coverage_dir = get_test_coverage_dir(variant_dir)
    pass_fail_dirs = [join_path(coverage_dir, PASSED_TEST_COVERAGE_FOLDER_NAME),
                      join_path(coverage_dir, FAILED_TEST_COVERAGE_FOLDER_NAME)]
    count = 0
    for path in pass_fail_dirs:
        if os.path.isdir(path):
            count += 1
    if count == 2:
        return False
    return True


def ddu(variant_dir, variant_label):
    activity_matrix = create_activity_matrix_variant_level(variant_dir, variant_label)
    density_metric = density(activity_matrix)
    diversity_metric = diversity(activity_matrix)
    uniqueness_metric = uniqueness(activity_matrix)
    ddu = density_metric * diversity_metric * uniqueness_metric
    return ddu


def ddu_system_level(activity_matrix):
    density_metric = density(activity_matrix)
    diversity_metric = diversity(activity_matrix)
    uniqueness_metric = uniqueness(activity_matrix)
    ddu = density_metric * diversity_metric * uniqueness_metric
    return ddu


def get_inconsistent_test_by_ddu(mutated_project_dir):
    variant_dirs = get_all_variant_dirs(mutated_project_dir)
    total = 0
    ddu_set = {}
    count = 0
    for var_dir in variant_dirs:
        activity_matrix = create_activity_matrix_variant_level(var_dir)
        density_metric = density(activity_matrix)
        diversity_metric = diversity(activity_matrix)
        uniqueness_metric = uniqueness(activity_matrix)
        ddu = density_metric * diversity_metric * uniqueness_metric
        total += ddu
        count += 1
        if check_test_status(var_dir):
            ddu_set[var_dir.split("/")[-1]] = ddu
        else:
            ddu_set[var_dir.split("/")[-1]] = 1
    average_ddu = total / count
    inconsistent = []
    for v in ddu_set:
        if ddu_set[v] < average_ddu:
            inconsistent.append(v)
    return inconsistent

