import SPCsManager


if __name__ == "__main__":
    mutated_project_dir = "/Users/tuanngokien/Downloads/4wise-ExamDB-FH-JML-FMB/_MultipleBugs_.NOB_2.ID_45"
    spc_log_file_path = SPCsManager.find_SPCs(mutated_project_dir, 0.0)
    # SlicingManager.do_slice(spc_file_path=spc_log_file_path, filtering_coverage_rate=0.95,
    #                         coverage_version=coverage_version)
