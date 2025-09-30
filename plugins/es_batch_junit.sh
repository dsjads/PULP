#!/bin/sh
if [ "${1}" = "-src" ]; then
    SRC_DIR=${2};
fi;

if [ "${3}" = "-test" ]; then
    TEST_DIR=${4};
fi;

if [ "${5}" = "-build.classes" ]; then
    SRC_CLASSES_DIR=${6};
fi;

if [ "${7}" = "-build.testclasses" ]; then
    TEST_CLASSES_DIR=${8};
fi;

if [ "${9}" = "-report.coveragedir" ]; then
    # batch test will not write coverage report
    # parameter COVERAGE_DIR is unusable
    COVERAGE_DIR=${10};
fi;

if [ "${11}" = "-junit.haltonfailure" ]; then
    JUNIT_HALT_ON_FALURE=${12};
fi;

if [ "${13}" = "-ant.name" ]; then
    ANT_FOLDER_NAME=${14}
else
    ANT_FOLDER_NAME=apache-ant-1.10.7
fi;

if [ "${15}" = "-lib_path" ]; then
    LIB_PATH=${16};
fi;


BASEDIR=$(dirname "$0")
ANT_PATH="$BASEDIR/$ANT_FOLDER_NAME";
#export ANT_OPTS=-Xmx1g
$ANT_PATH/bin/ant batchtest.all -buildfile $ANT_PATH/batch_junit_build.xml -Dsrc $SRC_DIR -Dtest $TEST_DIR -Dbuild.classes $SRC_CLASSES_DIR -Dbuild.testclasses $TEST_CLASSES_DIR -Djunit.haltonfailure=$JUNIT_HALT_ON_FALURE -Dlib_path=$LIB_PATH