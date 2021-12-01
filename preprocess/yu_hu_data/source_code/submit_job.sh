#!/bin/bash
#
#SBATCH --job-name=check_dup
#SBATCH --output=res_%j.txt  # output file
#SBATCH -e res_%j.err        # File to which STDERR will be written
#SBATCH --partition=longq    # Partition to submit to
#
#SBATCH --ntasks-per-node=40
#SBATCH --nodes=1
#SBATCH --time=7-00:00         # Maximum runtime in D-HH:MM
#SBATCH --mem-per-cpu=3000   # Memory in MB per cpu allocated

POSITIONAL=()
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --n_proc)
    NPROC="$2"
    shift # past argument
    shift # past value
    ;;
    --py_exe)
    PY_EXE="$2"
    shift # past argument
    shift # past value
    ;;
    --start_ind)
    START_IND="$2"
    shift # past argument
    shift # past value
    ;;
    --end_ind)
    END_IND="$2"
    shift # past argument
    shift # past value
    ;;
    --filenames)
    FILENAMES="$2"
    shift # past argument
    shift # past value
    ;;
    --query_dir)
    QUERY_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    --big_dir)
    BIG_DIR="$2"
    shift # past argument
    shift # past value
    ;;
    --rotation)
    ROTATION="$2"
    shift # past argument
    shift # past value
    ;;
    --query_model_name)
    QUERY_MODEL_NAME="$2"
    shift # past argument
    shift # past value
    ;;
    --ref_model_name)
    REF_MODEL_NAME="$2"
    shift # past argument
    shift # past value
    ;;
    --swarm)
    SWARM=" --swarm"
    shift
    ;;
    *)    # unknown option
    POSITIONAL+=("$1") # save it in an array for later
    shift # past argument
    ;;
esac
done
set -- "${POSITIONAL[@]}" # restore positional parameters

args="--n_proc $NPROC --start_ind $START_IND --end_ind $END_IND"
args="$args --query_dir $QUERY_DIR --filenames $FILENAMES --big_dir $BIG_DIR --rotation $ROTATION"
args="$args --query_model_name $QUERY_MODEL_NAME --ref_model_name $REF_MODEL_NAME"
echo "`pwd`"
echo "$PY_EXE duplicates.py $args"
$PY_EXE duplicates.py $args
