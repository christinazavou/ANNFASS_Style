#!/bin/bash

die () {
    echo >&2 "$@"
    exit 1
}

# Mandatory arguments
n_models=$1
n_jobs=$2
py_exe=$3
query_dir=$4
filenames=$5
big_dir=$6
job_name=$7
basedir=$8
rotation=$9
query_model_name=${10}
ref_model_name=${11}
[ -z $n_models ] && die "1st argument missing - give number of models you want to preprocess"
[ -z $n_jobs ] && die "2nd argument missing - give number of jobs"
[ -z $py_exe ] && die "3rd argument missing - give python executable"
[ -z $query_dir ] && die "4th argument missing - give query dir"
[ -z $filenames ] && die "5th argument missing - give filenames"
[ -z $big_dir ] && die "6th argument missing - give big dir"
[ -z $job_name ] && die "7th argument missing - give job name"
[ -z $basedir ] && die "8th argument missing - give basedir"
[ -z $rotation ] && die "9th argument missing - give rotation"
[ -z $rotation ] && die "9th argument missing - give rotation"
[ -z $query_model_name ] && die "10th argument missing - give query_model_name"
[ -z $ref_model_name ] && die "11th argument missing - give ref_model_name"


# Create basedir
#basedir=jobs_ABC
mkdir $basedir

# Create subdirs
echo "Creating directories..."
n=1;
subdir=job_
while [ "$n" -le "$n_jobs" ]; do
  mkdir "$basedir/$subdir$n"
  n=`expr "$n" + 1`;
done
echo "Done"

# Copy source code to each subdir
echo "Copying source files..."
sourceDir=source_code
n=1
while [ "$n" -le "$n_jobs" ]; do
	cp -r $sourceDir/* "$basedir/$subdir$n"
#	chmod 777 "$basedir/$subdir$n/*"
	n=`expr "$n" + 1`;
done
echo "Done"

# Starting jobs
n=1
numerator=$(echo $n_models+$n_jobs-1 | bc)
step=$(echo $numerator/$n_jobs | bc)
cur_ind=0
n_proc=38
while [ "$n" -le "$n_jobs" ]; do
	cd "$basedir/$subdir$n"
  echo "`pwd`" > submit_log.txt

  SBATCH_CMD="sbatch -J $job_name submit_job.sh"
	SBATCH_CMD="$SBATCH_CMD --py_exe $py_exe"
	SBATCH_CMD="$SBATCH_CMD --n_proc $n_proc"
	SBATCH_CMD="$SBATCH_CMD --start_ind $cur_ind"
	SBATCH_CMD="$SBATCH_CMD --end_ind `expr "$cur_ind" + "$step"`"
	SBATCH_CMD="$SBATCH_CMD --query_dir $query_dir"
	SBATCH_CMD="$SBATCH_CMD --filenames $filenames"
	SBATCH_CMD="$SBATCH_CMD --big_dir $big_dir"
	SBATCH_CMD="$SBATCH_CMD --rotation $rotation"
	SBATCH_CMD="$SBATCH_CMD --query_model_name $query_model_name"
	SBATCH_CMD="$SBATCH_CMD --ref_model_name $ref_model_name"

	echo $SBATCH_CMD
	echo $SBATCH_CMD > submit_log.txt
	eval $SBATCH_CMD
	cd "../.."
	n=`expr "$n" + 1`;
	cur_ind=`expr "$cur_ind" + "$step"`;
done

# /bin/bash /media/graphicslab/BigData/zavou/ANNFASS_CODE/style_detection/preprocess/yu_data/run_jobs.sh 5 2 /home/graphicslab/miniconda3/envs/style_detect_env/bin/python /media/graphicslab/BigData/zavou/ANNFASS_DATA/DATA_YU_LUN_HU/objs/building_yu_rotated filenames_buildings.txt /media/graphicslab/BigData/zavou/ANNFASS_DATA/BUILDNET_Buildings/normalizedObj
