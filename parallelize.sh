#!/bin/bash
#$ -cwd
#$ -j y
#$ -V
#$ -notify
#$ -N rename_extracted_fits_files
#$ -S /bin/bash
#$ -q astro.q

. /etc/profile.d/modules.sh
module load astro

#$ -t 1:100:1

# initialize vars
Njobs=100

# read in file
if [ $# -eq 0 ]; then
  echo Enter filename containing objects 
  read FNAME
else
  FNAME=$1
fi

Nlines=`awk '{x++}END{ print x}' $FNAME`
#echo "Nlines = $Nlines"
#echo "Njobs = $Njobs"

# determine which subset of rows to run script on
NUM_PER_PARTITION=`echo "$Nlines / $Njobs"|bc`
echo "Number of objects being processed per job = $NUM_PER_PARTITION"

REMAINING_NUM=`echo "$Nlines % $Njobs"|bc`
echo "Number of 'left-over' objects processed in last job = $REMAINING_NUM"

# run python script
if [[ $SGE_TASK_ID -gt 0 ]]; then
	echo "Task: $SGE_TASK_ID"
	BIN='python search_objects_parallel.py'
	START_INDEX=`echo "$SGE_TASK_ID - 1"|bc`
	START=`echo "$NUM_PER_PARTITION*$START_INDEX + 1"|bc`
	END=`echo "$START + $NUM_PER_PARTITION"|bc`
	echo "$BIN $FNAME $START $END"
	$BIN $FNAME $START $END
elif [[ $SGE_TASK_ID -eq $Njobs ]]; then
	# handle left-over rows
	echo "Task: $SGE_TASK_ID"
	START_INDEX=`echo "$SGE_TASK_ID - 1"|bc`
    START=`echo "$NUM_PER_PARTITION*$START_INDEX"|bc`
    END=`echo "$START + $REMAINING_NUM"|bc`
    echo "This is the last job"
	$BIN $FNAME $START $END
else
	echo "Something went wrong"
	exit 0
fi
