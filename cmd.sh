#!/bin/sh
#SBATCH -J deeplearnproj1           # Job name
#SBATCH -o experiment.out    # Specify stdout output file (%j expands to jobId)
#SBATCH -p gpu                           # Queue name
#SBATCH -N 1                     # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                     # Total number of tasks
#SBATCH -t 12:00:00              # Run time (hh:mm:ss) - 3.5 hours
#SBATCH -A CS395T         # Specify allocation to charge against

# default value for $reg
if [[ -z $reg ]]; then
    reg=1e-5
fi

if [[ -z $dr ]]; then
    dr=0.5
fi

if [[ -z $fname ]]; then
    fname='u'
fi


python runvgg.py --learning_rate $lr --eps $eps --dropout $dr --reg $reg --fname $fname
