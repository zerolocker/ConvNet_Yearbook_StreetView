#!/bin/sh
#SBATCH -J deeplearnproj1           # Job name
#SBATCH -o experiment_x.out    # Specify stdout output file (%j expands to jobId) # Change X to desired order
#SBATCH -p gpu                           # Queue name
#SBATCH -N 1                     # Total number of nodes requested (16 cores/node)
#SBATCH -n 1                     # Total number of tasks
#SBATCH -t 3:30:00              # Run time (hh:mm:ss) - 3.5 hours
#SBATCH -A CS395T         # Specify allocation to charge against


python runvgg.py --learning_rate $lr --eps $eps
