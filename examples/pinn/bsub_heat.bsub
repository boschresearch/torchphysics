#!/bin/bash -l
# Sample script for pytorch Lightning for PINN

## Scheduler parameters ##

#BSUB -J heat_low            # job name
#BSUB -o heat.%J.stdout   # optional: have output written to specific file
#BSUB -e heat.%J.stderr   # optional: have errors written to specific file
#BSUB -q batch_a100               # optional: use highend nodes w/ Volta GPUs (default: Geforce GPUs)
#BSUB -W 1:30                       # fill in desired wallclock time [hours,]minutes (hours are optional)
#BSUB -n 2                          # min CPU cores,max CPU cores (max cores is optional)
#BSUB -M 92000                      # fill in required amount of RAM (in Mbyte)
# #BSUB -R "span[hosts=1]"          # optional: run on single host (if using more than 1 CPU cores)
# #BSUB -R "span[ptile=28]"         # optional: fill in to specify cores per node (max 28)
# #BSUB -P myProject                # optional: fill in cluster project
## BSUB -gpu "num=1"
#BSUB -gpu "num=2:mode=exclusive_process:mps=no" # use 1 GPU (in explusive process mode)

## Job parameters ##
vEnv=TorchPhysics

# Load modules
module purge 
module load conda
module load cuda/11.5.1
module load cudnn/11.5_v8.3

python heat_equation.py