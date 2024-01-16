#!/bin/bash
#SBATCH --nodes=1 --gpus=1
#SBATCH --time=01:00:00
#SBATCH --job-name=GP


# These are comment lines
# Load the Anaconda distribution of Python, which comes
# pre-bundled with many of the popular scientific computing tools like
# numpy, scipy, pandas, scikit-learn, etc.
module load anaconda/5.1.0-py36
source activate py36


### Place your python script run command here
### Example :

python ModelB-Htest_Unavailable.py $1

