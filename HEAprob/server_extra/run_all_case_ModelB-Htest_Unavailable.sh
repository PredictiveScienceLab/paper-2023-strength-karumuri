# Run all the cases
# Checks
#!/bin/bash

for i in {1..100..1}
  do 
     sbatch server_general_ModelB-Htest_Unavailable.slurm -seed=$i
 done


