#!/bin/bash

#ONE=$(qsub e06_4.pbs)
#echo $ONE
#TWO=$(qsub -W depend=afterany:$ONE e06_5.pbs)
#echo $TWO
#THR=$(qsub -W depend=afterany:$TWO e06_3.pbs)
#echo $THR
FIRST=$(qsub e05_1.pbs)
echo $FIRST
SECOND=$(qsub -W depend=afterany:$FIRST e05_2.pbs)
echo $SECOND
THIRD=$(qsub -W depend=afterany:$SECOND e05_3.pbs)
echo $THIRD
FOURTH=$(qsub -W depend=afterany:$THIRD e05_4.pbs)
echo $FOURTH
FIFTH=$(qsub -W depend=afterany:$FOURTH e05_5.pbs)
echo $FIFTH
SIXTH=$(qsub -W depend=afterany:$FIFTH e05_6.pbs)
echo $SIXTH
