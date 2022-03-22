#!/bin/bash

# Usage: [binary] [config_filename]

./clean.sh

bin=../build/MUI_Testing

MPI_RANKS_1=2
MPI_RANKS_2=2

instanceOne="-n ${MPI_RANKS_1} ${bin} config1"
instanceTwo="-n ${MPI_RANKS_2} ${bin} config2"

mpiexec.hydra -check_mpi ${instanceOne} : ${instanceTwo}
