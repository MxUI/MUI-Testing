#!/bin/bash

# Usage: [binary] [config_filename] 

sh clean.sh

bin=../build/Release/MUI_Testing

instanceOne="-n 2 ${bin} config1"
instanceTwo="-n 2 ${bin} config2"

#mpirun ${instanceOne} : ${instanceTwo}
mpirun --output-filename console_output ${instanceOne} : ${instanceTwo}

