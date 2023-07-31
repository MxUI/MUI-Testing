/*****************************************************************************
* Multiscale Universal Interface Code Coupling Library Testing Framework     *
*                                                                            *
* Copyright (C) 2021 S. M. Longshaw^, A. Skillen^, J. Grasset^               *
*                                                                            *
* ^UK Research and Innovation Science and Technology Facilities Council      *
*                                                                            *
* This software is jointly licensed under the Apache License, Version 2.0    *
* and the GNU General Public License version 3, you may use it according     *
* to either.                                                                 *
*                                                                            *
* ** Apache License, version 2.0 **                                          *
*                                                                            *
* Licensed under the Apache License, Version 2.0 (the "License");            *
* you may not use this file except in compliance with the License.           *
* You may obtain a copy of the License at                                    *
*                                                                            *
* http://www.apache.org/licenses/LICENSE-2.0                                 *
*                                                                            *
* Unless required by applicable law or agreed to in writing, software        *
* distributed under the License is distributed on an "AS IS" BASIS,          *
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
* See the License for the specific language governing permissions and        *
* limitations under the License.                                             *
*                                                                            *
* ** GNU General Public License, version 3 **                                *
*                                                                            *
* This program is free software: you can redistribute it and/or modify       *
* it under the terms of the GNU General Public License as published by       *
* the Free Software Foundation, either version 3 of the License, or          *
* (at your option) any later version.                                        *
*                                                                            *
* This program is distributed in the hope that it will be useful,            *
* but WITHOUT ANY WARRANTY; without even the implied warranty of             *
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the              *
* GNU General Public License for more details.                               *
*                                                                            *
* You should have received a copy of the GNU General Public License          *
* along with this program.  If not, see <http://www.gnu.org/licenses/>.      *
*****************************************************************************/

/**
 * @file MUI_testing.cpp
 * @author S. M. Longshaw
 * @date 25 October 2021
 * @brief Testing and benchmarking framework for the Multiscale Universal
 *        Interface library
 */

#include "MUI_testing.h"

int main(int argc, char** argv) {
  //Parse input parameters
  if( argc < 2 ) {  //Check the number of parameters
    std::cerr << "Usage: " << argv[0] << " [config_filename]" << std::endl;
    exit(-1);
  }

  std::string fileName(argv[1]);
  parameters params;
  params.staticPoints = static_cast<bool>(mui::tf_config::FIXEDPOINTS);

  if( !readConfig(fileName, params) ) { //Parse input parameters, exit if false returned
    std::cerr << "Problem opening or reading configuration file" << std::endl;
    exit( -1 );
  }

  if( params.enableMPI ) {
    if( !initMPI(argc, argv, params) ) { //Initialise MPI, quit if false
      std::cerr << "Problem initialising MPI" << std::endl;
      exit( -1 );
    }
  }
  else { //Not using MPI but still need to initialise for MUI
    //mpi_split_by_app() calls MPI_Init() at start and MPI_Finalize() on exit
    world = mui::mpi_split_by_app(argc, argv);
    procName = "localhost";
  }

  calculateGridValues(params); //Calculate grid properties

  if( !createMUIInterfaces(params.interfaceFilePath, params) ) //Create MUI interface(s), quit if failed
    return 0;

  if( !createGridData(params) ) //Create grid data
    return 0;

  if( params.consoleOut )
    printData(params); //Print information to console

  // Ensure all ranks are to this point before starting run()
  if( params.enableMPI )
    MPI_Barrier(world);

  timing wallTime;

  if( params.useInterp ) {
    if( params.pushFetchOrder == 1 ) //Do work through the MUI interface using interpolation with push/fetch ordering, runtime returned in ms
      wallTime = run<true, true>(params);
    else if( params.pushFetchOrder == 2 ) //Do work through the MUI interface using interpolation with fetch/push ordering, runtime returned in ms
      wallTime = run<true, false>(params);
  }
  else {
    if( params.pushFetchOrder == 1 ) //Do work through the MUI interface without interpolation with push/fetch ordering, runtime returned in ms
      wallTime = run<false, true>(params);
    else if( params.pushFetchOrder == 2 ) //Do work through the MUI interface without interpolation with fetch/push ordering, runtime returned in ms
      wallTime = run<false, false>(params);
  }

  double globalTimeTotal = wallTime.totalTime;
  double globalTimeMUI = wallTime.muiTime;

  if( params.enableMPI ) { //Ensure each rank has created its data structure if using MPI
    MPI_Reduce(&wallTime.totalTime, &globalTimeTotal, 1, MPI_DOUBLE, MPI_SUM, 0, world);  // Perform MPI reduction for time values
    MPI_Reduce(&wallTime.muiTime, &globalTimeMUI, 1, MPI_DOUBLE, MPI_SUM, 0, world);  // Perform MPI reduction for time values
  }

  // Print average time value through master rank
  if( (!params.enableMPI) || (params.enableMPI && mpiRank == 0) ) {
    double avgTimeTotal = globalTimeTotal / static_cast<double>(mpiWorldSize);
    double avgTimeMUI = globalTimeMUI / static_cast<double>(mpiWorldSize);

    std::cout << outName << " Per-iteration wall clock value (MUI only): " << std::setprecision(10) << (avgTimeMUI / static_cast<double>(params.itCount)) << " ms" << std::endl;
    std::cout << outName << " Total wall clock value (MUI only): " << std::setprecision(10) << avgTimeMUI << " ms" << std::endl;
    std::cout << outName << " Per-iteration wall clock value (total): " << std::setprecision(10) << (avgTimeTotal / static_cast<double>(params.itCount)) << " ms" << std::endl;
    std::cout << outName << " Total wall clock value (total): " << std::setprecision(10) << avgTimeTotal << " ms" << std::endl;
  }

  finalise( (params.enableMPI && params.dataToSend > 0) ); //Clean up before exit

  return 0;
}

//***********************************************************
//* Function to perform work through MUI interface(s)
//***********************************************************
template<bool interpolated, bool pushFetchOrder>
timing run(parameters& params) {
  std::vector<REAL> rcvValues(muiInterfaces.size(), 10);
  std::vector<INT> numValues(muiInterfaces.size(), 1);

  if (!params.enableMPI || (params.enableMPI && mpiRank == 0)) {
    std::cout << outName << " Sending global parameters" << std::endl;
  }

  for (size_t interface = 0; interface < muiInterfaces.size(); interface++) {
    // This is an all-to-all send so only need to do it through MPI rank 0
    if (!params.enableMPI || (params.enableMPI && mpiRank == 0)) {
      // Assign value to send to interface
      muiInterfaces[interface].interface->push("rcvValue", params.sendValue);

      // Assign number of values to send to interface
      muiInterfaces[interface].interface->push("numValues", params.numMUIValues);
    }

    // All ranks must issue commit so timestamp at t=0 is sent to all ranks
    muiInterfaces[interface].interface->commit(static_cast<TIME>(0));
  }

  if (!params.enableMPI || (params.enableMPI && mpiRank == 0)) {
    std::cout << outName << " Initial values sent" << std::endl;
  }

  if (!params.enableMPI || (params.enableMPI && mpiRank == 0)) {
    std::cout << outName << " Receiving initial values" << std::endl;
  }

  // Need time barrier here to ensure other side has sent values
  for (size_t interface = 0; interface < muiInterfaces.size(); interface++) {
    muiInterfaces[interface].interface->barrier(static_cast<INT>(0));
  }

  // Fetch values (non-blocking)
  for (size_t interface = 0; interface < muiInterfaces.size(); interface++) {
    //Receive the value to be received through the interface
    rcvValues[interface] = muiInterfaces[interface].interface->fetch<REAL>("rcvValue");

    //Receive the number of values to be received through the interface
    numValues[interface] = muiInterfaces[interface].interface->fetch<INT>("numValues");
  }

  // Forget received time and reset interface log
  for (size_t interface = 0; interface < muiInterfaces.size(); interface++) {
    muiInterfaces[interface].interface->forget(static_cast<INT>(0), true);
  }

  if (!params.enableMPI || (params.enableMPI && mpiRank == 0)) {
    std::cout << outName << " Initial values received" << std::endl;
  }

  if( params.smartSend ) { //Enable MUI smart send comms reducing capability if enabled
    if( !params.enableMPI || (params.enableMPI && mpiRank == 0) ) {
      std::cout << outName << " Announcing Smart Send values" << std::endl;
    }

    //Announce send and receive region
    for(size_t interface=0; interface < muiInterfaces.size(); interface++) {
      mui::geometry::box<mui::tf_config> sendRcvRegion({params.rankDomainMin[0]+params.rankDomainMin[0]*1e-6, params.rankDomainMin[1]+params.rankDomainMin[1]*1e-6, params.rankDomainMin[2]+params.rankDomainMin[2]*1e-6},
                                                       {params.rankDomainMax[0]-params.rankDomainMax[0]*1e-6, params.rankDomainMax[1]-params.rankDomainMax[1]*1e-6, params.rankDomainMax[2]-params.rankDomainMax[2]*1e-6});

      // Announce Smart Send regions with communications blocking enabled to ensure synchronisation
      muiInterfaces[interface].interface->announce_send_span(static_cast<TIME>(0), static_cast<TIME>(params.itCount), sendRcvRegion, true);
      muiInterfaces[interface].interface->announce_recv_span(static_cast<TIME>(0), static_cast<TIME>(params.itCount), sendRcvRegion, true);
    }

    if( !params.enableMPI || (params.enableMPI && mpiRank == 0) ) {
      std::cout << outName << " Smart Send set up complete" << std::endl;
    }
  }

  if( !params.enableMPI || (params.enableMPI && mpiRank == 0) ) {
    std::cout << outName << " Beginning iterations..." << std::endl;
  }

  std::vector<POINT> rcvPoints;
  std::vector<REAL> rcvDirectValues;
  REAL rcvValue;
  bool checkValue;

  // Create parameter names for sending and receiving
  std::vector<std::string> sendParams;
  std::vector< std::vector<std::string> > rcvParams(muiInterfaces.size());

  for( size_t i=0; i<params.numMUIValues; i++ ) {
    std::stringstream paramName;
    paramName << "data_" << i;
    sendParams.push_back(paramName.str());
  }

  // Create receive parameter names for each interface
  for( size_t i=0; i<muiInterfaces.size(); i++ ) {
    for( size_t j=0; j<numValues[i]; j++) {
      std::stringstream paramName;
      paramName << "data_" << j;
      rcvParams[i].push_back(paramName.str());
    }
  }

  size_t total_arr = params.itot * params.jtot * params.ktot;

  // Individual MUI interfaces can have different set ups, so need one RBF filter per interface if enabled
  std::vector<mui::sampler_rbf<mui::tf_config>*> s1_rbf;

  // If RBF interpolation enabled then gather local (sending) point set used to generate basis matrix
  if( params.interpMode == 2 && interpolated ) {
    for( size_t interface=0; interface < muiInterfaces.size(); interface++ ) {
      // Define unique output directory for RBF matrix files if this has been enabled (otherwise a blank string turns this off in the filter)
      std::string outputDir;
      if( params.rbf_Write )
    	  outputDir.assign(params.rbf_dirName+"_"+muiInterfaces[interface].interfaceName+"_"+std::to_string(mpiRank));

      if( muiInterfaces[interface].sendRecv == 0 || muiInterfaces[interface].sendRecv == 2 ) { //Only push and commit if this interface is for sending or for send & receive
        // Gather active sending points for this rank into local std::vector
        std::vector<POINT> rbfPoints;
        for( size_t i=0; i<total_arr; i++ ) {
          if( sendEnabled[interface][i] ) {
            rbfPoints.push_back(sendRcvPoints[i].point);
          }
        }

        if ( rbfPoints.size() != 0 ) {
          // Create RBF sampler instance
          if( params.enableMPI ) {
            mui::sampler_rbf<mui::tf_config>* s1_rbf_local = new mui::sampler_rbf<mui::tf_config>(params.rbf_Radius, rbfPoints, params.rbf_BasisFunc,
                                                                   params.rbf_Conservative, params.rbf_Smooth, true, outputDir,
                                                                   params.rbf_Cutoff, params.rbf_CgSolveTol, params.rbf_CgSolveMaxIt,
                                                                   params.rbf_PoUSize, params.rbf_CgPreCon, world);

            s1_rbf.push_back(s1_rbf_local);
          }
          else { // MPI disabled so no need to provide RBF filter with MPI communicator
            mui::sampler_rbf<mui::tf_config>* s1_rbf_local = new mui::sampler_rbf<mui::tf_config>(params.rbf_Radius, rbfPoints, params.rbf_BasisFunc,
                                                                   params.rbf_Conservative, params.rbf_Smooth, true, outputDir,
                                                                   params.rbf_Cutoff, params.rbf_CgSolveTol, params.rbf_CgSolveMaxIt,
                                                                   params.rbf_PoUSize, params.rbf_CgPreCon);

            s1_rbf.push_back(s1_rbf_local);
          }
        }
        else { // No points were found to send for this rank so add null pointer to ensure std::vector size correct
          s1_rbf.push_back(NULL);
          std::cout << outName << " WARNING: no points found whilst generating RBF filter" << std::endl;
        }
      }
      else // Interface not enabled to send so add null pointer to ensure std::vector size correct
        s1_rbf.push_back(NULL);
    }
  }

  // Create single instance spatial and temporal samplers
  mui::sampler_gauss<mui::tf_config> s1_g(params.gauss_Radius, params.gauss_Height);
  mui::sampler_exact<mui::tf_config> s1_e;
  mui::temporal_sampler_exact<mui::tf_config> s2;

  double muiTime = 0;

  // Get starting time
  auto tStart = std::chrono::high_resolution_clock::now();

  //Iterate for as many times and send/receive through MUI interface(s)
  for(size_t iter=0; iter < static_cast<size_t>(params.itCount); iter++) {
    //Output progress to console
    if( params.consoleOut ) {
      if( !params.enableMPI || (params.enableMPI && mpiRank == 0) ) //Only perform on master rank if not in serial mode
        std::cout << outName << " Starting iteration " << iter+1 << std::endl;
    }

    TIME currTime = static_cast<TIME>(iter+1);
    auto tStartMUI = std::chrono::high_resolution_clock::now();

    // Push values if order is push/fetch
    if( pushFetchOrder ) {
      //Push and commit enabled values for each interface
      for( size_t interface=0; interface < muiInterfaces.size(); interface++ ) {
        if( muiInterfaces[interface].sendRecv == 0 || muiInterfaces[interface].sendRecv == 2 ) { //Only push and commit if this interface is for sending or for send & receive
          for( size_t i=0; i<total_arr; i++ ) {
            if( sendEnabled[interface][i] ) { //Push the value if it is enabled for this rank
              for( size_t vals=0; vals<sendParams.size(); vals++ ) {
                //Push value to interface
                muiInterfaces[interface].interface->push(sendParams[vals], sendRcvPoints[i].point, sendRcvPoints[i].value);
              }
            }
          }
          //Commit values to interface
          muiInterfaces[interface].interface->commit(currTime);
        }
      }
    }

    //Iterate through MUI interfaces and fetch enabled values
    for( size_t interface=0; interface < muiInterfaces.size(); interface++ ) {
      //Only fetch if this interface is for receiving or for send & receive
      if( muiInterfaces[interface].sendRecv == 1 || muiInterfaces[interface].sendRecv == 2) {
        if( !interpolated ) { // Using direct receive
          for( size_t vals=0; vals<numValues[interface]; vals++) {
            rcvPoints = muiInterfaces[interface].interface->fetch_points<REAL>(rcvParams[interface][vals], currTime, s2);

            if( rcvPoints.size() != 0 ) { //Check if any points exist in the interface for this rank
              rcvDirectValues = muiInterfaces[interface].interface->fetch_values<REAL>(rcvParams[interface][vals], currTime, s2);

              if( rcvDirectValues.size() == 0 && params.consoleOut ) {  //No values were received, report error
                  if( !params.enableMPI )
                    std::cout << outName << " Error: No values found in interface but points exist " << muiInterfaces[interface].interfaceName << std::endl;
                  else
                    std::cout << outName << " Error: No values found in interface but points exist " << muiInterfaces[interface].interfaceName << " for MPI rank " << mpiRank << std::endl;
              }
            }
          }
        }
        else { // Using spatial interpolation
          for( size_t i=0; i<total_arr; i++ ) {
            if( rcvEnabled[interface][i] ) { //Fetch the value if it is enabled for this rank
              for( size_t vals=0; vals<numValues[interface]; vals++) { //Iterate through as many values to receive per point
                //Fetch value from interface
                if( params.interpMode == 0 ) // Exact
                  rcvValue = muiInterfaces[interface].interface->fetch(rcvParams[interface][vals], sendRcvPoints[i].point, currTime, s1_e, s2);
                else if ( params.interpMode == 1 ) // Gaussian
                  rcvValue = muiInterfaces[interface].interface->fetch(rcvParams[interface][vals], sendRcvPoints[i].point, currTime, s1_g, s2);
                else if ( params.interpMode == 2 && s1_rbf[interface] != NULL ) // RBF
                  rcvValue = muiInterfaces[interface].interface->fetch(rcvParams[interface][vals], sendRcvPoints[i].point, currTime, *s1_rbf[interface], s2);
              }
            }
            else if( !params.smartSend ) { // Not using Smart Send so need to fetch anyway to clear MPI buffers (will return zero)
              for( size_t vals=0; vals<numValues[interface]; vals++) { //Iterate through as many values to receive per point
                //Fetch value from interface
                if( params.interpMode == 0 )
                  muiInterfaces[interface].interface->fetch(rcvParams[interface][vals], sendRcvPoints[i].point, currTime, s1_e, s2);
                else if( params.interpMode == 1 )
                  muiInterfaces[interface].interface->fetch(rcvParams[interface][vals], sendRcvPoints[i].point, currTime, s1_g, s2);
                else if ( params.interpMode == 2 ) // RBF
                  muiInterfaces[interface].interface->fetch(rcvParams[interface][vals], sendRcvPoints[i].point, currTime, *s1_rbf[interface], s2);
              }
            }
          }
        }
        // Forget fetched data frame from MUI interface to ensure memory free'd
        muiInterfaces[interface].interface->forget(currTime);
      }
    }

    if( pushFetchOrder )
      muiTime += static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - tStartMUI).count());
    else {
      muiTime += static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - tStartMUI).count());
      tStartMUI = std::chrono::high_resolution_clock::now();
    }

    //Sleep process for pre-defined period of time to simulate work being done by host code
    if( params.waitIt > 0 )
      std::this_thread::sleep_for(std::chrono::milliseconds(params.waitIt));

    // If artificial MPI data send enabled then perform (blocking)
    if( params.enableMPI && params.dataToSend > 0 ) {
      int err = MPI_Neighbor_alltoall(sendBuf, params.dataToSend, MPI_MB, recvBuf, params.dataToSend, MPI_MB, comm_cart);
      if(err != MPI_SUCCESS)
        std::cout << "Error: When calling MPI_Neighbor_alltoall" << std::endl;
    }

    // Push values if order is fetch/push
    if( !pushFetchOrder ) {
      //Push and commit enabled values for each interface
      for( size_t interface=0; interface < muiInterfaces.size(); interface++ ) {
        if( muiInterfaces[interface].sendRecv == 0 || muiInterfaces[interface].sendRecv == 2 ) { //Only push and commit if this interface is for sending or for send & receive
          for( size_t i=0; i<total_arr; i++ ) {
            if( sendEnabled[interface][i] ) { //Push the value if it is enabled for this rank
              for( size_t vals=0; vals<sendParams.size(); vals++ ) {
                //Push value to interface
                muiInterfaces[interface].interface->push(sendParams[vals], sendRcvPoints[i].point, sendRcvPoints[i].value);
              }
            }
          }
          //Commit values to interface
          muiInterfaces[interface].interface->commit(currTime);
        }
      }

      muiTime += static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - tStartMUI).count());
    }

    //Output progress to console
    if( params.consoleOut ) {
      if( !params.enableMPI || (params.enableMPI && mpiRank == 0) ) //Only perform on master rank if not in serial mode
        std::cout << outName << " Completed iteration " << iter+1 << std::endl;
    }
  }

  timing timings;
  timings.totalTime = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - tStart).count());
  timings.muiTime = muiTime;

  // Return iteration runtimes for this rank
  return timings;
}

//****************************************************
//* Function to initialise MPI
//****************************************************
bool initMPI(int argc, char** argv, parameters& params) {
  //mpi_split_by_app() calls MPI_Init() at start and MPI_Finalize() on exit
  world = mui::mpi_split_by_app(argc, argv);
  MPI_Comm_size(world, &mpiWorldSize);
  MPI_Comm_rank(world, &mpiRank);

  //Get processor host name
  int len;
  char name[MPI_MAX_PROCESSOR_NAME];
  MPI_Get_processor_name(name,&len);
  procName = name;

  // Create identifier for any output
  outName = std::string( "[" + procName + "] (" + params.domainName + ")" );

  if(mpiWorldSize > 1) {
    if(mpiRank == 0 && params.consoleOut){
      std::cout << outName << " MPI initialised, there are " << mpiWorldSize << " ranks" << std::endl;
    }
  }

  decompose(static_cast<INT>(params.numGridCells[0]), static_cast<INT>(params.numGridCells[1]), static_cast<INT>(params.numGridCells[2]), &params.px, &params.py, &params.pz);

  int num_partition[3] = {params.px, params.py, params.pz};
  int periods[3] = {params.usePeriodic, params.usePeriodic, params.usePeriodic};

  int err = MPI_Cart_create(world, 3, num_partition, periods, true, &comm_cart);
  if(err != MPI_SUCCESS) {
    std::cerr << outName << " Error: When creating split Cartesian MPI communicator." << std::endl;
    return false;
  }

  err = MPI_Cart_coords(comm_cart, mpiRank, 3, mpiCartesianRank);
  if(err != MPI_SUCCESS) {
    std::cerr << outName << " Error: When retrieving decomposed Cartesian coordinates." << std::endl;
    return false;
  }

    if( params.dataToSend > 0 ) {
    err = MPI_Type_contiguous(megabyte, MPI_BYTE, &MPI_MB);
    if(err != MPI_SUCCESS) {
      std::cout << outName << " Error: When creating new MPI_MB datatype" << std::endl;
    }

    err = MPI_Type_commit(&MPI_MB);
    if(err != MPI_SUCCESS) {
      std::cout << outName << " Error: When committing new MPI_MB datatype" << std::endl;
    }

    // Fill send and receive buffers with random data
    if( params.consoleOut && mpiRank == 0 ) {
       std::cout << outName << " Allocating MPI buffers, filling send with random data." << std::endl;
    }

    int dataSize = megabyte * params.dataToSend * 6;
    sendBuf = new char[dataSize];
    recvBuf = new char[dataSize];
    std::srand(std::time(nullptr));
    int r = std::rand();
    for(size_t i = 0; i< static_cast<size_t>(dataSize); i++){
      sendBuf[i] = r;
    }
  }

  return true;
}

//In theory this function does the same as MPI_Dims_create.
//In practice MPI_Dims_create is buggy in OpenMPI <3.1.6, <4.0.3, <3.0.6
//Although it seems to work fine with MPICH
void decompose(int ni, int nj, int nk, int* ni_, int* nj_, int* nk_) {
  *ni_ = 1;
  *nj_ = 1;
  *nk_ = 1;
  int np = mpiWorldSize;
  double lsq=1e10;

  for( int ii=1; ii<=np; ii++ ) {
    for( int jj=1; jj<=np/ii; jj++ ) {
      if( std::fabs( double(np)/double(ii)/double(jj) - np/ii/jj ) < 1e-5 ) {
        int kk=np/ii/jj;
        double lsqc = std::pow( double(ni)/double(ii), 2 ) + std::pow( double(nj)/double(jj), 2 ) + std::pow( double(nk)/double(kk), 2 );
        if( lsqc < lsq ) {
          lsq=lsqc;
          *ni_=ii;
          *nj_=jj;
          *nk_=kk;
        }
      }
    }
  }
}

//****************************************************
//* Function to calculate the grid details
//****************************************************
void calculateGridValues(parameters& params) {
  //Calculate total cells (allow for zero quantity)
  params.totalCells = static_cast<INT>(params.numGridCells[0])
                     * static_cast<INT>(params.numGridCells[1])
                     * static_cast<INT>(params.numGridCells[2]);

  //Calculate size of each grid element
  params.gridSize[0] = (params.domainMax[0] - params.domainMin[0]) / params.numGridCells[0];
  params.gridSize[1] = (params.domainMax[1] - params.domainMin[1]) / params.numGridCells[1];
  params.gridSize[2] = (params.domainMax[2] - params.domainMin[2]) / params.numGridCells[2];

  //Calculate centre of each grid element
  params.gridCentre[0] = static_cast<REAL>(0.5) * params.gridSize[0];
  params.gridCentre[1] = static_cast<REAL>(0.5) * params.gridSize[1];
  params.gridCentre[2] = static_cast<REAL>(0.5) * params.gridSize[2];

  // Not using local MPI so need to apply decomposition here
  if( !params.enableMPI )
    decompose(static_cast<INT>(params.numGridCells[0]), static_cast<INT>(params.numGridCells[1]), static_cast<INT>(params.numGridCells[2]), &params.px, &params.py, &params.pz);

  int rangeX, rangeY, rangeZ;
  rangeX = static_cast<int>((params.domainMax[0] - params.domainMin[0])) / params.px;
  rangeY = static_cast<int>((params.domainMax[1] - params.domainMin[1])) / params.py;
  rangeZ = static_cast<int>((params.domainMax[2] - params.domainMin[2])) / params.pz;

  //Find the coordinates of the rank's partition.
  //All partitions are represented as a rectangular parallelepiped and
  //can then be defined as two points.
  POINT partitionBegin, partitionEnd;
  partitionBegin[0] = mpiCartesianRank[0] * rangeX;
  partitionBegin[1] = mpiCartesianRank[1] * rangeY;
  partitionBegin[2] = mpiCartesianRank[2] * rangeZ;

  //If the partition is the last one before the end of the domain
  //then the partition goes up to the end of the domain in order to handle
  //the case where the dimension isn't divisible by the number of MPI rank
  if(mpiCartesianRank[0] == params.px-1) {
    partitionEnd[0] = params.domainMax[0] - params.domainMin[0];
		params.itot = static_cast<INT>(params.numGridCells[0]) / params.px + static_cast<INT>(params.numGridCells[0]) % params.px;
  }
  else {
    partitionEnd[0]=(mpiCartesianRank[0] + 1) * rangeX;
		params.itot = static_cast<INT>(params.numGridCells[0]) / params.px;
  }

  if(mpiCartesianRank[1] == params.py-1) {
    partitionEnd[1] = params.domainMax[1] - params.domainMin[1];
		params.jtot = static_cast<INT>(params.numGridCells[1]) / params.py + static_cast<INT>(params.numGridCells[1]) % params.py;
  }
  else {
    partitionEnd[1] = (mpiCartesianRank[1] + 1) * rangeY;
		params.jtot = static_cast<INT>(params.numGridCells[1]) / params.py;
  }

  if(mpiCartesianRank[2] == params.pz-1){
    partitionEnd[2] = params.domainMax[2] - params.domainMin[2];
		params.ktot = static_cast<INT>(params.numGridCells[2]) / params.pz + static_cast<INT>(params.numGridCells[2]) % params.pz;
  }
  else {
    partitionEnd[2] = (mpiCartesianRank[2] + 1) * rangeZ;
		params.ktot = static_cast<INT>(params.numGridCells[2]) / params.pz;
  }

  //Calculate domain extents for this rank
  params.rankDomainMin[0] = params.domainMin[0] + partitionBegin[0];
  params.rankDomainMin[1] = params.domainMin[1] + partitionBegin[1];
  params.rankDomainMin[2] = params.domainMin[2] + partitionBegin[2];
  params.rankDomainMax[0] = params.domainMin[0] + partitionEnd[0];
  params.rankDomainMax[1] = params.domainMin[1] + partitionEnd[1];
  params.rankDomainMax[2] = params.domainMin[2] + partitionEnd[2];
}

//****************************************************
//* Function to populate data in grid array
//****************************************************
bool createGridData(parameters& params) {
  if( params.consoleOut ) {
    std::cout << outName << " Grid points for rank " << mpiRank << ": [" << params.rankDomainMin[0] << "," << params.rankDomainMin[1] << "," << params.rankDomainMin[2] << "] - ["
              << params.rankDomainMax[0] << "," << params.rankDomainMax[1] << "," << params.rankDomainMax[2] << "]" << " MUI: " << sendInterfaces << " send, "
              << rcvInterfaces << " receive" << std::endl;
  }

  //Create array of 3D points of type double to send
  sendRcvPoints.resize(params.itot * params.jtot * params.ktot);

  // Create contiguous 3D arrays per interface for enabling send/receive
  for(size_t i=0; i<muiInterfaces.size(); i++) {
	  sendEnabled[i].resize(params.itot * params.jtot * params.ktot, false);
	  rcvEnabled[i].resize(params.itot * params.jtot * params.ktot, false);
  }

  size_t final_index;
  for(size_t i=0; i < params.itot; i++) {
    for(size_t j=0; j < params.jtot; j++) {
      for(size_t k=0; k < params.ktot; k++) {
        final_index = i + params.itot * j + params.itot * params.jtot * k;
        //Update send array
        sendRcvPoints[final_index].point[0] = params.rankDomainMin[0] + static_cast<REAL>(i * params.gridSize[0]) + params.gridCentre[0];
        sendRcvPoints[final_index].point[1] = params.rankDomainMin[1] + static_cast<REAL>(j * params.gridSize[1]) + params.gridCentre[1];
        sendRcvPoints[final_index].point[2] = params.rankDomainMin[2] + static_cast<REAL>(k * params.gridSize[2]) + params.gridCentre[2];
        sendRcvPoints[final_index].value = params.sendValue;

        // Check if the point is within MUI interface send/receive regions
        for( size_t interface=0; interface<muiInterfaces.size(); interface++) {
            // Create box structure of the overall send region for this interface
            mui::geometry::box<mui::tf_config> sendRegion({muiInterfaces[interface].domMinSend[0], muiInterfaces[interface].domMinSend[1], muiInterfaces[interface].domMinSend[2]},
                                                          {muiInterfaces[interface].domMaxSend[0], muiInterfaces[interface].domMaxSend[1], muiInterfaces[interface].domMaxSend[2]});

            sendEnabled[interface][final_index] = intersectPoint<mui::tf_config>(sendRcvPoints[final_index].point, sendRegion);

            // Create box structure of the overall send region for this interface
            mui::geometry::box<mui::tf_config> rcvRegion({muiInterfaces[interface].domMinRcv[0], muiInterfaces[interface].domMinRcv[1], muiInterfaces[interface].domMinRcv[2]},
                                                         {muiInterfaces[interface].domMaxRcv[0], muiInterfaces[interface].domMaxRcv[1], muiInterfaces[interface].domMaxRcv[2]});

            rcvEnabled[interface][final_index] = intersectPoint<mui::tf_config>(sendRcvPoints[final_index].point, rcvRegion);
        }
      }
    }
  }

  if(params.generateCSV) {
    if (mkdir("csv_output", S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
      if( errno != EEXIST )
        std::cerr << "Error creating main CSV output folder" << std::endl;
    }

    std::string dirName = "csv_output/" + params.domainName;

    if (mkdir(dirName.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH) == -1) {
      if( errno != EEXIST )
        std::cerr << "Error creating domain CSV output folder" << std::endl;
    }

    std::ofstream out;
    std::string filename = dirName + "/partition_" + std::to_string(mpiRank) + ".csv";
    out.open(filename, std::ios::out | std::ios::trunc);
    if(!out.is_open()) {
      std::cerr << outName << " Could not open file: " << filename << std::endl;
    }
    else {
      out << "x" << "," << "y" << "," << "z" << std::endl;
      for(size_t i=0; i < params.itot; i++) {
        for(size_t j=0; j < params.jtot; j++) {
          for(size_t k=0; k < params.ktot; k++) {
        	final_index = i + params.itot * j + params.itot * params.jtot * k;
            out << sendRcvPoints[final_index].point[0] << ",";
            out << sendRcvPoints[final_index].point[1] << ",";
            out << sendRcvPoints[final_index].point[2] << "\n";
          }
        }
      }
      out.close();
    }
  }

  return true;
}

//****************************************************
//* Function to print information to console
//****************************************************
void printData(parameters& params) {
  if(!params.enableMPI || (params.enableMPI && mpiRank == 0)) { //Only perform on master rank if not in serial mode
    double dataSize = static_cast<double>(params.totalCells * sizeof(pointData)) / static_cast<double>(megabyte);
    std::cout << outName << " Total grid data size: " << std::setprecision(4) << dataSize << " MB" << std::endl;
    std::cout << outName << " Total cell count: " << params.totalCells << std::endl;
    std::cout << outName << " Total domain cells: [" << static_cast<INT>(params.numGridCells[0]) << "," << static_cast<INT>(params.numGridCells[1]) << "," << static_cast<INT>(params.numGridCells[2]) << "]" << std::endl;
    std::cout << outName << " Total domain size: [" << params.domainMin[0] << "," << params.domainMin[1] << "," << params.domainMin[2] << "] - ["
              << params.domainMax[0] << "," << params.domainMax[1] << "," << params.domainMax[2] << "]" << std::endl;
    std::cout << outName << " Coupling logic: " << (params.pushFetchOrder == 1? "push/fetch": "fetch/push") << std::endl;
    std::cout << outName << " Iterations: " << params.itCount << std::endl;
    std::cout << outName << " Send value: " << params.sendValue << std::endl;
    std::cout << outName << " Static points: " << (params.staticPoints? "Enabled": "Disabled") << std::endl;
    std::cout << outName << " Smart Send: " << (params.smartSend? "Enabled": "Disabled") << std::endl;
    std::cout << outName << " Spatial interpolation: " << (params.useInterp? "Enabled": "Disabled") << std::endl;
    std::cout << outName << " Interpolation mode: " << (params.interpMode==0? "Exact": (params.interpMode==1? "Gaussian": "RBF")) << std::endl;
    std::cout << outName << " Artificial MPI data overhead: " << ((params.dataToSend > 0 && params.enableMPI)? "Enabled": "Disabled") << std::endl;
    std::cout << outName << " Artificial work time: " << params.waitIt << " ms" << std::endl;
  }
}

//****************************************************
//* Function to create MUI interfaces
//****************************************************
bool createMUIInterfaces(std::string& muiFileName, parameters& params) {
  // Read the interface definitions in from external file
  if( !readInterfaces(muiFileName, params.enableMPI) )
    return false;

  //Initialise interface counts for this rank
  sendInterfaces = 0;
  rcvInterfaces = 0;

  // Create flat list of interface names to create
  std::vector<std::string> interfaceNames;
  for( size_t i=0; i<muiInterfaces.size(); i++ ) {
    interfaceNames.emplace_back(muiInterfaces[i].interfaceName);
  }

  //Create MUI interface(s)
  std::vector<std::unique_ptr<mui::uniface<mui::tf_config>> > createdInterfaces = mui::create_uniface<mui::tf_config>(params.domainName, interfaceNames);

  //Iterate through created interfaces and create global data structures
  for(size_t i=0; i<createdInterfaces.size(); i++) {
    // Release unique pointer to interface and store in existing struct
    for( size_t j=0; j<muiInterfaces.size(); j++ ) {
      if( createdInterfaces[i]->uri_path().compare(muiInterfaces[j].interfaceName) == 0 ) {
        muiInterfaces[j].interface = createdInterfaces[i].release();

        // Increment global counts for send/receive interfaces
        if(muiInterfaces[j].sendRecv == 0 || muiInterfaces[j].sendRecv == 2)
          sendInterfaces++; //Increment enabled send interface count

        if(muiInterfaces[j].sendRecv == 1 || muiInterfaces[j].sendRecv == 2)
          rcvInterfaces++; //Increment enabled receive interface count
      }
    }
  }

  // Resize arrays to set whether each point should be sent/received for each interface
  sendEnabled.resize(muiInterfaces.size());
  rcvEnabled.resize(muiInterfaces.size());

  return true;
}

//****************************************************
//* Function to finalise before exit
//****************************************************
void finalise(bool usingMPI) {
 if( usingMPI ){
    int err = MPI_Type_free(&MPI_MB);
     if(err != MPI_SUCCESS)
       std::cerr << outName << " Error: When freeing new MPI_MB datatype" << std::endl;
     delete[] sendBuf;
     delete[] recvBuf;
  }

  //Delete MUI interfaces (finalises MPI)
  for(size_t interface=0; interface<muiInterfaces.size(); interface++)
    delete muiInterfaces[interface].interface;
}

//****************************************************
//* Function to read config file
//****************************************************
bool readConfig(std::string& fileName, parameters& params) {
  //Define possible input parameters
  std::vector<std::string> inputParams;
  inputParams.push_back("ENABLE_LOCAL_MPI");
  inputParams.push_back("DOMAIN");
  inputParams.push_back("NUM_GRID_CELLS");
  inputParams.push_back("CONSOLE_OUTPUT");
  inputParams.push_back("USE_STATIC_POINTS");
  inputParams.push_back("USE_SMART_SEND");
  inputParams.push_back("GENERATE_OUTPUT_CSV");
  inputParams.push_back("DOMAIN_NAME");
  inputParams.push_back("INTERFACE_FILE_PATH");
  inputParams.push_back("ITERATION_COUNT");
  inputParams.push_back("SEND_VALUE");
  inputParams.push_back("NUM_SEND_VALUES");
  inputParams.push_back("USE_INTERPOLATION");
  inputParams.push_back("INTERPOLATION_MODE");
  inputParams.push_back("GAUSS_RADIUS");
  inputParams.push_back("GAUSS_HEIGHT");
  inputParams.push_back("RBF_RADIUS");
  inputParams.push_back("RBF_BASIS_FUNC");
  inputParams.push_back("RBF_MODE");
  inputParams.push_back("RBF_SMOOTHING");
  inputParams.push_back("RBF_WRITE_MATRICES");
  inputParams.push_back("RBF_DIR_NAME");
  inputParams.push_back("RBF_GAUSS_CUT_OFF");
  inputParams.push_back("RBF_CG_SOLVE_TOL");
  inputParams.push_back("RBF_CG_MAX_ITER");
  inputParams.push_back("RBF_POU_SIZE");
  inputParams.push_back("RBF_CG_PRECON");
  inputParams.push_back("CHECK_RECEIVE_VALUE");
  inputParams.push_back("WAIT_PER_ITERATION");
  inputParams.push_back("DATA_TO_SEND_MPI");
  inputParams.push_back("USE_PERIODIC_PATTERN");
  inputParams.push_back("PUSH_FETCH_ORDER");

  //Create input stream and open the file
  std::ifstream configFile;
  configFile.open(fileName);

  //Process the file if it opened correctly
  if( configFile.is_open() ) {
    std::string line;
    int lineCount=1;
    std::string paramName;
    bool sendRcvAll = false;

    //Iterate through all lines in the file
    while( getline(configFile, line) ) {
      //Process any line that does not contain a '#' and is not blank
      if( line.find('#') == std::string::npos && line.length() > 0 ) {
        std::istringstream linestream(line); //Get the current line as an istringstream
        std::string item;
        int count = 0;
        bool newLineFound = false;
        bool param = false;
        int nodeNum = -1;

        //Iterate through all the items in the line, checking for space separated values
        while( getline(linestream, item, ' ') ) {
          switch( count ) {
            case 0: { // First item on a line
              //Check against possible input parameters
              for ( size_t i=0; i<inputParams.size(); i++) {
                if( item.compare(inputParams[i]) == 0 ) {
                  paramName = item;
                  param = true;
                  break;
                }
              }
              break;
            }
            case 1: { // Second item on a line
              // First value was a recognised parameter
              if( param ) { // Determine which parameter it was and process accordingly
                if( paramName.compare("ENABLE_LOCAL_MPI") == 0 ) {
                  if ( item.compare("YES") == 0 || item.compare("yes") == 0 )
                    params.enableMPI = true;
                  else if ( item.compare("NO") == 0 || item.compare("no") == 0 )
                    params.enableMPI = false;
                  else {
                    std::cerr << "Problem reading ENABLE_LOCAL_MPI parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("DOMAIN") == 0 ) {
                  if( !(processPoint(item, params.domainMin)) ) {
                    std::cerr << "Problem reading DOMAIN parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("NUM_GRID_CELLS") == 0 ) {
                  if( !(processPoint(item, params.numGridCells)) ) {
                    std::cerr << "Problem reading NUM_GRID_CELLS parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("CONSOLE_OUTPUT") == 0 ) {
                  if ( item.compare("YES") == 0 || item.compare("yes") == 0 )
                    params.consoleOut = true;
                  else if ( item.compare("NO") == 0 || item.compare("no") == 0 )
                    params.consoleOut = false;
                  else {
                    std::cerr << "Problem reading CONSOLE_OUTPUT parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("USE_SMART_SEND") == 0 ) {
                  if ( item.compare("YES") == 0 || item.compare("yes") == 0 )
                    params.smartSend = true;
                  else if ( item.compare("NO") == 0 || item.compare("no") == 0 )
                    params.smartSend = false;
                  else {
                    std::cerr << "Problem reading USE_SMART_SEND parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("GENERATE_OUTPUT_CSV") == 0 ) {
                  if ( item.compare("YES") == 0 || item.compare("yes") == 0 )
                    params.generateCSV = true;
                  else if ( item.compare("NO") == 0 || item.compare("no") == 0 )
                    params.generateCSV = false;
                  else {
                    std::cerr << "Problem reading GENERATE_OUTPUT_CSV parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("DOMAIN_NAME") == 0 ) {
                  if( item.empty() ) {
                    std::cerr << "Problem reading DOMAIN_NAME parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }

                  params.domainName = item;
                }

                if( paramName.compare("INTERFACE_FILE_PATH") == 0 ) {
                  if( item.empty() ) {
                    std::cerr << "Problem reading INTERFACE_FILE_PATH parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }

                  params.interfaceFilePath = item;
                }

                if( paramName.compare("ITERATION_COUNT") == 0 ) {
                  std::stringstream tmpItem(item); // Create stringstream of string

                  if( !(tmpItem >> params.itCount) ) {
                    std::cerr << "Problem reading ITERATION_COUNT parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("SEND_VALUE") == 0 ) {
                  std::stringstream tmpItem(item); // Create stringstream of string

                  if( !(tmpItem >> params.sendValue) ) {
                    std::cerr << "Problem reading SEND_VALUE parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("NUM_SEND_VALUES") == 0 ) {
                  std::stringstream tmpItem(item); // Create stringstream of string

                  if( !(tmpItem >> params.numMUIValues) ) {
                    std::cerr << "Problem reading NUM_SEND_VALUES parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("USE_INTERPOLATION") == 0 ) {
                  if ( item.compare("YES") == 0 || item.compare("yes") == 0 )
                    params.useInterp = true;
                  else if ( item.compare("NO") == 0 || item.compare("no") == 0 )
                    params.useInterp = false;
                  else {
                    std::cerr << "Problem reading USE_INTERPOLATION parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("INTERPOLATION_MODE") == 0 ) {
                  if ( item.compare("EXACT") == 0 || item.compare("exact") == 0 || item.compare("Exact") == 0 )
                    params.interpMode = 0;
                  else if ( item.compare("GAUSS") == 0 || item.compare("gauss") == 0 || item.compare("Gauss") == 0 )
                    params.interpMode = 1;
                  else if ( item.compare("RBF") == 0 || item.compare("rbf") == 0 )
                    params.interpMode = 2;
                  else {
                    std::cerr << "Problem reading INTERPOLATION_MODE parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("GAUSS_RADIUS") == 0 ) {
                  std::stringstream tmpItem(item); // Create stringstream of string

                  if( !(tmpItem >> params.gauss_Radius) ) {
                    std::cerr << "Problem reading GAUSS_RADIUS parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("GAUSS_HEIGHT") == 0 ) {
                  std::stringstream tmpItem(item); // Create stringstream of string

                  if( !(tmpItem >> params.gauss_Height) ) {
                    std::cerr << "Problem reading GAUSS_HEIGHT parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("RBF_RADIUS") == 0 ) {
                  std::stringstream tmpItem(item); // Create stringstream of string

                  if( !(tmpItem >> params.rbf_Radius) ) {
                    std::cerr << "Problem reading RBF_RADIUS parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("RBF_BASIS_FUNC") == 0 ) {
                  std::stringstream tmpItem(item); // Create stringstream of string

                  if( !(tmpItem >> params.rbf_BasisFunc) ) {
                    std::cerr << "Problem reading RBF_BASIS_FUNC parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }

                  if(params.rbf_BasisFunc < 0 || params.rbf_BasisFunc > 4) {
                    std::cerr << "Please ensure the value of RBF_BASIS_FUNC parameter on line " << lineCount
                        << "is valid (Gaussian=0; Wendland C0=1; Wendland C2=2; Wendland C4=3; Wendland C6=4)" << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("RBF_MODE") == 0 ) {
                  if ( item.compare("0") == 0 )
                    params.rbf_Conservative = false;
                  else if ( item.compare("1") == 0 )
                    params.rbf_Conservative = true;
                  else {
                    std::cerr << "Problem reading RBF_MODE parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("RBF_SMOOTHING") == 0 ) {
                  if ( item.compare("YES") == 0 || item.compare("yes") == 0 )
                    params.rbf_Smooth = true;
                  else if ( item.compare("NO") == 0 || item.compare("no") == 0 )
                    params.rbf_Smooth = false;
                  else {
                    std::cerr << "Problem reading RBF_SMOOTHING parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("RBF_WRITE_MATRICES") == 0 ) {
                  if ( item.compare("YES") == 0 || item.compare("yes") == 0 )
                    params.rbf_Write = true;
                  else if ( item.compare("NO") == 0 || item.compare("no") == 0 )
                    params.rbf_Write = false;
                  else {
                    std::cerr << "Problem reading RBF_WRITE_MATRICES parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("RBF_DIR_NAME") == 0 ) {
                  params.rbf_dirName = item;
                }

                if( paramName.compare("RBF_GAUSS_CUT_OFF") == 0 ) {
                  std::stringstream tmpItem(item); // Create stringstream of string

                  if( !(tmpItem >> params.rbf_Cutoff) ) {
                    std::cerr << "Problem reading RBF_GAUSS_CUT_OFF parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("RBF_CG_SOLVE_TOL") == 0 ) {
                  std::stringstream tmpItem(item); // Create stringstream of string

                  if( !(tmpItem >> params.rbf_CgSolveTol) ) {
                    std::cerr << "Problem reading RBF_CG_SOLVE_TOL parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("RBF_CG_MAX_ITER") == 0 ) {
                  std::stringstream tmpItem(item); // Create stringstream of string

                  if( !(tmpItem >> params.rbf_CgSolveMaxIt) ) {
                    std::cerr << "Problem reading RBF_CG_MAX_ITER parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }

                  if(params.rbf_CgSolveMaxIt < 0) {
                    std::cerr << "Please ensure the value of RBF_CG_MAX_ITER parameter on line " << lineCount
                        << "is valid (must be positive)" << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("RBF_POU_SIZE") == 0 ) {
                  std::stringstream tmpItem(item); // Create stringstream of string

                  if( !(tmpItem >> params.rbf_PoUSize) ) {
                    std::cerr << "Problem reading RBF_POU_SIZE parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }

                  if(params.rbf_PoUSize < 0) {
                    std::cerr << "Please ensure the value of RBF_POU_SIZE parameter on line " << lineCount
                        << "is valid (must be positive)" << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("RBF_CG_PRECON") == 0 ) {
                  std::stringstream tmpItem(item); // Create stringstream of string

                  if( !(tmpItem >> params.rbf_CgPreCon) ) {
                    std::cerr << "Problem reading RBF_CG_PRECON parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }

                  if(params.rbf_CgPreCon < 0 || params.rbf_CgPreCon > 1) {
                    std::cerr << "Please ensure the value of RBF_POU_SIZE parameter on line " << lineCount
                        << "is valid (None=0; diagonal=1)" << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("WAIT_PER_ITERATION") == 0 ) {
                  std::stringstream tmpItem(item); // Create stringstream of string

                  if( !(tmpItem >> params.waitIt) ) {
                    std::cerr << "Problem reading WAIT_PER_ITERATION parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("DATA_TO_SEND_MPI") == 0 ) {
                  std::stringstream tmpItem(item); // Create stringstream of string

                  if( !(tmpItem >> params.dataToSend) ) {
                    std::cerr << "Problem reading DATA_TO_SEND_MPI parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("USE_PERIODIC_PATTERN") == 0 ) {
                  if ( item.compare("YES") == 0 || item.compare("yes") == 0 )
                    params.usePeriodic = true;
                  else if ( item.compare("NO") == 0 || item.compare("no") == 0 )
                    params.usePeriodic = false;
                  else {
                    std::cerr << "Problem reading USE_PERIODIC_PATTERN parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }

                if( paramName.compare("PUSH_FETCH_ORDER") == 0 ) {
                  std::stringstream tmpItem(item); // Create stringstream of string

                  if( !(tmpItem >> params.pushFetchOrder) ) {
                    std::cerr << "Problem reading PUSH_FETCH_ORDER parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }

                  if( params.pushFetchOrder < 1 || params.pushFetchOrder > 2 ) {
                    std::cerr << "PUSH_FETCH_ORDER parameter must equal 1 or 2 on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                }
              }
              break;
            }
            case 2: {// Third value on a line
              if( param ) { // This could be the third value for some parameters or a new line
                bool paramDefined = false;

                if( paramName.compare("DOMAIN") == 0 ) {
                  if( !(processPoint(item, params.domainMax)) ) {
                    std::cerr << "Problem reading DOMAIN parameter on line " << lineCount << std::endl;
                    exit( -1 );
                  }
                  paramDefined = true;
                }

                if( !paramDefined )
                  newLineFound = true;
              }
              break;
            }
            default: {
              break;
            }
          }
          count++;
        }

        // If a newline character was found then subtract overall line count to account for this
        if( newLineFound )
          count--;

        if( param ) { // Parameter line
          if ( count < 2 || count > 3 ) {
            std::cerr << "Incorrect number of values read on line " << lineCount << " (must be 2 or 3)" << std::endl;
            exit( -1 );
          }
        }
      }

      lineCount++;
    }

    configFile.close();

    return true;
  }
  else {
    std::cerr << "Opening configuration failed" << std::endl;
    return false;
  }
}

//****************************************************
//* Function to read MUI interface file
//****************************************************
bool readInterfaces(std::string& fileName, bool usingMPI) {
  std::ifstream muiInFile(fileName);

  if( muiInFile.is_open() ) {
    while( muiInFile ) {
      std::string s;
      if( !getline(muiInFile,s) ) break;
      if( s.find('#') == std::string::npos && s.length() > 0 ) { //Omit lines containing a # character
        std::istringstream ss(s);
        std::vector <std::string> record;

        while(ss) {
          std::string s;
          if(!getline(ss,s,' ')) break;
          record.push_back(s);
        }

        if( record.size() == 6 ) { //Reading in MUI interface definition
          if( record[0].compare("") == 0 )
            std::cerr << outName << " Interface name cannot be empty in MUI input file\n";
          else {
            muiInterface newInterface;
            newInterface.interfaceName = record[0];
            std::istringstream sendRecvss(record[1]);
            std::istringstream dom_min_ss_send(record[2]);
            std::istringstream dom_max_ss_send(record[3]);
            std::istringstream dom_min_ss_rcv(record[4]);
            std::istringstream dom_max_ss_rcv(record[5]);

            if( !(sendRecvss >> newInterface.sendRecv) ) { //Something wrong with value
              if(!usingMPI || (usingMPI && mpiRank == 0)) //Only perform on master rank if not in serial mode
                std::cerr << outName << " Error: Ensure send_or_recv valid in MUI input file" << std::endl;
              return false;
            }
            if(newInterface.sendRecv < 0 || newInterface.sendRecv > 2) {
              if(!usingMPI || (usingMPI && mpiRank == 0)) //Only perform on master rank if not in serial mode
                std::cerr << outName << " Error: Ensure send_or_recv valid in MUI input file" << std::endl;
              return false;
            }

            if( !(processPoint(dom_min_ss_send.str(), newInterface.domMinSend)) ) { //Something wrong with value
              if(!usingMPI || (usingMPI && mpiRank == 0)) //Only perform on master rank if not in serial mode
                std::cerr << outName << " Error: Ensure send_min valid in MUI input file" << std::endl;
              return false;
            }

            if( !(processPoint(dom_max_ss_send.str(), newInterface.domMaxSend)) ) { //Something wrong with value
              if(!usingMPI || (usingMPI && mpiRank == 0)) //Only perform on master rank if not in serial mode
                std::cerr << outName << " Error: Ensure send_max valid in MUI input file" << std::endl;
              return false;
            }

            if( !(processPoint(dom_min_ss_rcv.str(), newInterface.domMinRcv)) ) { //Something wrong with value
              if(!usingMPI || (usingMPI && mpiRank == 0)) //Only perform on master rank if not in serial mode
                std::cerr << outName << " Error: Ensure receive_min valid in MUI input file" << std::endl;
              return false;
            }

            if( !(processPoint(dom_max_ss_rcv.str(), newInterface.domMaxRcv)) ) { //Something wrong with value
              if(!usingMPI || (usingMPI && mpiRank == 0)) //Only perform on master rank if not in serial mode
                std::cerr << outName << " Error: Ensure receive_max valid in MUI input file" << std::endl;
              return false;
            }

            muiInterfaces.emplace_back(newInterface);
          }
        }
        else { //A problem line in the file
          std::cerr << outName << " Error: Incorrect number of values in MUI input file\n";
          return false;
        }
      }
    }

    muiInFile.close();
    return true;
  }
  else {
    std::cerr << outName << " Error: Unable to open MUI input file\n";
    return false;
  }
}

bool processPoint(const std::string& item, POINT& value) {
  std::string strValue(item);

  // Remove curly braces
  strValue.erase(std::remove(strValue.begin(), strValue.end(), '{'), strValue.end());
  strValue.erase(std::remove(strValue.begin(), strValue.end(), '}'), strValue.end());

  std::istringstream innerLinestream(strValue); //Get the item as an istringstream
  std::string innerItem;
  int innerCount = 0;

  // Iterate through comma separated contents of item
  while( getline(innerLinestream, innerItem, ',') ) {
    std::stringstream tmpInnerItem(innerItem); // Create stringstream of string
    switch( innerCount ) {
      case 0: { // X value
        if( !(tmpInnerItem >> value[0]) ) return false;
        break;
      }
      case 1: { // Y Value
        if( !(tmpInnerItem >> value[1]) ) return false;
        break;
      }
      case 2: { // Z Value
        if( !(tmpInnerItem >> value[2]) ) return false;
        break;
      }
      default : {
        break;
      }
    }
    innerCount++;
  }

  return true;
}

//****************************************************
//* Function to check if point inside a box
//****************************************************
template <typename T> inline bool intersectPoint(POINT& point, mui::geometry::box<T>& box) {
  bool gtltCheck = (point[0] > box.get_min()[0] && point[0] < box.get_max()[0]) &&
                   (point[1] > box.get_min()[1] && point[1] < box.get_max()[1]) &&
                   (point[2] > box.get_min()[2] && point[2] < box.get_max()[2]);

  bool eqCheck = (almostEqual<REAL>(point[0], box.get_min()[0]) || almostEqual<REAL>(point[0], box.get_max()[0]) ||
                  almostEqual<REAL>(point[1], box.get_min()[1]) || almostEqual<REAL>(point[1], box.get_max()[1]) ||
                  almostEqual<REAL>(point[2], box.get_min()[2]) || almostEqual<REAL>(point[2], box.get_max()[2]));

  return gtltCheck || eqCheck;
}

//****************************************************
//* Function to perform AABB intersection test
//****************************************************
template <typename T> inline bool intersectBox(mui::geometry::box<T>& a, mui::geometry::box<T>& b) {
  bool gtltCheck = (a.get_min()[0] < b.get_max()[0] && a.get_max()[0] > b.get_min()[0]) &&
                   (a.get_min()[1] < b.get_max()[1] && a.get_max()[1] > b.get_min()[1]) &&
                   (a.get_min()[2] < b.get_max()[2] && a.get_max()[2] > b.get_min()[2]);

  bool eqCheck = (almostEqual<REAL>(a.get_min()[0], b.get_max()[0])) ||
                 (almostEqual<REAL>(a.get_min()[1], b.get_max()[1])) ||
                 (almostEqual<REAL>(a.get_min()[2], b.get_max()[2]));

  return gtltCheck || eqCheck;
}

//******************************************************************
//* Function to check if two floating point values are almost equal
//******************************************************************
template <typename T> inline bool almostEqual(T x, T y) {
  return (x == y) ||
         (std::fabs(x-y) < std::numeric_limits<T>::epsilon() * std::fabs(x+y)) ||
         (std::fabs(x-y) < std::numeric_limits<T>::min());
}
