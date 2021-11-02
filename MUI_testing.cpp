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
    if(!initMPI(argc, argv, params)) { //Initialise MPI, quit if false
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

  if( params.enableMPI ) //Ensure each rank has created its data structure if using MPI
    MPI_Barrier(world);

  if( params.consoleOut )
    printData(params); //Print information to console

  tStart = MPI_Wtime(); //Get start timer

  if( !run(params) ) //Do work through the MUI interface
    return 0;

  if( params.enableMPI) //Ensure each rank has reached here if using MPI
    MPI_Barrier(world);

  tEnd = MPI_Wtime(); //Get end time

  if(params.consoleOut)
    std::cout << "[" << procName << "] " "Wall-time for rank " << mpiRank << ": " << tEnd - tStart << "s" << std::endl;

  finalise(params.enableMPI); //Clean up before exit

  return 0;
}

//***********************************************************
//* Function to perform work through MUI interface(s)
//***********************************************************
bool run(parameters& params) {
  double progress = 0.0;
  double progressChunk = 1.0/static_cast<double>(params.itCount-1);
  int barWidth = 40;
  int pos=0;

  std::vector<REAL> rcvValues(muiInterfaces.size(), -1);
  POINT sendPoint(0, 0, 0);
  REAL gaussParam = std::max(std::max(params.gridSize[0], params.gridSize[1]), params.gridSize[2]);
  mui::sampler_gauss<mui::tf_config> s1( gaussParam, gaussParam / static_cast<REAL>(2) );
  mui::chrono_sampler_exact<mui::tf_config> s2;

  if( params.smartSend ) { //Enable MUI smart send comms reducing capability if enabled
    //Announce send and receive region for each interface
    for(size_t interface=0; interface < muiInterfaces.size(); interface++) {
      double smallValue = 1e-8;
      int endTime = static_cast<int>(params.itCount);

      //Create communication send region according to this ranks oveerall domain to allow MUI
      // to reduce communication between ranks where possible
      mui::geometry::box<mui::tf_config> send_region({params.domainMin[0]+smallValue, params.domainMin[1], params.domainMin[2]},
                                                     {params.domainMax[0]-smallValue, params.domainMax[1], params.domainMax[2]});

      REAL rcvOffsetX = rcvMax[0] - sendMax[0];
      REAL rcvOffsetY = rcvMax[1] - sendMax[1];
      REAL rcvOffsetZ = rcvMax[2] - sendMax[2];

      //Create communication receive region to allow MUI to reduce communication between ranks where possible
      mui::geometry::box<mui::tf_config> rcv_region({(params.domainMin[0]+smallValue)+rcvOffsetX, params.domainMin[1]+rcvOffsetY, params.domainMin[2]+rcvOffsetZ},
                                                    {(params.domainMax[0]-smallValue)+rcvOffsetX, params.domainMin[1]+rcvOffsetY, params.domainMin[2]+rcvOffsetZ});

      if( muiInterfaces[interface].sendRecv == 0 || muiInterfaces[interface].sendRecv == 2) { //If the interface is set to send or send & receive
        muiInterfaces[interface].interface->announce_send_span(static_cast<TIME>(0), endTime, send_region);
      }

      if(muiInterfaces[interface].sendRecv == 1 || muiInterfaces[interface].sendRecv == 2) { //If the interface is set to receive or send & receive
        muiInterfaces[interface].interface->announce_recv_span(static_cast<TIME>(0), endTime, rcv_region);
      }

      //Assign value to send to interface
      muiInterfaces[interface].interface->push("rcv_value", params.sendValue);

      //Commit values to interface at t=0 so barrier can release
      muiInterfaces[interface].interface->commit(static_cast<TIME>(0));
    }

    //Barrier to ensure other side of interface has pushed timeframe so smart_send enabled across ranks and receive value sent
    for(size_t interface=0; interface < muiInterfaces.size(); interface++) {
      muiInterfaces[interface].interface->barrier(static_cast<TIME>(0));
    }
  }
  else { //Not using smart_send so only set up receive value
    //Barrier to ensure other side of interface has pushed timeframe so smart_send enabled across ranks and receive value sent
    for(size_t interface=0; interface < muiInterfaces.size(); interface++) {
      //Assign value to send to interface
      muiInterfaces[interface].interface->push("rcv_value", params.sendValue);
    }

    //Commit at t=0 so other side of interface can release barrier and continue
    for(size_t interface=0; interface < muiInterfaces.size(); interface++) {
      //Commit values to interface at t=0 so barrier can release
      muiInterfaces[interface].interface->commit(static_cast<TIME>(0));
    }

    //Barrier to ensure other side of interface has pushed timeframe so smart_send enabled across ranks and receive value sent
    for(size_t interface=0; interface < muiInterfaces.size(); interface++) {
      muiInterfaces[interface].interface->barrier(static_cast<TIME>(0));
    }
  }

  if( params.enableMPI ) //Ensure each rank has reached here if using MPI
    MPI_Barrier(world);

  for(size_t interface=0; interface < muiInterfaces.size(); interface++) {
    //Receive the value to be received through the interface
    rcvValues[interface] = muiInterfaces[interface].interface->fetch<REAL>("rcv_value");
  }

  std::vector<POINT> rcvPoints;
  std::vector<REAL> rcvDirectValues;
  std::vector<pointData> lclRcvPoints;
  std::vector<bool> rcvDirectValuesEnabled(muiInterfaces.size(), false);

  //Iterate for as many times and send/receive through MUI interface(s)
  for(size_t iter=0; iter < static_cast<size_t>(params.itCount); iter++) {
    int currTime = static_cast<int>(iter+1);

    //Output progress to console
    if( params.consoleOut ) {
      if( !params.enableMPI || (params.enableMPI && mpiRank == 0) ) { //Only perform on master rank if not in serial mode
        std::cout << "Iteration [";
        pos = barWidth * progress;
        for(int bar=0; bar < barWidth; bar++) {
          if(bar < pos)
            std::cout << "=";
          else if (bar == pos)
            std::cout << ">";
          else
            std::cout << " ";
        }
        std::cout << "] " << iter+1 << "\r";
        std::cout.flush();

        progress += progressChunk;
      }
    }

    //Send and receive value for receive from each interface
    for( size_t interface=0; interface < muiInterfaces.size(); interface++ ) {
      if( iter > 0 ) //Forget the previous buffer
        muiInterfaces[interface].interface->forget(static_cast<TIME>(iter-1));

      if( muiInterfaces[interface].sendRecv == 0 || muiInterfaces[interface].sendRecv == 2 ) { //Only push if this interface is for sending or for send & receive
        for( size_t i=0; i<params.itot; ++i ) {
          for( size_t j=0; j<params.jtot; ++j ) {
            for( size_t k=0; k<params.ktot; ++k ) {
              if( array3d_send[i][j][k].enabled ) { //Push the value if it is enabled for this rank
                sendPoint[0] = array3d_send[i][j][k].point[0];
                sendPoint[1] = array3d_send[i][j][k].point[1];
                sendPoint[2] = array3d_send[i][j][k].point[2];

                //Push value to interface
                muiInterfaces[interface].interface->push("data", sendPoint, array3d_send[i][j][k].value);
              }
            }
          }
        }

        //Commit values to interface
        muiInterfaces[interface].interface->commit(currTime);
      }
    }

    for( size_t interface=0; interface < muiInterfaces.size(); interface++ ) {
      if( muiInterfaces[interface].sendRecv == 1 || muiInterfaces[interface].sendRecv == 2) { //Only fetch if this interface is for receiving or for send & receive
        //if this is the first iteration and using static points then fetch all of the point locations sent through the
        //interface to set up receive array or if not static then do every iteration
        if( !params.staticPoints || (params.staticPoints && iter == 0) ) {
          rcvPoints = muiInterfaces[interface].interface->fetch_points<REAL>("data", currTime, s2);

          if( rcvPoints.size() != 0 ) { //Check if any points exist in the interface for this rank
            if( !params.useInterp ) { //Receiving values directly from the interface
              rcvDirectValuesEnabled[interface] = true; //Flag that this rank received values
              rcvDirectValues = muiInterfaces[interface].interface->fetch_values<REAL>("data", currTime, s2);

              if( rcvDirectValues.size() == 0 ) { //Check points were actually received
                if( params.consoleOut ) {
                  if( !params.enableMPI )
                    std::cout << "[" << procName << "] " << "Error: No values found in interface but points exist " << muiInterfaces[interface].interfaceName << std::endl;
                  else
                    std::cout << "[" << procName << "] " << "Error: No values found in interface but points exist " << muiInterfaces[interface].interfaceName << " for MPI rank " << mpiRank << std::endl;
                }
              }
              else { //Check values received make sense
                if( params.consoleOut ) {
                  for( size_t j=0; j<rcvDirectValues.size(); j++) {
                    if( rcvDirectValues[j] != rcvValues[interface]) {
                      if( !params.enableMPI )
                        std::cout << "[" << procName << "] " << "Error: Received value not as expected for interface " << muiInterfaces[interface].interfaceName << std::endl;
                      else
                        std::cout << "[" << procName << "] " << "Error: Received value not as expected for interface " << muiInterfaces[interface].interfaceName << " for MPI rank " << mpiRank << std::endl;
                    }
                  }
                }
              }
            }
            else { //Using spatial interpolation to receive values
              lclRcvPoints.resize(rcvPoints.size());
              for( size_t i=0; i<rcvPoints.size(); ++i ) {
                lclRcvPoints[i].point[0] = rcvPoints[i][0];
                lclRcvPoints[i].point[1] = rcvPoints[i][1];
                lclRcvPoints[i].point[2] = rcvPoints[i][2];
              }

              createRcvGridData(interface, lclRcvPoints); //Create the receive grid array

              //Iterate for as many points as we are to receive through the interface
              for( size_t i=0; i<array3d_rcv[interface].first; ++i ) {
                sendPoint[0] = array3d_rcv[interface].second[i].point[0];
                sendPoint[1] = array3d_rcv[interface].second[i].point[1];
                sendPoint[2] = array3d_rcv[interface].second[i].point[2];

                //Receive value from other side
                array3d_rcv[interface].second[i].value = muiInterfaces[interface].interface->fetch("data", sendPoint, currTime, s1, s2);

                if( params.enableMPI ) {
                  //Check value received make sense
                  if( array3d_rcv[interface].second[i].value != rcvValues[interface]) {
                    if( !params.enableMPI )
                      std::cout << "[" << procName << "] " << "Error: Received value not as expected for interface " << muiInterfaces[interface].interfaceName << std::endl;
                    else
                      std::cout << "[" << procName << "] " << "Error: Received value not as expected for interface " << muiInterfaces[interface].interfaceName << " for MPI rank " << mpiRank << std::endl;
                  }
                }
              }
            }
          }
        }
        else { //Using static points and this is not iteration 0 so no need to re-create local data structure
          if( !params.useInterp ) { //Receiving values directly from the interface
            if( rcvDirectValuesEnabled[interface] ) { //Check values were received for this rank during the first iteration
              rcvDirectValues = muiInterfaces[interface].interface->fetch_values<REAL>("data", currTime, s2);

              if( rcvDirectValues.size() == 0 ) { //Check points were actually received
                if( params.consoleOut ) {
                  if( !params.enableMPI ) {
                    std::cout << "[" << procName << "] " << "Error: No values found in interface but points exist " << muiInterfaces[interface].interfaceName << std::endl;
                  }
                  else {
                    std::cout << "[" << procName << "] " << "Error: No values found in interface but points exist " << muiInterfaces[interface].interfaceName << " for MPI rank " << mpiRank << std::endl;
                  }
                }
              }
              else {
                if( params.consoleOut ) {
                  //Check values received make sense
                  for( size_t j=0; j<rcvDirectValues.size(); j++ ) {
                    if( rcvDirectValues[j] != rcvValues[interface] ) {
                      if( !params.enableMPI ) {
                        std::cout << "[" << procName << "] " << "Error: Received value not as expected for interface " << muiInterfaces[interface].interfaceName << std::endl;
                      }
                      else {
                        std::cout << "[" << procName << "] " << "Error: Received value not as expected for interface " << muiInterfaces[interface].interfaceName << " for MPI rank " << mpiRank << std::endl;
                      }
                    }
                  }
                }
              }
            }
          }
          else { //Using spatial interpolation to receive values
            //Iterate for as many points as we are to receive through the interface
            for( size_t i=0; i<array3d_rcv[interface].first; ++i ) {
              sendPoint[0] = array3d_rcv[interface].second[i].point[0];
              sendPoint[1] = array3d_rcv[interface].second[i].point[1];
              sendPoint[2] = array3d_rcv[interface].second[i].point[2];

              //Receive value from other side
              array3d_rcv[interface].second[i].value = muiInterfaces[interface].interface->fetch("data", sendPoint, currTime, s1, s2);

              if( params.consoleOut ) {
                //Check value received make sense
                if( array3d_rcv[interface].second[i].value != rcvValues[interface]) {
                  if( !params.enableMPI ) {
                    std::cout << "[" << procName << "] " << "Error: Received value not as expected for interface " << muiInterfaces[interface].interfaceName << std::endl;
                  }
                  else {
                    std::cout << "[" << procName << "] " << "Error: Received value not as expected for interface " << muiInterfaces[interface].interfaceName << " for MPI rank " << mpiRank << std::endl;
                  }
                }
              }
            }
          }
        }
      }
    }

    //Sleep process for pre-defined period of time to simulate work being done by host code
    std::this_thread::sleep_for(std::chrono::milliseconds(params.waitIt));

    if( params.enableMPI ) {
      int err = MPI_Neighbor_alltoall(sendBuf, params.dataToSend, MPI_MB, recvBuf, params.dataToSend, MPI_MB, comm_cart);
      if(err != MPI_SUCCESS) {
        std::cout << "Error when calling MPI_Neighbor_alltoall" << std::endl;
      }
    }
  }

  if( params.consoleOut ) {
    if( !params.enableMPI || (params.enableMPI && mpiRank == 0))
      std::cout << std::endl;
  }

  return true;
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

  if(mpiWorldSize > 1) {
    if(mpiRank == 0 && params.consoleOut){
      std::cout << "[" << procName << "] " << "MPI initialised, there are " << mpiWorldSize << " ranks" << std::endl;
    }
  }

  //Create Cartesian MPI communicator
  int px,py,pz;
  decompose(static_cast<INT>(params.numGridCells[0]), static_cast<INT>(params.numGridCells[1]), static_cast<INT>(params.numGridCells[2]), &px, &py, &pz);
  int num_partition[3] = {px, py, pz};
  int periods[3] = {params.usePeriodic, params.usePeriodic, params.usePeriodic};
  int err = MPI_Cart_create(world, 3, num_partition, periods, false, &comm_cart);
  if(err != MPI_SUCCESS) {
    std::cerr << "Error when creating the communicator." << std::endl;
    return false;
  }

  err = MPI_Cart_coords(comm_cart, mpiRank, 3, mpiCartesianRank);
  if(err != MPI_SUCCESS) {
    std::cerr << "Error when retrieving coordinates." << std::endl;
    return false;
  }

  err = MPI_Type_contiguous(megabyte, MPI_BYTE, &MPI_MB);
  if(err != MPI_SUCCESS) {
    std::cout << "Error when creating MPI_MB" << std::endl;
  }

  err = MPI_Type_commit(&MPI_MB);
  if(err != MPI_SUCCESS) {
    std::cout << "Error when commiting MPI_MB" << std::endl;
  }

  if(params.consoleOut) {
    if(!params.enableMPI || (params.enableMPI && mpiRank == 0)){
      std::cout << "[" << procName << "] Allocating send/recv buffers, filling send with random data." << std::endl;
    }
  }

  // Fill send and receive buffers with random data
  int dataSize = megabyte * params.dataToSend * 6;
  sendBuf = new char[dataSize];
  recvBuf = new char[dataSize];
  std::srand(std::time(nullptr));
	int r = std::rand();
  for(size_t i = 0; i< static_cast<size_t>(dataSize); i++){
    sendBuf[i] = r;
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
  params.totalCells = static_cast<INT>(params.numGridCells[0])*
                      static_cast<INT>(params.numGridCells[1])*
                      static_cast<INT>(params.numGridCells[2]);

  //Calculate size of each grid element
  params.gridSize[0] = (params.domainMax[0] - params.domainMin[0]) / params.numGridCells[0];
  params.gridSize[1] = (params.domainMax[1] - params.domainMin[1]) / params.numGridCells[1];
  params.gridSize[2] = (params.domainMax[2] - params.domainMin[2]) / params.numGridCells[2];

  //Calculate centre of each grid element
  params.gridCentre[0] = static_cast<REAL>(0.5) * params.gridSize[0];
  params.gridCentre[1] = static_cast<REAL>(0.5) * params.gridSize[1];
  params.gridCentre[2] = static_cast<REAL>(0.5) * params.gridSize[2];

  int px,py,pz;
  decompose(static_cast<INT>(params.numGridCells[0]), static_cast<INT>(params.numGridCells[1]), static_cast<INT>(params.numGridCells[2]), &px, &py, &pz);

  int rangeX, rangeY, rangeZ;
  rangeX = static_cast<int>((params.domainMax[0] - params.domainMin[0])) / px;
  rangeY = static_cast<int>((params.domainMax[1] - params.domainMin[1])) / py;
  rangeZ = static_cast<int>((params.domainMax[2] - params.domainMin[2])) / pz;

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
  if(mpiCartesianRank[0] == px-1) {
    partitionEnd[0] = params.domainMax[0] - params.domainMin[0];
		params.itot = static_cast<INT>(params.numGridCells[0]) / px + static_cast<INT>(params.numGridCells[0]) % px;
  }
  else {
    partitionEnd[0]=(mpiCartesianRank[0] + 1) * rangeX;
		params.itot = static_cast<INT>(params.numGridCells[0]) / px;
  }

  if(mpiCartesianRank[1] == py-1) {
    partitionEnd[1] = params.domainMax[1] - params.domainMin[1];
		params.jtot = static_cast<INT>(params.numGridCells[1]) / py + static_cast<INT>(params.numGridCells[1]) % py;
  }
  else {
    partitionEnd[1] = (mpiCartesianRank[1] + 1) * rangeY;
		params.jtot = static_cast<INT>(params.numGridCells[1]) / py;
  }

  if(mpiCartesianRank[2] == pz-1){
    partitionEnd[2] = params.domainMax[2] - params.domainMin[2];
		params.ktot = static_cast<INT>(params.numGridCells[2]) / pz + static_cast<INT>(params.numGridCells[2]) % pz;
  }
  else {
    partitionEnd[2] = (mpiCartesianRank[2] + 1) * rangeZ;
		params.ktot = static_cast<INT>(params.numGridCells[2]) / pz;
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
  if(params.consoleOut) {
    std::cout << "[" << procName << "] " << "Grid points for rank " << mpiRank << ": [" << params.rankDomainMin[0] << "," << params.rankDomainMin[1] << "," << params.rankDomainMin[2] << "] - ["
              << params.rankDomainMax[0] << "," << params.rankDomainMax[1] << "," << params.rankDomainMax[2] << "]" << " MUI: " << sendInterfaces << " send, "
              << rcvInterfaces << " receive" << std::endl;
  }

  //Create grid data
  if(params.consoleOut) {
    if(!params.enableMPI || (params.enableMPI && mpiRank == 0)) //Only perform on master rank if not in serial mode
      std::cout << "[" << procName << "] " << "Populating grid array... ";
  }

  //Create contiguous 3D array of 3D points of type double to send
  array3d_send = create3DArr<pointData>(params.itot, params.jtot, params.ktot);

  for(size_t i=0; i < params.itot; i++) {
    for(size_t j=0; j < params.jtot; j++) {
      for(size_t k=0; k < params.ktot; k++) {
        //Update send array
        array3d_send[i][j][k].point[0] = params.rankDomainMin[0] + static_cast<double>(i * params.gridSize[0]) + params.gridCentre[0];
        array3d_send[i][j][k].point[1] = params.rankDomainMin[1] + static_cast<double>(j * params.gridSize[1]) + params.gridCentre[1];
        array3d_send[i][j][k].point[2] = params.rankDomainMin[2] + static_cast<double>(k * params.gridSize[2]) + params.gridCentre[2];
        array3d_send[i][j][k].value = params.sendValue;

        //Check if the point is within this ranks sending domain, if not set it to disabled
        if(array3d_send[i][j][k].point[0] > sendMin[0] && array3d_send[i][j][k].point[1] > sendMin[1] && array3d_send[i][j][k].point[2] > sendMin[2] &&
           array3d_send[i][j][k].point[0] < sendMax[0] && array3d_send[i][j][k].point[1] < sendMax[1] && array3d_send[i][j][k].point[2] < sendMax[2]){
          array3d_send[i][j][k].enabled = true;
        }else{
          array3d_send[i][j][k].enabled = false;
        }
      }
    }
  }

  if(params.generateCSV) {
    std::ofstream out;
    std::string filename = "partition_" + std::to_string(mpiRank) + ".csv";
    out.open(filename, std::ios::out | std::ios::trunc);
    if(!out.is_open()) {
      std::cerr << "Could not open: " << filename << std::endl;
    }
    for(size_t i=0; i < params.itot; i++) {
      for(size_t j=0; j < params.jtot; j++) {
        for(size_t k=0; k < params.ktot; k++) {
          out << array3d_send[i][j][k].point[0] << ",";
          out << array3d_send[i][j][k].point[1] << ",";
          out << array3d_send[i][j][k].point[2] << "\n";
        }
      }
    }
    out.close();
	}
  if(params.consoleOut) {
    if(!params.enableMPI || (params.enableMPI && mpiRank == 0)){ //Only perform on master rank if not in serial mode
      std::cout << "Done" <<std::endl;
    }
  }

  return true;
}

//*****************************************************************
//* Function to populate data in receive grid array for Double data
//*****************************************************************
void createRcvGridData(size_t interface, std::vector<pointData> &points) {
  //Check if data already exists, if so delete first
  if(array3d_rcv[interface].second != NULL) { //Data is not null
    delete array3d_rcv[interface].second;
    array3d_rcv[interface].first = 0;
  }

  //Create array
  array3d_rcv[interface].first = points.size();
  array3d_rcv[interface].second = new pointData[array3d_rcv[interface].first];

  //Populate receive array
  for (size_t i=0; i<array3d_rcv[interface].first; i++) {
    array3d_rcv[interface].second[i] = points[i];
    array3d_rcv[interface].second[i].value = -DBL_MIN;
  }
}

//****************************************************
//* Function to print information to console
//****************************************************
void printData(parameters& params) {
  if(!params.enableMPI || (params.enableMPI && mpiRank == 0)) { //Only perform on master rank if not in serial mode
    double dataSize = static_cast<double>((params.totalCells * sizeof(pointData)) / megabyte);
    std::cout << "[" << procName << "] Total grid data size: " << dataSize << " MB" << std::endl;
    std::cout << "[" << procName << "] Total cell count: " << params.totalCells << std::endl;
    std::cout << "[" << procName << "] Domain cells: [" << static_cast<INT>(params.numGridCells[0]) << "," << static_cast<INT>(params.numGridCells[1]) << "," << static_cast<INT>(params.numGridCells[2]) << "]" << std::endl;
    std::cout << "[" << procName << "] Domain size: [" << params.domainMin[0] << "," << params.domainMin[1] << "," << params.domainMin[2] << "] - ["
              << params.domainMax[0] << "," << params.domainMax[1] << "," << params.domainMax[2] << "]" << std::endl;

    if(mpiRank == 0) { //Only perform on master rank
      double cellsPerRank = static_cast<double>(params.totalCells) / static_cast<double>(mpiWorldSize); //Calculate cells per rank
      std::cout << "[" << procName << "] " "Cells per rank: " << static_cast<int>(cellsPerRank) << std::endl;
    }

    std::cout << "[" << procName << "] Iterations: " << params.itCount << std::endl;
    std::cout << "[" << procName << "] Send value: " << params.sendValue << std::endl;
    std::cout << "[" << procName << "] Static points: " << (params.staticPoints? "Enabled": "Disabled") << std::endl;
    std::cout << "[" << procName << "] Smart Send: " << (params.smartSend? "Enabled": "Disabled") << std::endl;
    std::cout << "[" << procName << "] Spatial interpolation: " << (params.useInterp? "Disabled": "Enabled") << std::endl;
    std::cout << "[" << procName << "] Work time (ms): " << params.waitIt << std::endl;
  }

  int enabledPts = 0;

  for(size_t i=0; i < params.itot; i++) {
    for(size_t j=0; j < params.jtot; j++) {
      for(size_t k=0; k < params.ktot; k++) {
        if(array3d_send[i][j][k].enabled)
          enabledPts++;
      }
    }
  }

  double sendDataSize = static_cast<double>((enabledPts * sizeof(double)) / megabyte);

  if(params.enableMPI)
    std::cout << "[" << procName << "] " "Data to send for rank " << mpiRank << ": " << sendDataSize << " MB (" << enabledPts << " points)" << std::endl;
  else
    std::cout << "[" << procName << "] " "Data to send: " << sendDataSize << " MB (" << enabledPts << " points)" << std::endl;
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

  //Initialise min and max variables for overall sending and receiving MUI interfaces for this rank to the ranks domain
  sendMin[0] = params.domainMin[0];
  sendMin[1] = params.domainMin[1];
  sendMin[2] = params.domainMin[2];
  sendMax[0]= params.domainMax[0];
  sendMax[1]= params.domainMax[1];
  sendMax[2]= params.domainMax[2];

  rcvMin[0]= params.domainMin[0];
  rcvMin[1] = params.domainMin[1];
  rcvMin[2] = params.domainMin[2];
  rcvMax[0] = params.domainMax[0];
  rcvMax[1] = params.domainMax[1];
  rcvMax[2] = params.domainMax[2];

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

        //Initialise local storage for receiving data
        if(muiInterfaces[j].sendRecv == 1 || muiInterfaces[j].sendRecv == 2) //Interface is receive only or send/receive
          array3d_rcv.resize(array3d_rcv.size()+1, std::pair<size_t, pointData*>(0, NULL));

        //Resize
        if(muiInterfaces[j].sendRecv == 0 || muiInterfaces[j].sendRecv == 2) {   //If the interface is set to send or send & receive
          sendMin[0] = muiInterfaces[j].domMinSend[0];
          sendMin[1] = muiInterfaces[j].domMinSend[1];
          sendMin[2] = muiInterfaces[j].domMinSend[2];
          sendMax[0] = muiInterfaces[j].domMaxSend[0];
          sendMax[1] = muiInterfaces[j].domMaxSend[1];
          sendMax[2] = muiInterfaces[j].domMaxSend[2];

          sendInterfaces++; //Increment enabled send interface count
        }

        if(muiInterfaces[j].sendRecv == 1 || muiInterfaces[j].sendRecv == 2) { //If the interface is set to receive or send & receive
          rcvMin[0] = muiInterfaces[j].domMinRcv[0];
          rcvMin[1] = muiInterfaces[j].domMinRcv[1];
          rcvMin[2] = muiInterfaces[j].domMinRcv[2];
          rcvMax[0] = muiInterfaces[j].domMaxRcv[0];
          rcvMax[1] = muiInterfaces[j].domMaxRcv[1];
          rcvMax[2] = muiInterfaces[j].domMaxRcv[2];

          rcvInterfaces++; //Increment enabled receive interface count
        }

        break;
      }
    }
  }

  return true;
}

//****************************************************
//* Function to finalise before exit
//****************************************************
void finalise(bool usingMPI) {
  //Delete 3D send array
  if( array3d_send != NULL )
    delete3DArr<pointData>(array3d_send);

  //Cleanup receive array and MUI interfaces
  for(size_t interface=0; interface < muiInterfaces.size(); interface++) {
    //Check if data already exists, if so delete first
    if(array3d_rcv[interface].second != NULL) { //Data is not null
      delete[] array3d_rcv[interface].second;
      array3d_rcv[interface].first = 0;
    }

    delete muiInterfaces[interface].interface;
  }

  if( usingMPI ){
    int err = MPI_Type_free(&MPI_MB);
    if(err != MPI_SUCCESS){
      std::cerr << "Error when freeing MPI_MB MPI type" << std::endl;
    }
    delete []sendBuf;
    delete []recvBuf;
  }
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
  inputParams.push_back("USE_INTERPOLATION");
  inputParams.push_back("WAIT_PER_ITERATION");
  inputParams.push_back("DATA_TO_SEND_MPI");
  inputParams.push_back("USE_PERIODIC_PATTERN");

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
            std::cerr << "[" << procName << "] " << "Interface name cannot be empty in MUI input file\n";
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
                std::cerr << "[" << procName << "] " << "Error: Ensure send_or_recv valid in MUI input file" << std::endl;
              return false;
            }
            if(newInterface.sendRecv < 0 || newInterface.sendRecv > 2) {
              if(!usingMPI || (usingMPI && mpiRank == 0)) //Only perform on master rank if not in serial mode
                std::cerr << "[" << procName << "] " << "Error: Ensure send_or_recv valid in MUI input file" << std::endl;
              return false;
            }

            if( !(processPoint(dom_min_ss_send.str(), newInterface.domMinSend)) ) { //Something wrong with value
              if(!usingMPI || (usingMPI && mpiRank == 0)) //Only perform on master rank if not in serial mode
                std::cerr << "[" << procName << "] " << "Error: Ensure send_min valid in MUI input file" << std::endl;
              return false;
            }

            if( !(processPoint(dom_max_ss_send.str(), newInterface.domMaxSend)) ) { //Something wrong with value
              if(!usingMPI || (usingMPI && mpiRank == 0)) //Only perform on master rank if not in serial mode
                std::cerr << "[" << procName << "] " << "Error: Ensure send_max valid in MUI input file" << std::endl;
              return false;
            }

            if( !(processPoint(dom_min_ss_rcv.str(), newInterface.domMinRcv)) ) { //Something wrong with value
              if(!usingMPI || (usingMPI && mpiRank == 0)) //Only perform on master rank if not in serial mode
                std::cerr << "[" << procName << "] " << "Error: Ensure receive_min valid in MUI input file" << std::endl;
              return false;
            }

            if( !(processPoint(dom_max_ss_rcv.str(), newInterface.domMaxRcv)) ) { //Something wrong with value
              if(!usingMPI || (usingMPI && mpiRank == 0)) //Only perform on master rank if not in serial mode
                std::cerr << "[" << procName << "] " << "Error: Ensure receive_max valid in MUI input file" << std::endl;
              return false;
            }

            muiInterfaces.emplace_back(newInterface);
          }
        }
        else { //A problem line in the file
          std::cerr << "[" << procName << "] " << "Error: Incorrect number of values in MUI input file\n";
          return false;
        }
      }
    }

    muiInFile.close();
    return true;
  }
  else {
    std::cerr << "[" << procName << "] " << "Error: Unable to open MUI input file\n";
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
//* Function to create contiguous 3D array
//****************************************************
template <class T> T*** create3DArr(int x, int y, int z) {
  T*** array = new T**[x];
  array[0] = new T*[x*y];
  array[0][0] = new T[x*y*z];

  for(int i = 0; i < x; i++) {
    if(i < x - 1) {
      array[0][(i+1)*y] = &(array[0][0][(i+1)*z*y]);
      array[i+1] = &(array[0][(i+1)*y]);
    }
    for(int j = 0; j < y; j++) {
      if(j > 0)
        array[i][j] = array[i][j-1] + z;
    }
  }

  return array;
}

//****************************************************
//* Function to delete contiguous 3D array
//****************************************************
template <class T> void delete3DArr(T*** array) {
  delete[] array[0][0];
  delete[] array[0];
  delete[] array;
}