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
 * @file MUI_testing.h
 * @author S. M. Longshaw
 * @date 25 October 2021
 * @brief Header file for testing and benchmarking framework for the
 *        Multiscale Universal Interface library
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <math.h>
#include <float.h>
#include <vector>
#include <sstream>
#include <utility>
#include <chrono>
#include <thread>
#include <cstdlib>
#include <iomanip>
#include <sys/types.h>
#include <sys/stat.h>

// MUI specific headers
#include <mui.h>
#include "MUI_config.h"

#define megabyte 1048576

// Data types from MUI_config.h
using TIME = mui::tf_config::time_type;
using REAL = mui::tf_config::REAL;
using INT = mui::tf_config::INT;
using POINT = mui::tf_config::point_type;

//-Struct of input parameters
struct parameters {
  // General parameters
  bool enableMPI;
  POINT domainMin;
  POINT domainMax;
  POINT numGridCells;
  POINT gridSize;
  POINT gridCentre;
  INT totalCells;
  bool consoleOut;
  bool staticPoints;
  bool smartSend;
  bool generateCSV;
  size_t itot;
  size_t jtot;
  size_t ktot;
  POINT rankDomainMin;
  POINT rankDomainMax;
  INT px;
  INT py;
  INT pz;

  // MUI interface parameters
  std::string domainName;
  std::string interfaceFilePath;

  // Artificial Simulation Parameters
  INT itCount;
  REAL sendValue;
  bool useInterp;
  INT interpMode;
  INT waitIt;
  INT dataToSend;
  INT pushFetchOrder;
  bool usePeriodic;
  INT numMUIValues;

  parameters() :
    enableMPI(false),
    domainMin(0,0,0),
    domainMax(0,0,0),
    numGridCells(0,0,0),
    gridSize(0,0,0),
    gridCentre(0,0,0),
    px(0),
    py(0),
    pz(0),
    totalCells(0),
    consoleOut(false),
    staticPoints(false),
    smartSend(false),
    generateCSV(false),
    itot(0),
    jtot(0),
    ktot(0),
    domainName(),
    interfaceFilePath(),
    itCount(0),
    sendValue(0),
    useInterp(false),
    interpMode(-1),
    waitIt(0),
    dataToSend(0),
    pushFetchOrder(-1),
    usePeriodic(false),
    numMUIValues(0),
    checkValues(false)
  {}
};

//-Struct to hold details of created MUI interface
struct muiInterface {
  std::string interfaceName; //MUI interface (protocol(def=MPI)://domainName//[interfaceName]
  mui::uniface<mui::tf_config>* interface; //Pointer to MUI interface
  int sendRecv; //Define whether interface is for sending (0), receiving (1) or both (2)
  POINT domMinSend; //The minimum of the domain for the interface to send
  POINT domMaxSend; //The maximum of the domain for the interface to send
  POINT domMinRcv; //The minimum of the domain for the interface to receive
  POINT domMaxRcv; //The maximum of the domain for the interface to receive

  muiInterface() :
    interfaceName(),
    interface(nullptr),
    sendRecv(-1),
    domMinSend({0,0,0}),
    domMaxSend({0,0,0}),
    domMinRcv({0,0,0}),
    domMaxRcv({0,0,0})
  {}
};

struct pointData {
  POINT point;
  REAL value;

  pointData() :
    point({0,0,0}),
    value(-DBL_MIN)
  {}
};

struct timing {
  double totalTime;
  double muiTime;

  timing() :
    totalTime(0),
    muiTime(0)
  {}
};

int mpiWorldSize = 1; //-Number of MPI processes
int mpiRank; //-Rank of this MPI process
int mpiCartesianRank[3] = {0, 0, 0}; //-Coordinates of this MPI process in Cartesian topology
MPI_Datatype MPI_MB;
char *sendBuf, *recvBuf;
std::string procName; //-String to hold processor name as returned by MPI
std::string outName; //-String to hold processor name as returned by MPI
std::vector<pointData> sendRcvPoints; //- Points to send via MUI
std::vector<std::vector<bool>> sendEnabled; //-Vector to hold whether a point is enabled to send for an interface
std::vector<std::vector<bool>> rcvEnabled; //-Vector to hold whether a point is enabled to receive for an interface
int sendInterfaces, rcvInterfaces; //-Count of the number of send and receive MUI interfaces on this rank

//MUI Data
MPI_Comm world, comm_cart;
std::vector<muiInterface> muiInterfaces;

//Function declarations
template<bool, bool> timing run(parameters&);
bool initMPI(int, char**, parameters&);
void calculateGridValues(parameters&);
bool createGridData(parameters&);
void printData(parameters&);
bool createMUIInterfaces(std::string&, parameters&);
void decompose(int, int, int, int*, int*, int*);
void finalise(bool);
bool readConfig(std::string&, parameters&);
bool readInterfaces(std::string& fileName, bool usingMPI);
bool processPoint(const std::string&, POINT&);
template <typename T> inline bool intersectPoint(POINT&, mui::geometry::box<T>&);
template <typename T> inline bool intersectBox(mui::geometry::box<T>&, mui::geometry::box<T>&);
template <typename T> inline bool almostEqual(T x, T y);
