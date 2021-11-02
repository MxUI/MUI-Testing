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

  // MUI interface parameters
  std::string domainName;
  std::string interfaceFilePath;

  // Artificial Simulation Parameters
  INT itCount;
  REAL sendValue;
  bool useInterp;
  INT waitIt;
  INT dataToSend;
  bool usePeriodic;

  parameters() :
    enableMPI(false),
    domainMin(0,0,0),
    domainMax(0,0,0),
    numGridCells(0,0,0),
    gridSize(0,0,0),
    gridCentre(0,0,0),
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
    waitIt(0),
    dataToSend(0),
    usePeriodic(false)
  {}
};

//-Struct to hold details of created MUI interface
struct muiInterface {
  std::string interfaceName; //MUI interface (protocol(def=MPI)://domainName//[interfaceName]
  mui::uniface<mui::tf_config> *interface; //Pointer to MUI interface
  int sendRecv; //Define whether interface is for sending (0), receiving (1) or both (2)
  POINT domMinSend; //The minimum of the domain for the interface to send
  POINT domMaxSend; //The maximum of the domain for the interface to send
  POINT domMinRcv; //The minimum of the domain for the interface to receive
  POINT domMaxRcv; //The maximum of the domain for the interface to receive
};

struct pointData {
  POINT point;
  bool enabled;
  REAL value;
};

//bool usingMPI; //-Is run MPI parallelised
//bool enableOutput; //-Enable or disable application console output
//bool staticPoints; //-Enable or disable the use of static grid points (only send point values for first frame)
int mpiWorldSize = 1; //-Number of MPI processes
int mpiRank; //-Rank of this MPI process
int mpiCartesianRank[3] = {0, 0, 0}; //-Coordinates of this MPI process in Cartesian topology
//const int MB = 1024*1024;
MPI_Datatype MPI_MB;
char *sendBuf, *recvBuf;
//int cellsx, cellsy, cellsz; //-Dimensions of the problem grid
//double domx_min, domy_min, domz_min; //-Minimum dimensions of the computational domain
//double domx_max, domy_max, domz_max; //-Maximum dimensions of the computational domain
//int iterations; //-Number of iterations to perform
//std::string muifile; //-Filename of the file containing MUI interfaces
//double sendvalue; //-The value to be sent via MUI
double tStart, tEnd; //-MPI wall-time storage
//bool usingSmartSend; //-Flag to determine whether to use smart sending capability of MUI or not
std::string procName; //-String to hold processor name as returned by MPI
//bool usingDirectReceive; //-Flag to determine whether values are retrieved directly through the interface or using interpolation
//int sleepTime; //-The amount of time each iteration will wait to simulate work that would be done by a real solver
//int amountToSend; //-The amount of data that each rank will send to each of its neighbor
//bool isPeriodic; //-Are the communications done on a periodic mesh ?
//bool generateCSV; //-Should CSV files be generated ?

//Grid Data
//int totalCells; //- Total number of cells in grid
//double gridsizex, gridsizey, gridsizez; //-Size of a grid cell in each direction
//double gridcentrex, gridcentrey, gridcentrez; //-Centre of a grid cell
//size_t itot, jtot, ktot; //-Total number of cells in each direction per rank
pointData*** array3d_send; //-3D contiguous array of points to send via MUI
std::vector<std::pair<size_t, pointData* > > array3d_rcv; //-1D contiguous arrays of points to receive via MUI inerfaces
//double domXMin, domXMax; //-The min and max of this ranks domain in the x direction
//double domYMin, domYMax; //-The min and max of this ranks domain in the y direction
//double domZMin, domZMax; //-The min and max of this ranks domain in the z direction
POINT sendMin, sendMax; //-The min and max of all MUI sending interfaces for this rank
POINT rcvMin, rcvMax; //-The min and max of all MUI receiving interfaces for this rank
int sendInterfaces, rcvInterfaces; //-Count of the number of send and receive MUI interfaces on this rank

//MUI Data
MPI_Comm world, comm_cart;
std::vector<muiInterface> muiInterfaces;

//Function declarations
bool run(parameters&);
bool initMPI(int, char**, parameters&);
void calculateGridValues(parameters&);
bool createGridData(parameters&);
void createRcvGridData(size_t, std::vector<pointData>&);
void printData(parameters&);
bool createMUIInterfaces(std::string&, parameters&);
void decompose(int, int, int, int*, int*, int*);
void finalise(bool);
bool readConfig(std::string&, parameters&);
bool readInterfaces(std::string& fileName, bool usingMPI);
bool processPoint(const std::string&, POINT&);
template <class T> T*** create3DArr(int, int, int);
template <class T> void delete3DArr(T***);
