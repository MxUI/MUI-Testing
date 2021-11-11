/*****************************************************************************
* Multiscale Universal Interface Code Coupling Library                       *
*                                                                            *
* Copyright (C) 2021 Y. H. Tang, S. Kudo, X. Bian, Z. Li, G. E. Karniadakis, *
*                    ^S. M. Longshaw, ^A. Skillen, ^J. Grasset               *
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
 * @file MUI_config.h
 * @author S. M. Longshaw
 * @date 25 October 2021
 * @brief Data structure used in Multiscale Universal Interface Testing
 *        Framework for all created MUI interfaces
 */

#ifndef MUI_TF_CONFIG_H
#define MUI_TF_CONFIG_H

#include "util.h"
#include "dim.h"
#include "exception.h"

namespace mui {
struct tf_config {
  using EXCEPTION = exception_segv;       //- Exception handling type

  static const bool DEBUG = true;         //- Enable extra debugging output
  static const int D = 3;                 //- Dimensionality of the domain
  static const bool FIXEDPOINTS = true;  //- Enable optimisations for problems with fixed point structure
  static const bool QUIET = true;         //- If the library is quiet then it will only issue critical warning messages

  using REAL = double;                    //- REAL data type
  using INT  = int;                       //- INT data type

  using time_type  = INT;                 //- time_type INT for iteration coupling, REAL for exact coupling
  using point_type = point<REAL,D>;       //- "point" data type and dimensionality
  using data_types = type_list<int32_t,
                               int64_t,
                               float,
                               double,
                               std::string
                              >;
};
}
#endif
