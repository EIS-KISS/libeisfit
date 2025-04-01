// libeisfit - Levenberg-Marquardt fitting of equivalent circuit models to measured EIS spectra
// Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
//
// This file is part of libeisfit.
//
// libeisfit is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// libeisfit is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with libeisfit.  If not, see <http://www.gnu.org/licenses/>.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <eisgenerator/model.h>

namespace eisfit_internal
{

// physical lower/upper bounds and a start value per parameter, derived from the ranges the
// model string spells out, widened by `widenDecades` where the string states no explicit range
void deriveBox(eis::Model& model, const std::string& modelString, double widenDecades,
		std::vector<double>& lower, std::vector<double>& upper, std::vector<double>& start,
		int64_t& parameterCount);

}
