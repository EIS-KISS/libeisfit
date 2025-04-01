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

#include "util.h"

#include <iomanip>
#include <sstream>

namespace eisfit_internal
{

torch::Tensor doubles(const std::vector<double>& values)
{
	return torch::tensor(values).to(torch::kFloat64);
}

torch::Tensor thinIndices(int64_t points, int64_t limit)
{
	if(limit <= 0 or limit >= points or limit < 3)
		return torch::Tensor();
	std::vector<int64_t> idx;
	idx.reserve(limit);
	for(int64_t k = 0; k < limit; ++k)
		idx.push_back(k*(points-1)/(limit-1));
	idx.back() = points-1;
	return torch::tensor(idx);
}

std::string formatValue(double value)
{
	std::ostringstream stream;
	stream<<std::setprecision(10)<<value;
	return stream.str();
}

}
