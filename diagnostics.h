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

#include <torch/torch.h>

#include "eisfit/fit.h"

namespace eisfit_internal
{

// relative noise floor estimated from fourth and sixth differences of the spectrum,
// zero when the two estimates disagree
double estimateNoiseFloor(const torch::Tensor& target, const torch::Tensor& magnitude);

double lagOneAutocorrelation(const torch::Tensor& sequence);

// sets goodFitLimit, poorFitLimit and verdict on the result from its residual diagnostics
void applyVerdict(FitResult& result, double fitLimit);

}
