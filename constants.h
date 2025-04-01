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

namespace eisfit_internal
{

constexpr double FIXED_WIDENING_DECADES = 3;
constexpr double SWEEP_WIDENING_DECADES = 6;
constexpr double TINY_PARAMETER = 1e-300;
constexpr double BOUND_SLACK = 1e-3;
constexpr double NOISE_AGREEMENT = 2.0;
constexpr double POOR_FIT_FACTOR = 5;
constexpr double CORRELATED_RESIDUAL = 0.8;

}
