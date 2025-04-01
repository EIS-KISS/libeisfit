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

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <torch/torch.h>

#include <eisgenerator/model.h>

#include "modelscript.h"
#include "util.h"

namespace fixtures
{

// logarithmic frequency grid in rad/s spanning `decades` between 10^lo and 10^hi Hz
inline torch::Tensor frequencyGrid(int64_t points, double lo = -3.0, double hi = 5.0)
{
	return torch::logspace(lo, hi, points, 10.0, torch::TensorOptions().dtype(torch::kFloat64))*2.0*M_PI;
}

// simulate a noiseless spectrum for a fully parameterised model string, using the same
// TorchScript path the fit minimises against, so a round trip is self-consistent
inline torch::Tensor simulate(const std::string& modelString, const std::vector<double>& parameters,
		const torch::Tensor& omegas)
{
	eis::Model model(modelString);
	std::shared_ptr<torch::jit::CompilationUnit> script = eisfit_internal::compileModel(model);
	return eisfit_internal::runScriptModel(model, script, eisfit_internal::doubles(parameters),
			omegas.to(torch::kFloat64).reshape({omegas.numel(), 1}));
}

// multiplicative complex gaussian noise of the given relative amplitude, deterministic in seed
inline torch::Tensor addRelativeNoise(const torch::Tensor& spectrum, double relative, uint64_t seed)
{
	torch::manual_seed(seed);
	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat64);
	torch::Tensor noise = torch::complex(torch::randn({spectrum.numel()}, options),
			torch::randn({spectrum.numel()}, options));
	return spectrum*(1.0+relative*noise);
}

inline double relativeError(double value, double truth)
{
	double scale = std::max(std::abs(truth), 1e-300);
	return std::abs(value-truth)/scale;
}

inline bool relativeClose(double value, double truth, double tolerance)
{
	return relativeError(value, truth) <= tolerance;
}

// maximum relative deviation between a fitted parameter vector and the truth
inline double maxRelativeError(const torch::Tensor& parameters, const std::vector<double>& truth)
{
	REQUIRE(parameters.numel() == static_cast<int64_t>(truth.size()));
	auto accessor = parameters.accessor<double, 1>();
	double worst = 0;
	for(size_t i = 0; i < truth.size(); ++i)
		worst = std::max(worst, relativeError(accessor[static_cast<int64_t>(i)], truth[i]));
	return worst;
}

// maximum relative deviation between two fitted parameter vectors of equal length
inline double maxRelativeError(const torch::Tensor& parameters, const torch::Tensor& reference)
{
	REQUIRE(parameters.numel() == reference.numel());
	auto left = parameters.accessor<double, 1>();
	auto right = reference.accessor<double, 1>();
	double worst = 0;
	for(int64_t i = 0; i < parameters.numel(); ++i)
		worst = std::max(worst, relativeError(left[i], right[i]));
	return worst;
}

}
