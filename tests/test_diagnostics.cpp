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

#include <catch2/catch.hpp>

#include <cmath>
#include <vector>

#include <torch/torch.h>

#include "diagnostics.h"

using namespace eisfit_internal;

TEST_CASE("lagOneAutocorrelation guards short and flat sequences", "[diagnostics]")
{
	REQUIRE(lagOneAutocorrelation(torch::tensor({1.0, 2.0})) == 0.0);
	REQUIRE(lagOneAutocorrelation(torch::tensor({3.0, 3.0, 3.0, 3.0})) == 0.0);
}

TEST_CASE("lagOneAutocorrelation is positive for a smooth ramp", "[diagnostics]")
{
	std::vector<double> ramp;
	for(int i = 0; i < 40; ++i)
		ramp.push_back(static_cast<double>(i));
	REQUIRE(lagOneAutocorrelation(torch::tensor(ramp)) > 0.4);
}

TEST_CASE("lagOneAutocorrelation is negative for an alternating sequence", "[diagnostics]")
{
	std::vector<double> alternating;
	for(int i = 0; i < 40; ++i)
		alternating.push_back(i % 2 ? -1.0 : 1.0);
	REQUIRE(lagOneAutocorrelation(torch::tensor(alternating)) < -0.5);
}

TEST_CASE("estimateNoiseFloor guards short spectra", "[diagnostics]")
{
	torch::Tensor short8 = torch::randn({8}, torch::TensorOptions().dtype(torch::kFloat64));
	REQUIRE(estimateNoiseFloor(short8, torch::abs(short8)) == 0.0);
}

TEST_CASE("estimateNoiseFloor is zero for a smooth spectrum", "[diagnostics]")
{
	// a low order polynomial has vanishing fourth and sixth differences
	std::vector<double> smooth;
	for(int i = 0; i < 40; ++i)
	{
		double x = static_cast<double>(i);
		smooth.push_back(1.0+0.5*x-0.01*x*x);
	}
	torch::Tensor target = torch::tensor(smooth, torch::TensorOptions().dtype(torch::kFloat64));
	REQUIRE(estimateNoiseFloor(target, torch::abs(target)) == Approx(0.0).margin(1e-9));
}

TEST_CASE("estimateNoiseFloor detects white noise and grows with its amplitude", "[diagnostics]")
{
	torch::manual_seed(7);
	torch::TensorOptions options = torch::TensorOptions().dtype(torch::kFloat64);
	torch::Tensor base = torch::logspace(0.0, 2.0, 60, 10.0, options);
	torch::Tensor magnitude = torch::abs(base);

	torch::Tensor quiet = base+0.01*torch::randn({60}, options);
	torch::Tensor loud = base+0.10*torch::randn({60}, options);

	double quietFloor = estimateNoiseFloor(quiet, magnitude);
	double loudFloor = estimateNoiseFloor(loud, magnitude);

	REQUIRE(quietFloor > 0.0);
	REQUIRE(loudFloor > quietFloor);
}

namespace
{

FitResult makeResult(double rms, double noiseFloor, double correlation)
{
	FitResult result;
	result.success = true;
	result.rmsRelativeError = rms;
	result.noiseFloor = noiseFloor;
	result.residualCorrelation = correlation;
	return result;
}

}

TEST_CASE("applyVerdict classifies by residual size", "[diagnostics]")
{
	SECTION("well below the limit fits")
	{
		FitResult result = makeResult(0.001, 0.0, 0.0);
		applyVerdict(result, 0.01);
		REQUIRE(result.verdict == FIT_VERDICT_FIT);
		REQUIRE(result.goodFitLimit == Approx(0.01));
		REQUIRE(result.poorFitLimit == Approx(0.05));
	}
	SECTION("between the limits is marginal")
	{
		FitResult result = makeResult(0.02, 0.0, 0.0);
		applyVerdict(result, 0.01);
		REQUIRE(result.verdict == FIT_VERDICT_MARGINAL);
	}
	SECTION("above the poor limit does not fit")
	{
		FitResult result = makeResult(0.10, 0.0, 0.0);
		applyVerdict(result, 0.01);
		REQUIRE(result.verdict == FIT_VERDICT_UNFITTED);
	}
}

TEST_CASE("applyVerdict demotes a small but correlated residual", "[diagnostics]")
{
	FitResult result = makeResult(0.001, 0.0, 0.9);
	applyVerdict(result, 0.01);
	REQUIRE(result.verdict == FIT_VERDICT_MARGINAL);
}

TEST_CASE("applyVerdict raises its limits with the noise floor", "[diagnostics]")
{
	// with a noise floor of 0.01 the good limit becomes 3x and the poor limit 10x of it
	FitResult result = makeResult(0.02, 0.01, 0.0);
	applyVerdict(result, 0.01);
	REQUIRE(result.goodFitLimit == Approx(0.03));
	REQUIRE(result.poorFitLimit == Approx(0.10));
	// the same residual that was marginal against a zero noise floor now fits
	REQUIRE(result.verdict == FIT_VERDICT_FIT);
}
