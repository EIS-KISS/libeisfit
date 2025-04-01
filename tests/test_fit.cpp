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
#include <string>
#include <vector>

#include <torch/torch.h>

#include "eisfit/fit.h"
#include "fixtures.h"

namespace
{

FitOptions fastOptions()
{
	FitOptions options;
	options.restarts = 8;
	return options;
}

}

TEST_CASE("eisFit recovers a CPE circuit from a clean spectrum", "[fit]")
{
	std::vector<double> truth = {40.0, 5e-5, 0.85};
	torch::Tensor omegas = fixtures::frequencyGrid(60);
	torch::Tensor spectrum = fixtures::simulate("r-p", truth, omegas);

	FitResult result = eisFit(spectrum, omegas, "r-p", fastOptions());

	REQUIRE(result.success);
	REQUIRE(result.verdict != FIT_VERDICT_UNFITTED);
	REQUIRE(result.parameterNames.size() == truth.size());
	REQUIRE(fixtures::maxRelativeError(result.parameters, truth) < 1e-4);
	REQUIRE(result.fittedSpectra.numel() == spectrum.numel());
	REQUIRE(result.rmsRelativeError < 1e-6);

	// uncertainties are one sigma per parameter, finite and positive for a well posed fit
	REQUIRE(result.uncertainties.numel() == static_cast<int64_t>(truth.size()));
	for(int64_t i = 0; i < result.uncertainties.numel(); ++i)
	{
		double sigma = result.uncertainties[i].item<double>();
		REQUIRE(std::isfinite(sigma));
		REQUIRE(sigma > 0.0);
	}
}

TEST_CASE("eisFit recovers a Warburg circuit from a clean spectrum", "[fit]")
{
	std::vector<double> truth = {30.0, 50.0};
	torch::Tensor omegas = fixtures::frequencyGrid(60);
	torch::Tensor spectrum = fixtures::simulate("r-w", truth, omegas);

	FitResult result = eisFit(spectrum, omegas, "r-w", fastOptions());

	REQUIRE(result.success);
	REQUIRE(fixtures::maxRelativeError(result.parameters, truth) < 1e-4);
}

TEST_CASE("eisFit reports a fit verdict on a noisy but correct spectrum", "[fit]")
{
	std::vector<double> truth = {40.0, 5e-5, 0.85};
	torch::Tensor omegas = fixtures::frequencyGrid(60);
	torch::Tensor spectrum = fixtures::addRelativeNoise(
			fixtures::simulate("r-p", truth, omegas), 1e-3, 2024);

	FitResult result = eisFit(spectrum, omegas, "r-p", fastOptions());

	REQUIRE(result.success);
	REQUIRE(result.verdict == FIT_VERDICT_FIT);
	// with real noise the parameters are recovered to a few percent, not to machine precision
	REQUIRE(fixtures::maxRelativeError(result.parameters, truth) < 5e-2);
}

TEST_CASE("eisFit rejects malformed input", "[fit]")
{
	torch::Tensor omegas = fixtures::frequencyGrid(20);
	torch::Tensor spectrum = fixtures::simulate("r-p", {40.0, 5e-5, 0.85}, omegas);

	SECTION("fewer than two points")
	{
		FitResult result = eisFit(torch::tensor({1.0}), torch::tensor({100.0}), "r-p", fastOptions());
		REQUIRE_FALSE(result.success);
	}
	SECTION("spectrum and frequency grid disagree in length")
	{
		FitResult result = eisFit(spectrum, torch::logspace(0.0, 3.0, 5, 10.0,
				torch::TensorOptions().dtype(torch::kFloat64)), "r-p", fastOptions());
		REQUIRE_FALSE(result.success);
	}
}

TEST_CASE("eisFit honours supplied starting parameters", "[fit]")
{
	std::vector<double> truth = {40.0, 5e-5, 0.85};
	torch::Tensor omegas = fixtures::frequencyGrid(60);
	torch::Tensor spectrum = fixtures::simulate("r-p", truth, omegas);

	FitOptions options = fastOptions();
	options.startingParams = torch::tensor(truth);

	FitResult result = eisFit(spectrum, omegas, "r-p", options);
	REQUIRE(result.success);
	REQUIRE(fixtures::maxRelativeError(result.parameters, truth) < 1e-4);
	REQUIRE(result.starts >= 1);
}

TEST_CASE("eisFit is deterministic for a fixed seed", "[fit]")
{
	std::vector<double> truth = {40.0, 5e-5, 0.85};
	torch::Tensor omegas = fixtures::frequencyGrid(60);
	torch::Tensor spectrum = fixtures::simulate("r-p", truth, omegas);

	FitOptions options = fastOptions();
	options.seed = 4242;

	FitResult first = eisFit(spectrum, omegas, "r-p", options);
	FitResult second = eisFit(spectrum, omegas, "r-p", options);

	REQUIRE(first.success);
	REQUIRE(second.success);
	REQUIRE(torch::equal(first.parameters, second.parameters));
}

TEST_CASE("eisFit recovers the same parameters with the exact jacobian", "[fit]")
{
	std::vector<double> truth = {40.0, 5e-5, 0.85};
	torch::Tensor omegas = fixtures::frequencyGrid(60);
	torch::Tensor spectrum = fixtures::simulate("r-p", truth, omegas);

	FitOptions approximate = fastOptions();
	FitOptions exact = fastOptions();
	exact.approximateJacobian = false;

	FitResult approxResult = eisFit(spectrum, omegas, "r-p", approximate);
	FitResult exactResult = eisFit(spectrum, omegas, "r-p", exact);

	REQUIRE(approxResult.success);
	REQUIRE(exactResult.success);
	REQUIRE(fixtures::maxRelativeError(exactResult.parameters, truth) < 1e-4);
	REQUIRE(fixtures::maxRelativeError(approxResult.parameters, exactResult.parameters) < 1e-3);
}

TEST_CASE("eisFit still recovers when the search spectrum is thinned", "[fit]")
{
	std::vector<double> truth = {40.0, 5e-5, 0.85};
	torch::Tensor omegas = fixtures::frequencyGrid(80);
	torch::Tensor spectrum = fixtures::simulate("r-p", truth, omegas);

	FitOptions options = fastOptions();
	options.searchPoints = 30;

	FitResult result = eisFit(spectrum, omegas, "r-p", options);
	REQUIRE(result.success);
	// thinning trades search quality for speed, so allow a looser tolerance
	REQUIRE(fixtures::maxRelativeError(result.parameters, truth) < 1e-2);
	// the reported diagnostics are still computed on the full spectrum
	REQUIRE(result.fittedSpectra.numel() == spectrum.numel());
}
