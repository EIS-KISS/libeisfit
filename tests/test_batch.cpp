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

// stack one row per simulated spectrum into a {B, N} complex tensor
torch::Tensor batchOf(const std::string& modelString, const std::vector<std::vector<double>>& truths,
		const torch::Tensor& omegas)
{
	std::vector<torch::Tensor> rows;
	for(const std::vector<double>& truth: truths)
		rows.push_back(fixtures::simulate(modelString, truth, omegas).reshape({1, -1}));
	return torch::cat(rows, 0);
}

}

TEST_CASE("eisFitBatch recovers one parameter set per row", "[batch]")
{
	std::vector<std::vector<double>> truths = {
		{40.0, 5e-5, 0.85},
		{80.0, 1e-4, 0.95},
		{20.0, 2e-5, 0.75},
	};
	torch::Tensor omegas = fixtures::frequencyGrid(60);
	torch::Tensor batch = batchOf("r-p", truths, omegas);

	std::vector<FitResult> results = eisFitBatch(batch, omegas, "r-p", fastOptions());

	REQUIRE(results.size() == truths.size());
	for(size_t b = 0; b < truths.size(); ++b)
	{
		INFO("row " << b);
		REQUIRE(results[b].success);
		REQUIRE(results[b].verdict != FIT_VERDICT_UNFITTED);
		REQUIRE(fixtures::maxRelativeError(results[b].parameters, truths[b]) < 1e-4);
		REQUIRE(results[b].fittedSpectra.numel() == omegas.numel());
	}
}

TEST_CASE("eisFitBatch agrees with single fits", "[batch]")
{
	std::vector<std::vector<double>> truths = {{40.0, 5e-5, 0.85}, {70.0, 8e-5, 0.9}};
	torch::Tensor omegas = fixtures::frequencyGrid(60);
	torch::Tensor batch = batchOf("r-p", truths, omegas);

	std::vector<FitResult> batched = eisFitBatch(batch, omegas, "r-p", fastOptions());
	REQUIRE(batched.size() == truths.size());

	for(size_t b = 0; b < truths.size(); ++b)
	{
		FitResult single = eisFit(batch[static_cast<int64_t>(b)], omegas, "r-p", fastOptions());
		REQUIRE(single.success);
		REQUIRE(batched[b].success);
		REQUIRE(fixtures::maxRelativeError(batched[b].parameters, single.parameters) < 1e-3);
	}
}

TEST_CASE("eisFitBatch of a single row matches eisFit", "[batch]")
{
	std::vector<double> truth = {40.0, 5e-5, 0.85};
	torch::Tensor omegas = fixtures::frequencyGrid(60);
	torch::Tensor spectrum = fixtures::simulate("r-p", truth, omegas);

	FitResult single = eisFit(spectrum, omegas, "r-p", fastOptions());
	std::vector<FitResult> batched = eisFitBatch(spectrum.reshape({1, -1}), omegas, "r-p", fastOptions());

	REQUIRE(batched.size() == 1);
	REQUIRE(single.success);
	REQUIRE(batched[0].success);
	REQUIRE(fixtures::maxRelativeError(batched[0].parameters, single.parameters) < 1e-3);
}

TEST_CASE("eisFitBatch accepts shared and per-row starting parameters", "[batch]")
{
	std::vector<std::vector<double>> truths = {{40.0, 5e-5, 0.85}, {70.0, 8e-5, 0.9}};
	torch::Tensor omegas = fixtures::frequencyGrid(60);
	torch::Tensor batch = batchOf("r-p", truths, omegas);

	SECTION("one start shared by the batch")
	{
		FitOptions options = fastOptions();
		options.startingParams = torch::tensor({40.0, 5e-5, 0.85});
		std::vector<FitResult> results = eisFitBatch(batch, omegas, "r-p", options);
		REQUIRE(results.size() == truths.size());
		for(size_t b = 0; b < truths.size(); ++b)
			REQUIRE(fixtures::maxRelativeError(results[b].parameters, truths[b]) < 1e-4);
	}
	SECTION("one start per row")
	{
		FitOptions options = fastOptions();
		// {n, B}: column b is the start for row b
		options.startingParams = torch::tensor({{40.0, 70.0}, {5e-5, 8e-5}, {0.85, 0.9}});
		std::vector<FitResult> results = eisFitBatch(batch, omegas, "r-p", options);
		REQUIRE(results.size() == truths.size());
		for(size_t b = 0; b < truths.size(); ++b)
			REQUIRE(fixtures::maxRelativeError(results[b].parameters, truths[b]) < 1e-4);
	}
}

TEST_CASE("eisFitBatch rejects malformed input", "[batch]")
{
	torch::Tensor omegas = fixtures::frequencyGrid(20);

	SECTION("empty batch yields no results")
	{
		torch::Tensor empty = torch::zeros({0, omegas.numel()},
				torch::TensorOptions().dtype(torch::kComplexDouble));
		std::vector<FitResult> results = eisFitBatch(empty, omegas, "r-p", fastOptions());
		REQUIRE(results.empty());
	}
	SECTION("frequency grid disagrees with the row length")
	{
		torch::Tensor batch = batchOf("r-p", {{40.0, 5e-5, 0.85}}, omegas);
		torch::Tensor wrongGrid = torch::logspace(0.0, 3.0, 5, 10.0,
				torch::TensorOptions().dtype(torch::kFloat64));
		std::vector<FitResult> results = eisFitBatch(batch, wrongGrid, "r-p", fastOptions());
		REQUIRE(results.empty());
	}
}
