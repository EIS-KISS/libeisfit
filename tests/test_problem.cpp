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

#include "batchproblem.h"
#include "bounds.h"
#include "fixtures.h"
#include "problem.h"
#include "util.h"

using namespace eisfit_internal;

namespace
{

// the weight the fit applies to residuals, replicated here so the white box tests match
torch::Tensor weightScale(const torch::Tensor& magnitude, double weightFloor, int64_t points)
{
	torch::Tensor floor = torch::mean(magnitude)*weightFloor;
	torch::Tensor weight = 1.0/torch::sqrt(magnitude*magnitude+floor*floor);
	return torch::sqrt(weight/points);
}

std::vector<double> toLog(const std::vector<double>& physical)
{
	std::vector<double> out;
	for(double value: physical)
		out.push_back(std::log10(value));
	return out;
}

// relative Frobenius norm of the difference between two jacobians
double relativeDifference(const torch::Tensor& a, const torch::Tensor& b)
{
	return (a-b).norm().item<double>()/b.norm().item<double>();
}

}

TEST_CASE("Problem residual and cost vanish at the truth", "[problem]")
{
	std::vector<double> truth = {40.0, 5e-5, 0.85};
	torch::Tensor omegas = fixtures::frequencyGrid(50);

	Problem problem("r-p");
	problem.omegas = omegas.to(torch::kFloat64).reshape({omegas.numel(), 1});
	problem.target = fixtures::simulate("r-p", truth, omegas);

	std::vector<double> lower, upper, start;
	deriveBox(problem.model, problem.modelString, 0, lower, upper, start, problem.parameterCount);
	problem.logLower = doubles(lower).log10();
	problem.logUpper = doubles(upper).log10();
	problem.scale = weightScale(torch::abs(problem.target), 1e-3, problem.target.numel());

	torch::Tensor x = doubles(toLog(truth));

	// the truth is not exactly representable through pow(10, log10), so the residual is at
	// the level of that round trip rather than machine zero
	REQUIRE(problem.residual(x).norm().item<double>() < 1e-3);
	REQUIRE(problem.cost(x) < 1e-6);
}

TEST_CASE("Problem cost grows away from the truth", "[problem]")
{
	std::vector<double> truth = {40.0, 5e-5, 0.85};
	torch::Tensor omegas = fixtures::frequencyGrid(50);

	Problem problem("r-p");
	problem.omegas = omegas.to(torch::kFloat64).reshape({omegas.numel(), 1});
	problem.target = fixtures::simulate("r-p", truth, omegas);
	std::vector<double> lower, upper, start;
	deriveBox(problem.model, problem.modelString, 0, lower, upper, start, problem.parameterCount);
	problem.logLower = doubles(lower).log10();
	problem.logUpper = doubles(upper).log10();
	problem.scale = weightScale(torch::abs(problem.target), 1e-3, problem.target.numel());

	double atTruth = problem.cost(doubles(toLog(truth)));
	std::vector<double> off = truth;
	off[0] *= 2.0;
	double atOff = problem.cost(doubles(toLog(off)));
	REQUIRE(atOff > 100.0*atTruth);
}

TEST_CASE("Problem clampToBox clamps to the log bounds", "[problem]")
{
	Problem problem("r-p");
	std::vector<double> lower, upper, start;
	deriveBox(problem.model, problem.modelString, 0, lower, upper, start, problem.parameterCount);
	problem.logLower = doubles(lower).log10();
	problem.logUpper = doubles(upper).log10();

	torch::Tensor way = doubles({-1e6, 1e6, 0.0});
	torch::Tensor clamped = problem.clampToBox(way);
	auto accessor = clamped.accessor<double, 1>();
	REQUIRE(accessor[0] == Approx(problem.logLower[0].item<double>()));
	REQUIRE(accessor[1] == Approx(problem.logUpper[1].item<double>()));
}

TEST_CASE("Problem approximate jacobian matches the exact one", "[problem]")
{
	std::vector<double> truth = {40.0, 5e-5, 0.85};
	torch::Tensor omegas = fixtures::frequencyGrid(50);

	Problem problem("r-p");
	problem.omegas = omegas.to(torch::kFloat64).reshape({omegas.numel(), 1});
	problem.target = fixtures::simulate("r-p", truth, omegas);
	std::vector<double> lower, upper, start;
	deriveBox(problem.model, problem.modelString, 0, lower, upper, start, problem.parameterCount);
	problem.logLower = doubles(lower).log10();
	problem.logUpper = doubles(upper).log10();
	problem.scale = weightScale(torch::abs(problem.target), 1e-3, problem.target.numel());

	torch::Tensor x = doubles(toLog(truth));

	problem.approximateJacobian = true;
	torch::Tensor approx = problem.jacobian(x);
	problem.approximateJacobian = false;
	torch::Tensor exact = problem.jacobian(x);

	REQUIRE(approx.sizes() == exact.sizes());
	REQUIRE(approx.size(0) == 2*problem.target.numel());
	REQUIRE(approx.size(1) == problem.parameterCount);
	REQUIRE(relativeDifference(approx, exact) < 1e-3);
}

TEST_CASE("BatchProblem approximate jacobian matches the exact one", "[batchproblem]")
{
	// truths sit strictly inside the derived box: at a bound the clamp makes the exact gradient
	// (boundary inclusive) and the one sided forward difference disagree, which is not what this
	// test is about
	std::vector<std::vector<double>> truths = {{40.0, 5e-5, 0.85}, {80.0, 3e-5, 0.80}};
	torch::Tensor omegas = fixtures::frequencyGrid(40);
	const int64_t batch = static_cast<int64_t>(truths.size());

	BatchProblem problem("r-p");
	problem.batch = batch;
	problem.omegas = omegas.to(torch::kFloat64).reshape({omegas.numel(), 1});

	std::vector<torch::Tensor> columns;
	for(const std::vector<double>& truth: truths)
		columns.push_back(fixtures::simulate("r-p", truth, omegas).reshape({-1, 1}));
	problem.target = torch::cat(columns, 1);

	std::vector<double> lower, upper, start;
	deriveBox(problem.model, problem.modelString, 0, lower, upper, start, problem.parameterCount);
	problem.logLower = doubles(lower).reshape({problem.parameterCount, 1}).log10();
	problem.logUpper = doubles(upper).reshape({problem.parameterCount, 1}).log10();

	torch::Tensor magnitude = torch::abs(problem.target);
	torch::Tensor floor = torch::mean(magnitude, 0, true)*1e-3;
	torch::Tensor weight = 1.0/torch::sqrt(magnitude*magnitude+floor*floor);
	problem.scale = torch::sqrt(weight/problem.target.size(0));

	std::vector<torch::Tensor> logColumns;
	for(const std::vector<double>& truth: truths)
		logColumns.push_back(doubles(toLog(truth)).reshape({-1, 1}));
	torch::Tensor x = torch::cat(logColumns, 1);

	problem.approximateJacobian = true;
	torch::Tensor approx = problem.jacobian(x);
	problem.approximateJacobian = false;
	torch::Tensor exact = problem.jacobian(x);

	REQUIRE(approx.sizes() == exact.sizes());
	REQUIRE(approx.size(0) == 2*problem.target.size(0));
	REQUIRE(approx.size(1) == problem.parameterCount);
	REQUIRE(approx.size(2) == batch);
	REQUIRE(relativeDifference(approx, exact) < 1e-3);
}
