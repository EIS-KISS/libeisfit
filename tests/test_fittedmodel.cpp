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

#include <eisgenerator/model.h>

#include "eisfit/fit.h"

TEST_CASE("fittedModelString embeds single and multi element circuits", "[fittedmodel]")
{
	REQUIRE(fittedModelString("r", torch::tensor({50.0})) == "r{50}");
	REQUIRE(fittedModelString("r-r", torch::tensor({10.0, 20.0})) == "r{10}-r{20}");
}

TEST_CASE("fittedModelString embeds a two parameter element", "[fittedmodel]")
{
	std::string out = fittedModelString("r-p",
			torch::tensor({40.0, 2e-4, 0.9}, torch::TensorOptions().dtype(torch::kFloat64)));
	REQUIRE(out.rfind("r{40}", 0) == 0);
	REQUIRE(out.find("0.0002") != std::string::npos);
	REQUIRE(out.find("0.9") != std::string::npos);
}

TEST_CASE("fittedModelString replaces a range already present in the string", "[fittedmodel]")
{
	REQUIRE(fittedModelString("r{1~100}", torch::tensor({50.0})) == "r{50}");
}

TEST_CASE("fittedModelString rejects a parameter count mismatch", "[fittedmodel]")
{
	REQUIRE(fittedModelString("r-p", torch::tensor({1.0, 2.0})) == "");
	REQUIRE(fittedModelString("r", torch::tensor({1.0, 2.0})) == "");
	REQUIRE(fittedModelString("r-r", torch::tensor({1.0})) == "");
}

TEST_CASE("fittedModelString rejects an undefined tensor", "[fittedmodel]")
{
	REQUIRE(fittedModelString("r", torch::Tensor()) == "");
}

TEST_CASE("fittedModelString round trips through the generator", "[fittedmodel]")
{
	std::vector<double> truth = {40.0, 5e-5, 0.85};

	std::string embedded = fittedModelString("r-p",
			torch::tensor(truth, torch::TensorOptions().dtype(torch::kFloat64)));
	REQUIRE_FALSE(embedded.empty());

	// the embedded string parses and carries the same number of parameters
	eis::Model reparsed(embedded);
	REQUIRE(reparsed.getParameterCount() == truth.size());

	// reading the parameters back out of the re-parsed string recovers the values; the generator
	// stores them as float32, so the round trip is only accurate to float precision
	reparsed.resolveSteps(0);
	auto recovered = reparsed.getFlatParameters();
	REQUIRE(recovered.size() == truth.size());
	for(size_t i = 0; i < truth.size(); ++i)
		REQUIRE(std::abs(static_cast<double>(recovered[i])-truth[i]) <= 1e-5*std::abs(truth[i]));
}
