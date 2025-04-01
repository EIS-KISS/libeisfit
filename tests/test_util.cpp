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

#include <cstdlib>
#include <string>
#include <vector>

#include <torch/torch.h>

#include "util.h"

using namespace eisfit_internal;

TEST_CASE("doubles builds a float64 tensor", "[util]")
{
	std::vector<double> values = {1.5, -2.25, 3.0, 0.0, 1e-9};
	torch::Tensor tensor = doubles(values);

	REQUIRE(tensor.scalar_type() == torch::kFloat64);
	REQUIRE(tensor.numel() == static_cast<int64_t>(values.size()));
	REQUIRE(tensor.dim() == 1);

	auto accessor = tensor.accessor<double, 1>();
	for(size_t i = 0; i < values.size(); ++i)
		REQUIRE(accessor[static_cast<int64_t>(i)] == Approx(values[i]));
}

TEST_CASE("doubles of an empty vector", "[util]")
{
	torch::Tensor tensor = doubles({});
	REQUIRE(tensor.numel() == 0);
}

TEST_CASE("thinIndices returns undefined when no thinning is wanted", "[util]")
{
	SECTION("limit zero or negative")
	{
		REQUIRE_FALSE(thinIndices(100, 0).defined());
		REQUIRE_FALSE(thinIndices(100, -5).defined());
	}
	SECTION("limit at or above the point count")
	{
		REQUIRE_FALSE(thinIndices(100, 100).defined());
		REQUIRE_FALSE(thinIndices(100, 200).defined());
	}
	SECTION("limit below three points")
	{
		REQUIRE_FALSE(thinIndices(100, 1).defined());
		REQUIRE_FALSE(thinIndices(100, 2).defined());
	}
}

TEST_CASE("thinIndices spreads indices across the range", "[util]")
{
	const int64_t points = 101;
	const int64_t limit = 11;
	torch::Tensor indices = thinIndices(points, limit);

	REQUIRE(indices.defined());
	REQUIRE(indices.numel() == limit);
	REQUIRE(indices.scalar_type() == torch::kLong);

	auto accessor = indices.accessor<int64_t, 1>();
	REQUIRE(accessor[0] == 0);
	REQUIRE(accessor[limit-1] == points-1);

	// strictly increasing, so every index is unique and in range
	for(int64_t k = 1; k < limit; ++k)
	{
		REQUIRE(accessor[k] > accessor[k-1]);
		REQUIRE(accessor[k] < points);
	}
}

TEST_CASE("thinIndices keeps the endpoints for a minimal limit", "[util]")
{
	torch::Tensor indices = thinIndices(50, 3);
	REQUIRE(indices.defined());
	REQUIRE(indices.numel() == 3);
	auto accessor = indices.accessor<int64_t, 1>();
	REQUIRE(accessor[0] == 0);
	REQUIRE(accessor[2] == 49);
}

TEST_CASE("formatValue round trips a double", "[util]")
{
	SECTION("plain values")
	{
		REQUIRE(formatValue(123.456) == "123.456");
		REQUIRE(formatValue(1.0) == "1");
		REQUIRE(formatValue(-7.5) == "-7.5");
	}
	SECTION("parses back to the same value")
	{
		for(double value: {1e-6, 3.14159265358979, 123456.789, 1e12})
		{
			std::string text = formatValue(value);
			REQUIRE(std::atof(text.c_str()) == Approx(value).epsilon(1e-9));
		}
	}
}
