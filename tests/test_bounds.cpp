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

#include <eisgenerator/model.h>

#include "bounds.h"

using namespace eisfit_internal;

namespace
{

struct Box
{
	std::vector<double> lower;
	std::vector<double> upper;
	std::vector<double> start;
	int64_t count = 0;
};

Box box(const std::string& modelString, double widen = 0.0)
{
	eis::Model model(modelString);
	Box out;
	deriveBox(model, modelString, widen, out.lower, out.upper, out.start, out.count);
	return out;
}

}

TEST_CASE("deriveBox counts parameters like the model", "[bounds]")
{
	for(const std::string& modelString: {"r", "r-p", "r-r", "r-w", "lr-p(r-w)"})
	{
		eis::Model model(modelString);
		Box out;
		deriveBox(model, modelString, 0, out.lower, out.upper, out.start, out.count);
		REQUIRE(out.count == static_cast<int64_t>(model.getParameterCount()));
		REQUIRE(out.lower.size() == static_cast<size_t>(out.count));
		REQUIRE(out.upper.size() == static_cast<size_t>(out.count));
		REQUIRE(out.start.size() == static_cast<size_t>(out.count));
	}
}

TEST_CASE("deriveBox brackets a fixed parameter around its value", "[bounds]")
{
	Box out = box("r{100}");
	REQUIRE(out.count == 1);
	REQUIRE(out.start[0] == Approx(100.0).epsilon(1e-6));
	REQUIRE(out.lower[0] < 100.0);
	REQUIRE(out.upper[0] > 100.0);
}

TEST_CASE("deriveBox honours a stated range", "[bounds]")
{
	Box out = box("r{1~1000}");
	REQUIRE(out.count == 1);
	REQUIRE(out.lower[0] == Approx(1.0));
	REQUIRE(out.upper[0] == Approx(1000.0));
	// logarithmic element: the start is the geometric mean of the range
	REQUIRE(out.start[0] == Approx(std::sqrt(1000.0)).epsilon(1e-6));
}

TEST_CASE("deriveBox does not widen a stated range", "[bounds]")
{
	Box tight = box("r{1~1000}", 0.0);
	Box wide = box("r{1~1000}", 6.0);
	REQUIRE(wide.lower[0] == Approx(tight.lower[0]));
	REQUIRE(wide.upper[0] == Approx(tight.upper[0]));
}

TEST_CASE("deriveBox widens a defaulted range", "[bounds]")
{
	Box tight = box("r", 0.0);
	Box wide = box("r", 6.0);
	REQUIRE(wide.lower[0] < tight.lower[0]);
	REQUIRE(wide.upper[0] > tight.upper[0]);
}

TEST_CASE("deriveBox keeps every parameter inside a sane box", "[bounds]")
{
	for(const std::string& modelString: {"r", "r-p", "r-r", "r-w", "lr-p(r-w)", "r{50}-p{1e-4,0.85}"})
	{
		Box out = box(modelString);
		for(int64_t i = 0; i < out.count; ++i)
		{
			INFO("model " << modelString << " parameter " << i);
			REQUIRE(out.lower[i] > 0.0);
			REQUIRE(out.lower[i] < out.upper[i]);
			// the start is clamped strictly inside the box, never on a face
			REQUIRE(out.lower[i] < out.start[i]);
			REQUIRE(out.start[i] < out.upper[i]);
		}
	}
}

TEST_CASE("deriveBox widens monotonically with the widening parameter", "[bounds]")
{
	Box none = box("r", 0.0);
	Box some = box("r", 2.0);
	Box more = box("r", 6.0);
	REQUIRE(none.lower[0] > some.lower[0]);
	REQUIRE(some.lower[0] > more.lower[0]);
	REQUIRE(none.upper[0] < some.upper[0]);
	REQUIRE(some.upper[0] < more.upper[0]);
}
