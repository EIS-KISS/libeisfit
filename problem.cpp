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

#include "problem.h"

#include <cmath>
#include <exception>

namespace eisfit_internal
{

StartOutcome runStart(Problem& problem, const torch::Tensor& startLogParameters, double tolerance,
		int64_t budget)
{
	StartOutcome outcome;
	torch::Tensor x = problem.clampToBox(startLogParameters).detach().clone();

	double cost = problem.cost(x);
	if(not std::isfinite(cost))
		return outcome;

	double lambda = 1e-2;
	const int64_t usedByStart = problem.evaluations;

	while(problem.evaluations - usedByStart < budget)
	{
		torch::Tensor j = problem.jacobian(x);
		torch::Tensor r = problem.residual(x);
		torch::Tensor normal = torch::t(j).matmul(j);
		torch::Tensor gradient = torch::t(j).matmul(r.reshape({-1, 1}));
		torch::Tensor curvature = torch::diagonal(normal);
		double scaleOfCurvature = torch::max(torch::abs(curvature)).item<double>();
		if(not std::isfinite(scaleOfCurvature) or scaleOfCurvature <= 0)
			break;

		bool improved = false;
		for(int attempt = 0; attempt < 30 and not improved; ++attempt)
		{
			torch::Tensor damped = normal + lambda*torch::diag(curvature)
				+ 1e-12*scaleOfCurvature*torch::eye(normal.size(0), normal.options());
			torch::Tensor step;
			try
			{
				step = torch::linalg_solve(damped, gradient).reshape({-1});
			}
			catch(const std::exception&)
			{
				lambda *= 10;
				continue;
			}
			if(not std::isfinite(torch::sum(step*step).item<double>()))
			{
				lambda *= 10;
				continue;
			}

			torch::Tensor candidate = problem.clampToBox(x - step);
			double candidateCost = problem.cost(candidate);
			if(std::isfinite(candidateCost) and candidateCost < cost*(1.0-tolerance))
			{
				cost = candidateCost;
				x = candidate;
				lambda = std::max(lambda*0.3, 1e-14);
				improved = true;
			}
			else
				lambda *= 10;
		}

		if(not improved)
		{
			outcome.converged = true;
			break;
		}
	}

	outcome.valid = std::isfinite(cost);
	outcome.value = cost;
	outcome.logParameters = x;
	outcome.evaluations = problem.evaluations - usedByStart;
	return outcome;
}

}
