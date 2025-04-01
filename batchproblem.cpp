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

#include "batchproblem.h"

#include <cmath>
#include <exception>
#include <limits>

namespace eisfit_internal
{

BatchOutcome runBatchStart(BatchProblem& problem, const torch::Tensor& startLogParameters,
		double tolerance, int64_t budget)
{
	BatchOutcome outcome;
	torch::Tensor x = problem.clampToBox(startLogParameters).detach().clone();

	torch::Tensor cost = problem.cost(x);
	torch::Tensor valid = torch::isfinite(cost);

	torch::Tensor lambda = torch::full({problem.batch}, 1e-2, torch::kFloat64);
	torch::Tensor converged = torch::zeros({problem.batch}, torch::kBool);
	const int64_t usedByStart = problem.evaluations;

	while(problem.evaluations - usedByStart < budget)
	{
		torch::Tensor active = valid.logical_and(converged.logical_not());
		if(not torch::any(active).item<bool>())
			break;

		torch::Tensor j = problem.jacobian(x);
		torch::Tensor r = problem.residual(x);
		j = torch::where(valid.reshape({1, 1, problem.batch}), j, torch::zeros_like(j));
		r = torch::where(valid.reshape({1, problem.batch}), r, torch::zeros_like(r));

		torch::Tensor jt = j.permute({2, 1, 0});
		torch::Tensor normal = torch::bmm(jt, j.permute({2, 0, 1}));
		torch::Tensor gradient = torch::bmm(jt, r.t().unsqueeze(-1));
		torch::Tensor curvature = torch::diagonal(normal, 0, -2, -1);
		torch::Tensor scaleOfCurvature = std::get<0>(torch::max(torch::abs(curvature), 1));
		scaleOfCurvature = torch::where(torch::isfinite(scaleOfCurvature)
				& (scaleOfCurvature > 0), scaleOfCurvature, torch::ones_like(scaleOfCurvature));

		torch::Tensor improved = torch::zeros_like(active);
		int attempts = 0;
		for(; attempts < 30; ++attempts)
		{
			torch::Tensor pending = active.logical_and(improved.logical_not());
			if(not torch::any(pending).item<bool>())
				break;

			torch::Tensor identity = torch::eye(problem.parameterCount, normal.options());
			torch::Tensor damped = normal
				+ (lambda.reshape({-1, 1, 1})*curvature.unsqueeze(-1)
					+ 1e-12*scaleOfCurvature.reshape({-1, 1, 1}))*identity;

			torch::Tensor step;
			try
			{
				step = torch::linalg_solve(damped, gradient);
			}
			catch(const std::exception&)
			{
				step = torch::zeros({problem.batch, problem.parameterCount, 1}, torch::kFloat64);
				for(int64_t b = 0; b < problem.batch; ++b)
				{
					if(not pending[b].item<bool>())
						continue;
					try
					{
						step[b] = torch::linalg_solve(damped[b], gradient[b]);
					}
					catch(const std::exception&) {}
				}
			}

			torch::Tensor stepSum = torch::sum(step*step, {1, 2}).reshape({problem.batch});
			torch::Tensor candidate = problem.clampToBox(
					x - step.squeeze(-1).permute({1, 0}));
			torch::Tensor candidateCost = problem.cost(candidate);
			torch::Tensor accept = pending.logical_and(torch::isfinite(candidateCost))
				.logical_and(torch::isfinite(stepSum))
				.logical_and(candidateCost < cost*(1.0-tolerance));

			x = torch::where(accept.unsqueeze(0), candidate, x);
			cost = torch::where(accept, candidateCost, cost);
			lambda = torch::where(pending, torch::where(accept,
					torch::clamp(lambda*0.3, 1e-14, std::numeric_limits<double>::infinity()),
					lambda*10.0), lambda);
			improved = improved.logical_or(accept);
		}

		converged = converged.logical_or(active.logical_and(improved.logical_not()));
	}

	outcome.logParameters = x;
	outcome.values = cost;
	outcome.valid = valid;
	outcome.converged = converged;
	outcome.evaluations = problem.evaluations - usedByStart;
	return outcome;
}

}
