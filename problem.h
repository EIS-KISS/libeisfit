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

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <vector>

#include <eisgenerator/model.h>

#include <torch/torch.h>
#include <torch/csrc/autograd/autograd.h>

#include "modelscript.h"
#include "util.h"

namespace eisfit_internal
{

struct Problem
{
	eis::Model model;
	std::string modelString;
	std::shared_ptr<torch::CompilationUnit> script;
	std::string functionName;
	torch::Tensor omegas;
	torch::Tensor target;
	torch::Tensor scale;
	torch::Tensor logLower;
	torch::Tensor logUpper;
	int64_t parameterCount = 0;
	int64_t evaluations = 0;
	bool approximateJacobian = false;
	double jacobianStep = 1e-5;

	explicit Problem(const std::string& from): model(from), modelString(from)
	{
		script = compileModel(model);
		functionName = model.getCompiledFunctionName();
	}

	torch::Tensor simulate(const torch::Tensor& parameters)
	{
		++evaluations;
		return runScriptModel(model, script, parameters, omegas);
	}

	void warmUp(const std::vector<double>& parameters)
	{
		simulate(doubles(parameters));
	}

	torch::Tensor clampToBox(const torch::Tensor& logParameters) const
	{
		return torch::clamp(logParameters, logLower, logUpper);
	}

	torch::Tensor residualOfPhysical(const torch::Tensor& parameters)
	{
		torch::Tensor difference = simulate(parameters) - target;
		torch::Tensor stacked = torch::stack({torch::real(difference), torch::imag(difference)}, 0);
		return (scale * stacked).reshape({stacked.numel()});
	}

	torch::Tensor residual(const torch::Tensor& logParameters)
	{
		return residualOfPhysical(torch::pow(10.0, clampToBox(logParameters)));
	}

	double cost(const torch::Tensor& logParameters)
	{
		torch::Tensor r = residual(logParameters);
		return torch::sum(r*r).item<double>();
	}

	torch::Tensor jacobianExact(const torch::Tensor& logParameters)
	{
		torch::Tensor physical = torch::pow(10.0, clampToBox(logParameters));
		torch::Tensor leaf = physical.clone().detach().requires_grad_();
		torch::Tensor r = residualOfPhysical(leaf);

		std::vector<torch::Tensor> rows;
		rows.reserve(r.numel());
		for(int64_t k = 0; k < r.numel(); ++k)
		{
			torch::Tensor seed = torch::zeros_like(r);
			seed[k] = 1.0;
			torch::autograd::variable_list gradient = torch::autograd::grad({r}, {leaf}, {seed},
					true, false, true);
			rows.push_back(gradient.at(0).defined() ? gradient.at(0) : torch::zeros_like(leaf));
		}
		return torch::stack(rows, 0) * (physical * std::log(10.0)).reshape({1, -1});
	}

	torch::Tensor jacobianApprox(const torch::Tensor& logParameters)
	{
		torch::Tensor base = residual(logParameters); // {2N}
		std::vector<torch::Tensor> columns;
		columns.reserve(parameterCount);
		for(int64_t i = 0; i < parameterCount; ++i)
		{
			torch::Tensor bumped = logParameters.clone();
			bumped[i] = logParameters[i] + jacobianStep;
			columns.push_back((residual(bumped) - base)/jacobianStep); // {2N}
		}
		return torch::stack(columns, 1); // {2N,n}
	}

	torch::Tensor jacobian(const torch::Tensor& logParameters)
	{
		return approximateJacobian ? jacobianApprox(logParameters) : jacobianExact(logParameters);
	}
};

struct StartOutcome
{
	bool valid = false;
	bool converged = false;
	double value = std::numeric_limits<double>::quiet_NaN();
	torch::Tensor logParameters;
	int64_t evaluations = 0;
};

StartOutcome runStart(Problem& problem, const torch::Tensor& startLogParameters, double tolerance,
		int64_t budget);

}
