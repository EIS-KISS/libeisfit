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
#include <memory>
#include <string>
#include <vector>

#include <eisgenerator/model.h>

#include <torch/torch.h>
#include <torch/csrc/autograd/autograd.h>

#include "modelscript.h"

namespace eisfit_internal
{

struct BatchProblem
{
	eis::Model model;
	std::string modelString;
	std::shared_ptr<torch::CompilationUnit> script;
	torch::Tensor omegas;   // {N,1}
	torch::Tensor target;   // {N,B} complex
	torch::Tensor scale;    // {N,B}
	torch::Tensor logLower; // {n,1}, broadcast against the columns of {n,B}
	torch::Tensor logUpper; // {n,1}
	int64_t parameterCount = 0;
	int64_t batch = 0;
	int64_t evaluations = 0;
	bool approximateJacobian = false;
	double jacobianStep = 1e-5;

	explicit BatchProblem(const std::string& from): model(from), modelString(from)
	{
		script = compileModel(model);
	}

	torch::Tensor simulate(const torch::Tensor& parameters)
	{
		++evaluations;
		return runScriptModel(model, script, parameters, omegas)
			.reshape({omegas.numel(), parameters.size(1)});
	}

	void warmUp(const torch::Tensor& parameters)
	{
		simulate(parameters);
	}

	torch::Tensor clampToBox(const torch::Tensor& logParameters) const
	{
		return torch::clamp(logParameters, logLower, logUpper);
	}

	torch::Tensor residualOfPhysical(const torch::Tensor& parameters)
	{
		torch::Tensor difference = simulate(parameters) - target;
		torch::Tensor stacked = torch::stack({torch::real(difference), torch::imag(difference)}, 0);
		return (scale.unsqueeze(0)*stacked).reshape({stacked.size(0)*stacked.size(1), batch});
	}

	torch::Tensor residual(const torch::Tensor& logParameters)
	{
		return residualOfPhysical(torch::pow(10.0, clampToBox(logParameters)));
	}

	torch::Tensor cost(const torch::Tensor& logParameters)
	{
		torch::Tensor r = residual(logParameters);
		return torch::sum(r*r, 0);
	}

	// {2N,n,B}: exact jacobian in log10 space, one autograd backward pass per residual row.
	torch::Tensor jacobianExact(const torch::Tensor& logParameters)
	{
		torch::Tensor physical = torch::pow(10.0, clampToBox(logParameters));
		torch::Tensor leaf = physical.clone().detach().requires_grad_();
		torch::Tensor r = residualOfPhysical(leaf);

		std::vector<torch::Tensor> rows;
		rows.reserve(r.size(0));
		for(int64_t k = 0; k < r.size(0); ++k)
		{
			torch::Tensor seed = torch::zeros_like(r);
			seed[k] = 1.0;
			torch::autograd::variable_list gradient = torch::autograd::grad({r}, {leaf}, {seed},
					true, false, true);
			rows.push_back(gradient.at(0).defined() ? gradient.at(0) : torch::zeros_like(leaf));
		}
		return torch::stack(rows, 0)*(physical*std::log(10.0)).unsqueeze(0);
	}

	// {2N,n,B}: forward-difference jacobian in log10 space. One extra batched simulation per
	// parameter serves the whole batch, against the exact jacobian's one backward pass per
	// residual row per batch column.
	torch::Tensor jacobianApprox(const torch::Tensor& logParameters)
	{
		torch::Tensor base = residual(logParameters); // {2N,B}
		std::vector<torch::Tensor> parts;
		parts.reserve(parameterCount);
		for(int64_t i = 0; i < parameterCount; ++i)
		{
			torch::Tensor bumped = logParameters.clone();
			bumped[i] = logParameters[i] + jacobianStep;
			parts.push_back((residual(bumped) - base)/jacobianStep); // {2N,B}
		}
		return torch::stack(parts, 1); // {2N,n,B}
	}

	torch::Tensor jacobian(const torch::Tensor& logParameters)
	{
		return approximateJacobian ? jacobianApprox(logParameters) : jacobianExact(logParameters);
	}
};

struct BatchOutcome
{
	torch::Tensor logParameters; // {n,B}
	torch::Tensor values;        // {B}
	torch::Tensor valid;         // {B}
	torch::Tensor converged;     // {B}
	int64_t evaluations = 0;
};

BatchOutcome runBatchStart(BatchProblem& problem, const torch::Tensor& startLogParameters,
		double tolerance, int64_t budget);

}
