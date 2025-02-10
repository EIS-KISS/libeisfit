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

#include "eisfit/fit.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <ATen/CPUGeneratorImpl.h>

#include <eisgenerator/componant/componant.h>
#include <eisgenerator/model.h>

#include <torch/jit.h>
#include <torch/torch.h>
#include <torch/csrc/autograd/autograd.h>
#include <torch/csrc/jit/api/function_impl.h>

namespace
{

// Compile the TorchScript the generator emits for a model and evaluate it. These mirror the
// helpers of the same name in TorchKissAnn's utils/modelscript.cpp, which the fit was taken
// from; the fit is the only consumer here.
std::shared_ptr<torch::CompilationUnit> compileModel(eis::Model &model)
{
	std::string torchScript = model.getTorchScript();
	std::shared_ptr<torch::CompilationUnit> compiledModule = torch::jit::compile(torchScript);
	return compiledModule;
}

torch::Tensor runScriptModel(eis::Model& model, std::shared_ptr<torch::CompilationUnit> compiledScript,
							torch::Tensor parameters, torch::Tensor omegas)
{
	// the generated script mixes the frequencies with the parameters in one expression, so both
	// have to live on the same device
	if(omegas.device() != parameters.device())
		omegas = omegas.to(parameters.device());

	torch::Tensor result = compiledScript->run_method(model.getCompiledFunctionName(), parameters, omegas).toTensor();
	assert(result.scalar_type() == torch::kComplexFloat || result.scalar_type() == torch::kComplexDouble);
	assert(result.size(0) == omegas.numel());
	// the script broadcasts the parameter rows against the omega column vector, so a
	// batch of parameter vectors (one column per sample) yields one column per sample
	const int64_t expectedColumns = parameters.dim() > 1 ? parameters.size(1) : 1;
	assert(result.sizes().size() == 1 || result.size(1) == expectedColumns);
	return result.reshape({result.numel()});
}

constexpr double FIXED_WIDENING_DECADES = 3;
constexpr double SWEEP_WIDENING_DECADES = 6;
constexpr double TINY_PARAMETER = 1e-300;
constexpr double BOUND_SLACK = 1e-3;
constexpr double NOISE_AGREEMENT = 2.0;
constexpr double POOR_FIT_FACTOR = 5;
constexpr double CORRELATED_RESIDUAL = 0.8;

torch::Tensor doubles(const std::vector<double>& values)
{
	return torch::tensor(values).to(torch::kFloat64);
}

torch::Tensor thinIndices(int64_t points, int64_t limit)
{
	if(limit <= 0 or limit >= points or limit < 3)
		return torch::Tensor();
	std::vector<int64_t> idx;
	idx.reserve(limit);
	for(int64_t k = 0; k < limit; ++k)
		idx.push_back(k*(points-1)/(limit-1));
	idx.back() = points-1;
	return torch::tensor(idx);
}

std::string formatValue(double value)
{
	std::ostringstream stream;
	stream<<std::setprecision(10)<<value;
	return stream.str();
}

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

std::vector<bool> explicitParameters(const std::string& modelString,
		const std::vector<eis::Componant*>& componants)
{
	std::vector<bool> given;
	size_t element = 0;
	for(size_t i = 0; i < modelString.size() and element < componants.size(); ++i)
	{
		if(not std::isalpha(static_cast<unsigned char>(modelString[i])))
			continue;
		bool brace = i+1 < modelString.size() and modelString[i+1] == '{';
		for(size_t k = 0; k < componants[element]->paramCount(); ++k)
			given.push_back(brace);
		++element;
		if(brace)
			while(i < modelString.size() and modelString[i] != '}')
				++i;
	}
	return given;
}


void deriveBox(eis::Model& model, const std::string& modelString, double widenDecades,
		std::vector<double>& lower, std::vector<double>& upper, std::vector<double>& start,
		int64_t& parameterCount)
{
	std::vector<eis::Range> ranges = model.getFlatParameterRanges();
	std::vector<eis::Range> defaults = model.getDefaultParameters();
	std::vector<bool> given = explicitParameters(modelString, model.getFlatComponants());
	parameterCount = static_cast<int64_t>(ranges.size());

	for(size_t i = 0; i < ranges.size(); ++i)
	{
		const eis::Range& range = ranges[i];
		const eis::Range& fallback = i < defaults.size() ? defaults[i] : range;
		bool stated = i < given.size() and given[i];
		bool fixed = range.count <= 1 or range.start == range.end;
		bool logarithmic = fallback.log;

		double low = 0;
		double high = 0;
		double middle = 0;

		if(fixed)
		{
			double value = static_cast<double>(range.start);
			if(value <= 0)
				value = static_cast<double>(fallback.start);
			if(logarithmic)
			{
				double factor = std::pow(10.0, FIXED_WIDENING_DECADES);
				low = value/factor;
				high = value*factor;
			}
			else
			{
				double span = std::max(std::abs(value)*0.5, 1e-12);
				low = std::max(value-span, TINY_PARAMETER);
				high = value+span;
			}
			middle = value;
		}
		else if(stated)
		{
			low = static_cast<double>(range.start);
			high = static_cast<double>(range.end);
			middle = logarithmic ? std::sqrt(low*high) : 0.5*(low+high);
		}
		else
		{
			low = static_cast<double>(range.start);
			high = static_cast<double>(range.end);
			middle = logarithmic ? std::sqrt(low*high) : 0.5*(low+high);
			if(logarithmic)
			{
				double factor = std::pow(10.0, widenDecades);
				low = std::max(low/factor, TINY_PARAMETER);
				high = high*factor;
			}
			else
			{
				double slack = 0.5*(high-low)*widenDecades/SWEEP_WIDENING_DECADES;
				low = std::max(low-slack, TINY_PARAMETER);
				high = high+slack;
			}
		}

		low = std::max(low, TINY_PARAMETER);
		if(high <= low)
			high = low*10.0;

		double logLow = std::log10(low);
		double logHigh = std::log10(high);
		double logSpan = logHigh-logLow;
		double logMiddle = middle > 0 ? std::log10(middle) : 0.5*(logLow+logHigh);
		logMiddle = std::clamp(logMiddle, logLow+1e-3*logSpan, logHigh-1e-3*logSpan);
		middle = std::pow(10.0, logMiddle);

		lower.push_back(low);
		upper.push_back(high);
		start.push_back(middle);
	}
}


double estimateNoiseFloor(const torch::Tensor& target, const torch::Tensor& magnitude)
{
	if(target.numel() < 9)
		return 0;

	auto spread = [&target](const std::vector<double>& coefficients, double norm) {
		torch::Tensor acc;
		for(size_t shift = 0; shift < coefficients.size(); ++shift)
		{
			torch::Tensor slice = target.slice(0, shift, target.numel()-coefficients.size()+1+shift);
			torch::Tensor term = coefficients[shift]*slice;
			acc = acc.defined() ? acc + term : term;
		}
		return std::sqrt(torch::mean(torch::abs(acc)*torch::abs(acc)).item<double>())/norm;
	};

	double fourth = spread({1, -4, 6, -4, 1}, std::sqrt(70.0));
	double sixth = spread({1, -6, 15, -20, 15, -6, 1}, std::sqrt(924.0));

	if(not std::isfinite(fourth) or not std::isfinite(sixth))
		return 0;
	if(fourth > NOISE_AGREEMENT*sixth or sixth > NOISE_AGREEMENT*fourth)
		return 0;

	double mean = magnitude.mean().item<double>();
	if(not std::isfinite(mean) or mean <= 0)
		return 0;
	return 0.5*(fourth+sixth)/mean;
}

double lagOneAutocorrelation(const torch::Tensor& sequence)
{
	if(sequence.numel() < 3)
		return 0;
	torch::Tensor centred = sequence - sequence.mean();
	double energy = torch::sum(centred*centred).item<double>();
	if(not std::isfinite(energy) or energy <= 0)
		return 0;
	torch::Tensor first = centred.slice(0, 0, centred.numel()-1);
	torch::Tensor second = centred.slice(0, 1, centred.numel());
	return torch::sum(first*second).item<double>()/energy;
}

void applyVerdict(FitResult& result, double fitLimit)
{
	result.goodFitLimit = std::max(fitLimit, 3*result.noiseFloor);
	result.poorFitLimit = std::max(POOR_FIT_FACTOR*fitLimit, 10*result.noiseFloor);

	if(result.rmsRelativeError > result.poorFitLimit)
		result.verdict = FitVerdict::FIT_VERDICT_UNFITTED;
	else if(result.rmsRelativeError > result.goodFitLimit)
		result.verdict = FitVerdict::FIT_VERDICT_MARGINAL;
	else if(result.residualCorrelation > CORRELATED_RESIDUAL)
		result.verdict = FitVerdict::FIT_VERDICT_MARGINAL;
	else
		result.verdict = FitVerdict::FIT_VERDICT_FIT;
}

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

FitResult eisFit(torch::Tensor spectra, torch::Tensor omegas, const std::string& modelString,
		const FitOptions& options)
{
	FitResult result;

	Problem problem(modelString);
	problem.approximateJacobian = options.approximateJacobian;
	problem.jacobianStep = options.jacobianStep;

	problem.omegas = omegas.to(torch::kFloat64).reshape({omegas.numel(), 1});
	problem.target = spectra.to(torch::kComplexDouble).reshape({spectra.numel()});
	if(problem.target.numel() != problem.omegas.numel() or problem.target.numel() < 2)
		return result;

	std::vector<double> lower;
	std::vector<double> upper;
	std::vector<double> start;
	deriveBox(problem.model, problem.modelString, 0, lower, upper, start, problem.parameterCount);

	torch::Tensor fullOmgas = problem.omegas;
	torch::Tensor fullTarget = problem.target;
	torch::Tensor thinned = thinIndices(problem.target.numel(), options.searchPoints);
	if(thinned.defined())
	{
		problem.omegas = problem.omegas.index_select(0, thinned);
		problem.target = problem.target.index_select(0, thinned);
	}
	torch::Tensor magnitude = torch::abs(problem.target);
	torch::Tensor floor = torch::mean(magnitude)*options.weightFloor;
	torch::Tensor weight = 1.0/torch::sqrt(magnitude*magnitude + floor*floor);
	problem.scale = torch::sqrt(weight/problem.target.numel());
	// lands in a real basin
	std::vector<double> narrowLower = lower;
	std::vector<double> narrowUpper = upper;

	if(options.startingParams.defined() and options.startingParams.numel() == problem.parameterCount)
	{
		torch::Tensor given = options.startingParams.to(torch::kFloat64).reshape({problem.parameterCount});
		auto accessor = given.accessor<double, 1>();
		for(int64_t i = 0; i < problem.parameterCount; ++i)
			if(std::isfinite(accessor[i]) and accessor[i] > 0)
				start[i] = std::clamp(accessor[i], lower[i]*10.0, upper[i]*0.1);
	}

	problem.warmUp(start);

	result.parameterNames = problem.model.getParameterNames();
	result.uncertainties = torch::full({problem.parameterCount}, std::numeric_limits<double>::quiet_NaN(),
			torch::kFloat64);

	std::mt19937_64 generator(options.seed);
	std::uniform_real_distribution<double> uniform(0.02, 0.98);
	std::uniform_int_distribution<int64_t> pick(0, std::max<int64_t>(0, problem.parameterCount-1));

	double bestValue = std::numeric_limits<double>::infinity();
	StartOutcome best;
	std::vector<double> bestStart = start;

	for(int stage = 0; stage < 2; ++stage)
	{
		std::vector<double> stageLower;
		std::vector<double> stageUpper;
		std::vector<double> stageStart;
		deriveBox(problem.model, problem.modelString, stage == 0 ? 0 : SWEEP_WIDENING_DECADES,
				stageLower, stageUpper, stageStart, problem.parameterCount);

		if(stage == 1)
		{
			torch::Tensor settled = torch::pow(10.0, problem.clampToBox(best.logParameters));
			auto accessor = settled.accessor<double, 1>();
			for(int64_t i = 0; i < problem.parameterCount; ++i)
				stageStart[i] = std::clamp(accessor[i], stageLower[i], stageUpper[i]);
			lower = stageLower;
			upper = stageUpper;
		}

		problem.logLower = doubles(stageLower).log10();
		problem.logUpper = doubles(stageUpper).log10();

		std::vector<double> home;
		for(double value: stageStart)
			home.push_back(std::log10(value));

		for(int attempt = 0; attempt <= options.restarts; ++attempt)
		{
			std::vector<double> candidate;
			if(attempt == 0)
				candidate = home;
			else if(best.valid and attempt % 2 == 1)
			{
				candidate.clear();
				int64_t which = pick(generator);
				for(int64_t i = 0; i < problem.parameterCount; ++i)
				{
					if(i == which)
						candidate.push_back(std::log10(stageLower[i])
							+ uniform(generator)*(std::log10(stageUpper[i])-std::log10(stageLower[i])));
					else
						candidate.push_back(best.logParameters[i].item<double>());
				}
			}
			else
			{
				candidate = home;
				for(int64_t i = 0; i < problem.parameterCount; ++i)
					candidate[i] = std::log10(narrowLower[i])
						+ uniform(generator)*(std::log10(narrowUpper[i])-std::log10(narrowLower[i]));
			}

			StartOutcome outcome = runStart(problem, doubles(candidate), options.tolerance,
					std::max<int64_t>(1, options.maxEvaluations));
			if(not outcome.valid)
				continue;

			++result.starts;
			if(outcome.value < bestValue)
			{
				bestValue = outcome.value;
				best = outcome;
				bestStart = stageStart;
				result.bestStart = result.starts;
			}
		}
	}

	if(not best.valid)
		return result;

	torch::NoGradGuard noGrad;
	if(thinned.defined())
	{
		problem.omegas = fullOmgas;
		problem.target = fullTarget;
		magnitude = torch::abs(problem.target);
		floor = torch::mean(magnitude)*options.weightFloor;
		weight = 1.0/torch::sqrt(magnitude*magnitude + floor*floor);
		problem.scale = torch::sqrt(weight/problem.target.numel());
	}
	result.parameters = torch::pow(10.0, problem.clampToBox(best.logParameters)).detach();
	result.success = true;
	result.converged = best.converged;
	result.startParameters = doubles(bestStart);
	result.lowerBounds = doubles(lower);
	result.upperBounds = doubles(upper);
	result.evaluations = best.evaluations;
	result.chiSquared = problem.cost(best.logParameters);

	const int64_t points = problem.target.numel();
	const int64_t dof = std::max<int64_t>(1, 2*points - problem.parameterCount);
	result.reducedChiSquared = best.value*2.0*points/dof;

	result.fittedSpectra = problem.simulate(result.parameters);
	result.residuals = problem.target - result.fittedSpectra;
	torch::Tensor error = torch::abs(result.residuals)/magnitude;
	result.rmsRelativeError = torch::mean(error*error).item<double>();
	result.rmsRelativeError = std::sqrt(result.rmsRelativeError);
	result.maxRelativeError = error.max().item<double>();
	result.noiseFloor = estimateNoiseFloor(problem.target, magnitude);
	result.residualCorrelation = lagOneAutocorrelation(error);

	for(int64_t i = 0; i < problem.parameterCount; ++i)
	{
		double logValue = best.logParameters[i].item<double>();
		double logSpan = problem.logUpper[i].item<double>()-problem.logLower[i].item<double>();
		result.atBound.push_back(logValue-problem.logLower[i].item<double>() <= BOUND_SLACK*logSpan
				or problem.logUpper[i].item<double>()-logValue <= BOUND_SLACK*logSpan);
	}

	try
	{
		torch::AutoGradMode enableGrad(true);
		torch::Tensor j = problem.jacobian(best.logParameters);
		torch::Tensor singular = torch::linalg_svdvals(j);
		double largest = singular[0].item<double>();
		double smallest = singular[singular.numel()-1].item<double>();
		result.conditionNumber = smallest > 0 ? largest/smallest : std::numeric_limits<double>::infinity();

		torch::Tensor r = problem.residual(best.logParameters);
		double variance = torch::sum(r*r).item<double>()/dof;
		torch::Tensor diagonal = torch::diagonal(torch::linalg_inv(torch::t(j).matmul(j))*variance);
		for(int64_t i = 0; i < problem.parameterCount; ++i)
		{
			double entry = diagonal[i].item<double>();
			result.uncertainties[i] = std::isfinite(entry) and entry > 0 ? std::sqrt(entry)
				: std::numeric_limits<double>::quiet_NaN();
		}
		// the Jacobian was taken in log10 space, so the sigma is a fraction of the parameter
		result.uncertainties = result.uncertainties*result.parameters*std::log(10.0);
	}
	catch(const std::exception&)
	{
	}

	applyVerdict(result, options.fitLimit);

	return result;
}

std::vector<FitResult> eisFitBatch(const torch::Tensor& spectra, const torch::Tensor& omegas,
		const std::string& modelString, const FitOptions& options)
{
	std::vector<FitResult> results;

	torch::Tensor rows = spectra.to(torch::kComplexDouble);
	if(rows.dim() == 1)
		rows = rows.reshape({1, -1});
	if(rows.dim() != 2 or rows.size(1) < 2 or omegas.numel() != rows.size(1))
		return results;

	const int64_t batch = rows.size(0);
	const int64_t points = rows.size(1);
	results.resize(batch);
	if(batch == 0)
		return results;

	BatchProblem problem(modelString);
	problem.batch = batch;
	problem.approximateJacobian = options.approximateJacobian;
	problem.jacobianStep = options.jacobianStep;
	problem.omegas = omegas.to(torch::kFloat64).reshape({points, 1});
	problem.target = rows.t().contiguous();

	std::vector<double> lower;
	std::vector<double> upper;
	std::vector<double> start;
	deriveBox(problem.model, problem.modelString, 0, lower, upper, start, problem.parameterCount);

	torch::Tensor fullOmgas = problem.omegas;
	torch::Tensor fullTarget = problem.target;
	torch::Tensor thinned = thinIndices(points, options.searchPoints);
	if(thinned.defined())
	{
		problem.omegas = problem.omegas.index_select(0, thinned);
		problem.target = problem.target.index_select(0, thinned);
	}
	torch::Tensor magnitude = torch::abs(problem.target);
	torch::Tensor floor = torch::mean(magnitude, 0, true)*options.weightFloor;
	torch::Tensor weight = 1.0/torch::sqrt(magnitude*magnitude + floor*floor);
	problem.scale = torch::sqrt(weight/problem.target.size(0));
	std::vector<double> narrowLower = lower;
	std::vector<double> narrowUpper = upper;
	const int64_t parameterCount = problem.parameterCount;
	if(parameterCount < 1)
		return results;

	torch::Tensor startAll = doubles(start).reshape({parameterCount, 1})
		.expand({parameterCount, batch}).contiguous();
	if(options.startingParams.defined())
	{
		torch::Tensor given = options.startingParams.to(torch::kFloat64);
		if(given.numel() == parameterCount)
			given = given.reshape({parameterCount, 1}).expand({parameterCount, batch});
		else if(given.numel() == parameterCount*batch)
			given = given.reshape({parameterCount, batch});
		else
			given = torch::Tensor();
		if(given.defined())
		{
			torch::Tensor clamped = torch::clamp(given,
					doubles(lower).reshape({parameterCount, 1})*10.0,
					doubles(upper).reshape({parameterCount, 1})*0.1);
			startAll = torch::where(torch::isfinite(given).logical_and(given > 0), clamped, startAll);
		}
	}

	problem.warmUp(startAll);

	for(FitResult& result: results)
	{
		result.parameterNames = problem.model.getParameterNames();
		result.uncertainties = torch::full({parameterCount},
				std::numeric_limits<double>::quiet_NaN(), torch::kFloat64);
	}

	torch::Generator generator = torch::make_generator<at::CPUGeneratorImpl>(options.seed);

	torch::Tensor bestValue = torch::full({batch},
			std::numeric_limits<double>::infinity(), torch::kFloat64);
	torch::Tensor bestLog = torch::zeros({parameterCount, batch}, torch::kFloat64);
	torch::Tensor bestValid = torch::zeros({batch}, torch::kBool);
	torch::Tensor bestConverged = torch::zeros({batch}, torch::kBool);
	torch::Tensor bestStartParams = startAll.clone();
	torch::Tensor starts = torch::zeros({batch}, torch::TensorOptions().dtype(torch::kInt64));
	torch::Tensor bestStart = torch::zeros({batch}, torch::TensorOptions().dtype(torch::kInt64));
	std::vector<int64_t> evaluations(batch, 0);

	for(int stage = 0; stage < 2; ++stage)
	{
		std::vector<double> stageLower;
		std::vector<double> stageUpper;
		std::vector<double> stageStart;
		deriveBox(problem.model, problem.modelString, stage == 0 ? 0 : SWEEP_WIDENING_DECADES,
				stageLower, stageUpper, stageStart, problem.parameterCount);

		torch::Tensor stageLowerT = doubles(stageLower).reshape({parameterCount, 1});
		torch::Tensor stageUpperT = doubles(stageUpper).reshape({parameterCount, 1});
		torch::Tensor stageStartT = doubles(stageStart).reshape({parameterCount, 1})
			.expand({parameterCount, batch}).contiguous();

		if(stage == 1)
		{
			torch::Tensor settled = torch::pow(10.0, problem.clampToBox(bestLog));
			settled = torch::where(bestValid.unsqueeze(0), settled, stageStartT);
			stageStartT = torch::clamp(settled, stageLowerT, stageUpperT);
			lower = stageLower;
			upper = stageUpper;
		}

		problem.logLower = torch::log10(stageLowerT);
		problem.logUpper = torch::log10(stageUpperT);

		torch::Tensor home = torch::log10(stageStartT);
		torch::Tensor narrowLogLow = torch::log10(doubles(narrowLower).reshape({parameterCount, 1}));
		torch::Tensor narrowLogHigh = torch::log10(doubles(narrowUpper).reshape({parameterCount, 1}));

		for(int attempt = 0; attempt <= options.restarts; ++attempt)
		{
			torch::Tensor candidate;
			if(attempt == 0)
				candidate = home;
			else
			{
				torch::Tensor uniform = 0.02 + 0.96*torch::rand({parameterCount, batch},
						generator, torch::kFloat64);
				torch::Tensor random = narrowLogLow + uniform*(narrowLogHigh-narrowLogLow);
				torch::Tensor hop = bestLog.clone();
				torch::Tensor which = torch::floor(torch::rand({batch}, generator,
						torch::kFloat64)*static_cast<double>(parameterCount)).clamp(
							0, static_cast<double>(parameterCount-1)).to(torch::kLong);
				torch::Tensor logLow = torch::log10(stageLowerT);
				torch::Tensor logHigh = torch::log10(stageUpperT);
				torch::Tensor drawn = logLow + torch::rand({parameterCount, batch}, generator,
						torch::kFloat64)*(logHigh-logLow);
				// one fresh value per column, that of the parameter this column picked
				torch::Tensor value = drawn.gather(0, which.unsqueeze(0)).squeeze(0);
				hop.index_put_({which, torch::arange(batch, torch::TensorOptions().dtype(torch::kLong))},
						value);

				torch::Tensor useHop = (attempt % 2) == 1 ? bestValid
					: torch::zeros({batch}, torch::kBool);
				candidate = torch::where(useHop.unsqueeze(0), hop, random);
			}

			BatchOutcome outcome = runBatchStart(problem, candidate, options.tolerance,
					std::max<int64_t>(1, options.maxEvaluations));

			starts += outcome.valid.to(torch::kInt64);
			torch::Tensor better = outcome.valid.logical_and(outcome.values < bestValue);
			bestValue = torch::where(better, outcome.values, bestValue);
			bestLog = torch::where(better.unsqueeze(0), outcome.logParameters, bestLog);
			bestValid = bestValid.logical_or(outcome.valid);
			bestConverged = torch::where(better, outcome.converged, bestConverged);
			bestStartParams = torch::where(better.unsqueeze(0), stageStartT, bestStartParams);
			bestStart = torch::where(better, starts, bestStart);
			for(int64_t b = 0; b < batch; ++b)
				if(better[b].item<bool>())
					evaluations[b] = outcome.evaluations;
		}
	}

	torch::NoGradGuard noGrad;
	if(thinned.defined())
	{
		problem.omegas = fullOmgas;
		problem.target = fullTarget;
		magnitude = torch::abs(problem.target);
		floor = torch::mean(magnitude, 0, true)*options.weightFloor;
		weight = 1.0/torch::sqrt(magnitude*magnitude + floor*floor);
		problem.scale = torch::sqrt(weight/points);
	}
	torch::Tensor bestPhysical = torch::pow(10.0, problem.clampToBox(bestLog));
	torch::Tensor fittedAll = problem.simulate(bestPhysical);
	torch::Tensor chiAll = problem.cost(bestLog);

	torch::Tensor jacobianAll;
	torch::Tensor residualAll;
	try
	{
		torch::AutoGradMode enableGrad(true);
		jacobianAll = problem.jacobian(bestLog);
		residualAll = problem.residual(bestLog);
	}
	catch(const std::exception&) {}

	const int64_t dof = std::max<int64_t>(1, 2*points - parameterCount);

	for(int64_t b = 0; b < batch; ++b)
	{
		FitResult& result = results[b];
		if(not bestValid[b].item<bool>())
			continue;

		result.success = true;
		result.converged = bestConverged[b].item<bool>();
		result.parameters = bestPhysical.select(1, b).clone();
		result.startParameters = bestStartParams.select(1, b).clone();
		result.lowerBounds = doubles(lower);
		result.upperBounds = doubles(upper);
		result.evaluations = evaluations[b];
		result.chiSquared = chiAll[b].item<double>();
		result.reducedChiSquared = result.chiSquared*2.0*points/dof;
		result.starts = static_cast<int>(starts[b].item<int64_t>());
		result.bestStart = static_cast<int>(bestStart[b].item<int64_t>());

		torch::Tensor targetColumn = problem.target.select(1, b);
		torch::Tensor magnitudeColumn = magnitude.select(1, b);
		result.fittedSpectra = fittedAll.select(1, b).clone();
		result.residuals = targetColumn - result.fittedSpectra;
		torch::Tensor error = torch::abs(result.residuals)/magnitudeColumn;
		result.rmsRelativeError = torch::mean(error*error).item<double>();
		result.rmsRelativeError = std::sqrt(result.rmsRelativeError);
		result.maxRelativeError = error.max().item<double>();
		result.noiseFloor = estimateNoiseFloor(targetColumn.contiguous(), magnitudeColumn.contiguous());
		result.residualCorrelation = lagOneAutocorrelation(error);

		for(int64_t i = 0; i < parameterCount; ++i)
		{
			double logLow = problem.logLower[i].item<double>();
			double logHigh = problem.logUpper[i].item<double>();
			double logValue = bestLog[i][b].item<double>();
			result.atBound.push_back(logValue-logLow <= BOUND_SLACK*(logHigh-logLow)
					or logHigh-logValue <= BOUND_SLACK*(logHigh-logLow));
		}

		if(jacobianAll.defined())
		{
			try
			{
				torch::Tensor j = jacobianAll.select(2, b);
				torch::Tensor singular = torch::linalg_svdvals(j);
				double largest = singular[0].item<double>();
				double smallest = singular[singular.numel()-1].item<double>();
				result.conditionNumber = smallest > 0 ? largest/smallest
					: std::numeric_limits<double>::infinity();

				torch::Tensor r = residualAll.select(1, b);
				double variance = torch::sum(r*r).item<double>()/dof;
				torch::Tensor diagonal = torch::diagonal(torch::linalg_inv(torch::t(j).matmul(j))*variance);
				for(int64_t i = 0; i < parameterCount; ++i)
				{
					double entry = diagonal[i].item<double>();
					result.uncertainties[i] = std::isfinite(entry) and entry > 0 ? std::sqrt(entry)
						: std::numeric_limits<double>::quiet_NaN();
				}
				// the Jacobian was taken in log10 space, so the sigma is a fraction of the parameter
				result.uncertainties = result.uncertainties*result.parameters*std::log(10.0);
			}
			catch(const std::exception&)
			{
			}
		}

		applyVerdict(result, options.fitLimit);
	}

	return results;
}

std::string fittedModelString(const std::string& modelString, const torch::Tensor& parameters)
{
	if(not parameters.defined())
		return "";

	eis::Model model(modelString);
	std::vector<eis::Componant*> componants = model.getFlatComponants();
	torch::Tensor values = parameters.to(torch::kFloat64).reshape({parameters.numel()});
	auto accessor = values.accessor<double, 1>();

	std::string out;
	size_t componantIndex = 0;
	int64_t valueIndex = 0;

	for(size_t i = 0; i < modelString.size(); ++i)
	{
		char symbol = modelString[i];
		if(not std::isalpha(static_cast<unsigned char>(symbol)))
		{
			out += symbol;
			continue;
		}

		if(componantIndex >= componants.size())
			return "";
		size_t count = componants[componantIndex]->paramCount();
		++componantIndex;
		if(valueIndex+static_cast<int64_t>(count) > values.numel())
			return "";

		out += symbol;
		out += '{';
		for(size_t parameter = 0; parameter < count; ++parameter)
		{
			if(parameter > 0)
				out += ", ";
			out += formatValue(accessor[valueIndex++]);
		}
		out += '}';

		if(i+1 < modelString.size() and modelString[i+1] == '{')
			while(i < modelString.size() and modelString[i] != '}')
				++i;
	}

	if(componantIndex != componants.size() or valueIndex != values.numel())
		return "";
	return out;
}
