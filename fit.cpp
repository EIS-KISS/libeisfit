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
#include <cctype>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <ATen/CPUGeneratorImpl.h>

#include <eisgenerator/componant/componant.h>
#include <eisgenerator/model.h>

#include <torch/torch.h>

#include "batchproblem.h"
#include "bounds.h"
#include "constants.h"
#include "diagnostics.h"
#include "problem.h"
#include "util.h"

using namespace eisfit_internal;

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
