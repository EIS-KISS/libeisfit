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
#include <string>
#include <vector>

#include <torch/types.h>

struct FitOptions
{
	torch::Tensor startingParams;
	int64_t maxEvaluations = 2000;
	int restarts = 16;
	double tolerance = 1e-10;
	double weightFloor = 1e-3;
	double fitLimit = 0.01;
	uint64_t seed = 0x5655;
	// Spectra is decimated to this nummber of poiints to avoid jacobian costs one backward pass explosion if > 0
	int64_t searchPoints = 0;
};


typedef enum
{
	FIT_VERDICT_UNFITTED,
	FIT_VERDICT_MARGINAL,
	FIT_VERDICT_FIT
} FitVerdict;

struct FitResult
{
	bool success = false;
	bool converged = false;
	torch::Tensor parameters;
	torch::Tensor startParameters;
	torch::Tensor uncertainties; // One standard deviation per parameter, NaN where the fit does not determine it. */
	torch::Tensor lowerBounds;
	torch::Tensor upperBounds;
	std::vector<bool> atBound;
	std::vector<std::string> parameterNames;
	torch::Tensor fittedSpectra;
	torch::Tensor residuals;
	// The weighted sum of squared deviations between the measured and the simulated spectrum that was
	// minimised, normalised by the number of points. Each point is weighted by 1/sqrt(|Z|).
	double chiSquared = 0;
	// chiSquared corrected for the number of free parameters.
	double reducedChiSquared = 0;
	double rmsRelativeError = 0;
	double maxRelativeError = 0;
	double noiseFloor = 0;
	double residualCorrelation = 0;
	FitVerdict verdict = FIT_VERDICT_UNFITTED; // Only meaningful when success is true.
	// Residual limits verdict for verdict
	double goodFitLimit = 0;
	double poorFitLimit = 0;
	double conditionNumber = 0; // Condition number of the normal matrix: large means the parameters are degenerate.
	int64_t evaluations = 0;
	int starts = 0;
	int bestStart = 0;
};

// @brief Fits an equivalent circuit model to a measured spectrum using Levenberg Marquardt in log10 space
//
// @param spectra the measured spectrum, complex, one entry per frequency
// @param omegas the angular frequencies the spectrum was measured at, rad/s
// @param modelString the circuit to fit, e.g. r-rp, syntax as understood by eis::Model
// @param options see FitOptions
// @return the fitted parameters and the diagnostics of the fit
FitResult eisFit(torch::Tensor spectra, torch::Tensor omegas, const std::string& modelString,
		const FitOptions& options = FitOptions());

// @brief Fits the same equivalent circuit to a batch of measured spectra at once.
//
// @param spectra one row per spectrum, complex, one entry per frequency. All spectra must share
// the frequency grid given in omegas
// @param omegas the angular frequencies all spectra were measured at, rad/s
// @param modelString the circuit to fit, e.g. r-rp, syntax as understood by eis::Model
// @param options see FitOptions. startingParams may be one {n} vector shared by the batch or a
// {n,B} tensor with one start per spectrum. maxEvaluations bounds the simulations per start for
// the batch as a whole, FitResult::evaluations reports that shared count
// @return one FitResult per row of spectra, in the same order
std::vector<FitResult> eisFitBatch(const torch::Tensor& spectra, const torch::Tensor& omegas,
		const std::string& modelString, const FitOptions& options = FitOptions());


// @brief Rewrites a model string with the given parameters substituted for its own.
//
// @param modelString the model the parameters were fitted for
// @param parameters one physical value per parameter of the model
// @return the model string with the parameters embedded, empty if it does not parse or the number
// of parameters does not fit
std::string fittedModelString(const std::string& modelString, const torch::Tensor& parameters);
