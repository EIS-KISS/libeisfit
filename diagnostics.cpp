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

#include "diagnostics.h"

#include <cmath>
#include <vector>

#include "constants.h"

namespace eisfit_internal
{

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

}
