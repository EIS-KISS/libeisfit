//
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
//

#include <algorithm>
#include <cmath>
#include <complex>
#include <cstdint>
#include <eisgenerator/log.h>
#include <eisgenerator/model.h>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <kisstype/spectra.h>
#include <kisstype/type.h>
#include <map>
#include <sstream>
#include <string>
#include <torch/torch.h>
#include <vector>

#include "eisfit/fit.h"
#include "log.h"
#include "options.h"

namespace
{

torch::Tensor spectraToTensor(const std::vector<eis::DataPoint>& data, torch::Tensor* omegas)
{
	torch::Tensor values = torch::empty({static_cast<int64_t>(data.size())},
			torch::TensorOptions().dtype(torch::kComplexDouble));
	torch::Tensor freqs = torch::empty({static_cast<int64_t>(data.size())},
			torch::TensorOptions().dtype(torch::kFloat64));

	auto valueAccessor = values.accessor<c10::complex<double>, 1>();
	auto freqAccessor = freqs.accessor<double, 1>();
	for(size_t i = 0; i < data.size(); ++i)
	{
		double real = data[i].im.real();
		double imag = data[i].im.imag();
		if(std::isnan(real) || std::isinf(real))
			real = 0;
		if(std::isnan(imag) || std::isinf(imag))
			imag = 0;
		valueAccessor[i] = c10::complex<double>(real, imag);
		freqAccessor[i] = data[i].omega;
	}

	if(omegas)
		*omegas = freqs;
	return values;
}

std::string sci(double value, int precision = 4)
{
	std::ostringstream stream;
	stream<<std::scientific<<std::setprecision(precision)<<value;
	return stream.str();
}

std::string percent(double fraction, int precision = 3)
{
	std::ostringstream stream;
	stream<<std::fixed<<std::setprecision(precision)<<fraction*100<<" %";
	return stream.str();
}

void printParameters(const FitResult& fit)
{
	std::cout<<"\nParameter           Start         Fitted        +/- 1 sigma   Relative    Note\n";

	for(int64_t i = 0; i < fit.parameters.numel(); ++i)
	{
		double value = fit.parameters[i].item<double>();
		double start = fit.startParameters[i].item<double>();
		double error = fit.uncertainties[i].item<double>();

		std::string note;
		if(fit.atBound[i])
			note = "on bound";
		else if(std::isnan(error))
			note = "not identifiable";

		std::cout<<"  "<<std::setw(2)<<i<<' '<<std::left<<std::setw(14)<<fit.parameterNames[i]
			<<std::right<<std::setw(11)<<sci(start)
			<<std::setw(12)<<sci(value)
			<<std::setw(14)<<sci(error)
			<<std::setw(12)<<percent(std::abs(error/value))
			<<' '<<note<<'\n';
	}
}

void printQuality(const FitResult& fit)
{
	std::cout<<"\nFit quality\n"
		<<"  chi^2 per point          "<<sci(fit.chiSquared)<<'\n'
		<<"  reduced chi^2            "<<sci(fit.reducedChiSquared)<<'\n'
		<<"  rms relative residual    "<<percent(fit.rmsRelativeError)<<'\n'
		<<"  largest point residual   "<<percent(fit.maxRelativeError)<<'\n'
		<<"  noise floor of data      "<<(fit.noiseFloor > 0 ? percent(fit.noiseFloor) : "not estimable")<<'\n'
		<<"  residual autocorrelation "<<std::fixed<<std::setprecision(2)<<fit.residualCorrelation<<'\n'
		<<"  condition number         "<<sci(fit.conditionNumber)<<'\n'
		<<"  optimiser                "<<(fit.converged ? "converged" : "out of budget")<<", "<<fit.evaluations
		<<" simulations, start "<<fit.bestStart<<" of "<<fit.starts<<'\n';
}

/**
 * Say why eisFit() reached the verdict it did; the limits themselves are chosen by the fit, see
 * FitResult::goodFitLimit.
 */
void printVerdict(const FitResult& fit)
{
	switch(fit.verdict)
	{
		case FitVerdict::FIT_VERDICT_UNFITTED:
			Log(Log::INFO)<<"Verdict: the model does not fit the spectra, its residual is "
				<<percent(fit.rmsRelativeError)<<" where at worst "<<percent(fit.poorFitLimit)
				<<" should be reachable";
			break;
		case FitVerdict::FIT_VERDICT_MARGINAL:
			if(fit.rmsRelativeError > fit.goodFitLimit)
				Log(Log::INFO)<<"Verdict: marginal, the residual of "<<percent(fit.rmsRelativeError)
					<<" is above the "<<percent(fit.goodFitLimit)<<" a fitting model should reach";
			else
				Log(Log::INFO)<<"Verdict: marginal, the residual of "<<percent(fit.rmsRelativeError)
					<<" is small but varies smoothly with the frequency (autocorrelation "
					<<std::fixed<<std::setprecision(2)<<fit.residualCorrelation
					<<"), the model misses a feature of the spectra";
			break;
		case FitVerdict::FIT_VERDICT_FIT:
			Log(Log::INFO)<<"Verdict: the model fits the spectra, its residual of "
				<<percent(fit.rmsRelativeError)<<" is within the "<<percent(fit.goodFitLimit)<<" limit";
			break;
	}
}

void printWarnings(const FitResult& fit)
{
	if(!fit.converged)
		Log(Log::WARN)<<"the optimiser ran out of its budget, raise --evaluations if the residual looks too large";

	for(size_t i = 0; i < fit.atBound.size(); ++i)
		if(fit.atBound[i])
			Log(Log::WARN)<<fit.parameterNames[i]<<" ended on a bound of "
				<<sci(fit.lowerBounds[static_cast<int64_t>(i)].item<double>())<<" to "
				<<sci(fit.upperBounds[static_cast<int64_t>(i)].item<double>())
				<<", give it a wider range in the model string if that is not intended";

	if(fit.conditionNumber > 1e6 || std::isinf(fit.conditionNumber))
		Log(Log::WARN)<<"the parameters are degenerate (condition number "<<sci(fit.conditionNumber)
			<<"), the spectra do not determine them individually";
}

bool writeFittedSpectra(const std::filesystem::path& output, const std::filesystem::path& source,
		const FitResult& fit, const std::vector<eis::DataPoint>& measured, const std::string& modelString)
{
	if(modelString.empty())
	{
		Log(Log::WARN)<<"the fitted parameters do not fit back into the model string, writing the spectra without one";
	}

	torch::Tensor fitted = fit.fittedSpectra.to(torch::kComplexDouble);
	auto accessor = fitted.accessor<c10::complex<double>, 1>();

	std::vector<eis::DataPoint> points;
	points.reserve(measured.size());
	for(size_t i = 0; i < measured.size(); ++i)
		points.push_back(eis::DataPoint(std::complex<fvalue>(static_cast<fvalue>(accessor[i].real()),
				static_cast<fvalue>(accessor[i].imag())), measured[i].omega));

	try
	{
		eis::Spectra spectra(points, modelString, "Fitted to "+source.string()+" by eisfit, rms residual "
				+percent(fit.rmsRelativeError));
		spectra.saveToDisk(output);
	}
	catch(const eis::file_error& error)
	{
		Log(Log::ERROR)<<"could not write "<<output<<": "<<error.what();
		return false;
	}

	return true;
}

const char* verdictName(const FitResult& fit)
{
	if(not fit.success)
		return "no fit";
	switch(fit.verdict)
	{
		case FitVerdict::FIT_VERDICT_FIT:
			return "fits";
		case FitVerdict::FIT_VERDICT_MARGINAL:
			return "marginal";
		case FitVerdict::FIT_VERDICT_UNFITTED:
			return "does not fit";
	}
	return "unknown";
}

struct Spectrum
{
	std::filesystem::path path;
	std::vector<eis::DataPoint> measured;
	torch::Tensor values;
	torch::Tensor omegas;
};

/**
 * Fit every csv in a directory to the same circuit, batched.
 *
 * The batched fit simulates all spectra of a batch against one shared frequency column, so spectra
 * are grouped by their grid before they are handed to eisFitBatch(); a directory that mixes point
 * counts simply produces several batches. Each spectrum prints one line, the summary counts the
 * verdicts.
 */
int fitDirectory(const Config& config)
{
	std::vector<std::filesystem::path> paths;
	if(config.recursive)
	{
		for(const auto& entry: std::filesystem::recursive_directory_iterator(config.fileName))
			if(entry.is_regular_file() and entry.path().extension() == ".csv")
				paths.push_back(entry.path());
	}
	else
	{
		for(const auto& entry: std::filesystem::directory_iterator(config.fileName))
			if(entry.is_regular_file() and entry.path().extension() == ".csv")
				paths.push_back(entry.path());
	}
	std::sort(paths.begin(), paths.end());

	if(paths.empty())
	{
		Log(Log::ERROR)<<"no csv files found in "<<config.fileName;
		return 1;
	}

	std::vector<Spectrum> spectra;
	int64_t loadFailed = 0;
	for(const auto& path: paths)
	{
		try
		{
			std::vector<eis::DataPoint> measured = eis::Spectra::loadFromDisk(path).data;
			if(measured.size() < 3)
			{
				Log(Log::WARN)<<path<<" contains "<<measured.size()<<" usable points, at least 3 are needed";
				++loadFailed;
				continue;
			}
			std::sort(measured.begin(), measured.end());
			torch::Tensor omegas;
			torch::Tensor values = spectraToTensor(measured, &omegas);
			spectra.push_back({path, measured, values, omegas});
		}
		catch(const std::exception& error)
		{
			Log(Log::WARN)<<"could not load "<<path<<": "<<error.what();
			++loadFailed;
		}
	}

	if(spectra.empty())
	{
		Log(Log::ERROR)<<"none of the "<<paths.size()<<" files in "<<config.fileName<<" could be loaded";
		return 1;
	}

	std::map<std::vector<double>, std::vector<size_t>> groups;
	for(size_t i = 0; i < spectra.size(); ++i)
	{
		torch::Tensor grid = spectra[i].omegas.to(torch::kFloat64);
		std::vector<double> key(grid.size(0));
		std::copy_n(grid.data_ptr<double>(), grid.size(0), key.begin());
		groups[std::move(key)].push_back(i);
	}

	if(!config.outputFilename.empty())
		std::filesystem::create_directories(config.outputFilename);

	FitOptions options;
	options.restarts = config.restarts;
	options.maxEvaluations = config.evaluations;
	options.seed = config.seed;
	options.fitLimit = config.fitLimit;
	options.searchPoints = config.searchPoints;
	options.approximateJacobian = !config.exactJacobian;

	const int64_t batch = std::max<int64_t>(1, config.batchSize);

	Log(Log::INFO)<<"Fitting "<<config.modelString<<" to "<<spectra.size()<<" spectra in "
		<<config.fileName<<(config.recursive ? " including subdirectories" : "")
		<<", "<<groups.size()<<" distinct frequency grid"<<(groups.size() == 1 ? "" : "s");

	std::cout<<"\nVerdict           rms  File\n";

	int64_t fits = 0, marginal = 0, unfit = 0, noFit = 0;
	bool writeFailed = false;

	for(const auto& [grid, members]: groups)
	{
		for(size_t begin = 0; begin < members.size(); begin += static_cast<size_t>(batch))
		{
			size_t end = std::min(members.size(), begin+static_cast<size_t>(batch));

			std::vector<torch::Tensor> rows;
			rows.reserve(end-begin);
			for(size_t k = begin; k < end; ++k)
				rows.push_back(spectra[members[k]].values);

			std::vector<FitResult> results;
			try
			{
				results = eisFitBatch(torch::stack(rows), spectra[members[begin]].omegas,
						config.modelString, options);
			}
			catch(const std::exception& error)
			{
				// eis::Model throws on a circuit it cannot parse or compile, which is a bad
				// command line rather than a failed fit
				Log(Log::ERROR)<<"could not use the model "<<config.modelString<<": "<<error.what();
				return 1;
			}

			for(size_t k = begin; k < end; ++k)
			{
				const Spectrum& entry = spectra[members[k]];
				const FitResult& fit = results[k-begin];
				std::string modelString = fit.success ? fittedModelString(config.modelString, fit.parameters) : "";

				std::cout<<std::left<<std::setw(12)<<verdictName(fit)
					<<std::right<<std::setw(10)<<(fit.success ? percent(fit.rmsRelativeError) : std::string("-"))
					<<"  "<<std::filesystem::relative(entry.path, config.fileName).string();
				if(!modelString.empty())
					std::cout<<"  "<<modelString;
				std::cout<<'\n';

				switch(fit.verdict)
				{
					case FitVerdict::FIT_VERDICT_FIT:
						++fits;
						break;
					case FitVerdict::FIT_VERDICT_MARGINAL:
						++marginal;
						break;
					case FitVerdict::FIT_VERDICT_UNFITTED:
						++unfit;
						break;
				}
				if(not fit.success)
					++noFit;

				if(!config.outputFilename.empty() && fit.success)
				{
					std::filesystem::path out = config.outputFilename
						/ std::filesystem::relative(entry.path, config.fileName);
					if(!writeFittedSpectra(out, entry.path, fit, entry.measured, modelString))
						writeFailed = true;
					else
						Log(Log::DEBUG)<<"Wrote the spectrum of the fitted circuit to "<<out;
				}
			}
		}
	}

	std::cout<<"\nSummary: "<<fits<<" fit, "<<marginal<<" marginal, "<<unfit<<" do not fit, "
		<<noFit<<" produced no fit, "<<loadFailed<<" could not be loaded\n";

	if(writeFailed)
		return 3;
	if(unfit > 0 or noFit > 0)
		return 4;
	return 0;
}

}

int main(int argc, char** argv)
{
	Log::level = Log::INFO;
	eis::Log::level = eis::Log::ERROR;

	Config config;
	argp_parse(&argp, argc, argv, 0, 0, &config);

	if(config.fileName.empty())
	{
		Log(Log::ERROR)<<"A spectrum to fit must be given with -f";
		return 1;
	}
	if(config.modelString.empty())
	{
		Log(Log::ERROR)<<"A model to fit must be given with -m";
		return 1;
	}
	if(!(config.fitLimit > 0))
	{
		Log(Log::ERROR)<<"--fit-limit must be a positive fraction";
		return 1;
	}

	if(std::filesystem::is_directory(config.fileName))
		return fitDirectory(config);

	std::vector<eis::DataPoint> measured;
	try
	{
		measured = eis::Spectra::loadFromDisk(config.fileName).data;
	}
	catch(const eis::file_error& error)
	{
		Log(Log::ERROR)<<"could not load "<<config.fileName<<": "<<error.what();
		return 1;
	}

	if(measured.size() < 3)
	{
		Log(Log::ERROR)<<config.fileName<<" contains "<<measured.size()<<" usable points, at least 3 are needed";
		return 1;
	}
	std::sort(measured.begin(), measured.end());

	torch::Tensor omegas;
	torch::Tensor target = spectraToTensor(measured, &omegas);

	FitOptions options;
	options.restarts = config.restarts;
	options.maxEvaluations = config.evaluations;
	options.seed = config.seed;
	options.fitLimit = config.fitLimit;
	options.searchPoints = config.searchPoints;
	options.approximateJacobian = !config.exactJacobian;

	Log(Log::INFO)<<"Fitting "<<config.modelString<<" to "<<config.fileName<<" ("<<measured.size()
		<<" points, "<<sci(measured.front().omega)<<" to "<<sci(measured.back().omega)<<" rad/s)";

	FitResult fit;
	try
	{
		fit = eisFit(target, omegas, config.modelString, options);
	}
	catch(const std::exception& error)
	{
		Log(Log::ERROR)<<"could not use the model "<<config.modelString<<": "<<error.what();
		return 1;
	}
	if(!fit.success)
	{
		Log(Log::ERROR)<<"no fit could be produced for "<<config.modelString;
		return 2;
	}

	printParameters(fit);
	printQuality(fit);

	std::string modelString = fittedModelString(config.modelString, fit.parameters);
	if(!modelString.empty())
		Log(Log::INFO)<<"Fitted model: "<<modelString;

	printVerdict(fit);
	printWarnings(fit);

	if(!config.outputFilename.empty() && !writeFittedSpectra(config.outputFilename, config.fileName, fit,
			measured, modelString))
		return 3;

	return fit.verdict == FitVerdict::FIT_VERDICT_UNFITTED ? 4 : 0;
}
