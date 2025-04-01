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

#include "bounds.h"

#include <algorithm>
#include <cctype>
#include <cmath>

#include <eisgenerator/componant/componant.h>

#include "constants.h"

namespace
{

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

}

namespace eisfit_internal
{

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

}
