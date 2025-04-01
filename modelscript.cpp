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

#include "modelscript.h"

#include <cassert>

#include <torch/csrc/jit/api/function_impl.h>

namespace eisfit_internal
{

std::shared_ptr<torch::CompilationUnit> compileModel(eis::Model &model)
{
	std::string torchScript = model.getTorchScript();
	std::shared_ptr<torch::CompilationUnit> compiledModule = torch::jit::compile(torchScript);
	return compiledModule;
}

torch::Tensor runScriptModel(eis::Model& model, std::shared_ptr<torch::CompilationUnit> compiledScript,
							torch::Tensor parameters, torch::Tensor omegas)
{
	if(omegas.device() != parameters.device())
		omegas = omegas.to(parameters.device());

	torch::Tensor result = compiledScript->run_method(model.getCompiledFunctionName(), parameters, omegas).toTensor();
	assert(result.scalar_type() == torch::kComplexFloat || result.scalar_type() == torch::kComplexDouble);
	assert(result.size(0) == omegas.numel());
	const int64_t expectedColumns = parameters.dim() > 1 ? parameters.size(1) : 1;
	assert(result.sizes().size() == 1 || result.size(1) == expectedColumns);
	return result.reshape({result.numel()});
}

}
