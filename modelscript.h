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

#include <memory>
#include <string>

#include <eisgenerator/model.h>

#include <torch/jit.h>
#include <torch/torch.h>

namespace eisfit_internal
{

// Compile the TorchScript the generator emits for a model and evaluate it. These mirror the
// helpers of the same name in TorchKissAnn's utils/modelscript.cpp, which the fit was taken
// from; the fit is the only consumer here.
std::shared_ptr<torch::CompilationUnit> compileModel(eis::Model &model);

torch::Tensor runScriptModel(eis::Model& model, std::shared_ptr<torch::CompilationUnit> compiledScript,
							torch::Tensor parameters, torch::Tensor omegas);

}
