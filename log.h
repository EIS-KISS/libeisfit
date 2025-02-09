/* * libeisfit - Levenberg-Marquardt fitting of equivalent circuit models to measured EIS spectra
 * Copyright (C) 2025 Carl Klemm <carl@uvos.xyz>
 *
 * This file is part of libeisfit.
 *
 * libeisfit is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libeisfit is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with libeisfit.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once
#include <iostream>
#include <string>

class Log
{
public:

	enum Level
	{
		DEBUG,
		INFO,
		WARN,
		ERROR
	};

private:
	bool opened = false;
	Level msglevel = DEBUG;
	bool endline = true;

	std::string getLabel(Level level);

public:

	static bool headers;
	static Level level;

	Log() {}
	Log(Level type, bool endlineI = true);
	~Log();

	template<class T> Log &operator<<(const T &msg)
	{
		if(msglevel >= level)
		{
			if(msglevel == ERROR)
				std::cerr<<msg;
			else
				std::cout<<msg;
			opened = true;
		}
		return *this;
	}
};
