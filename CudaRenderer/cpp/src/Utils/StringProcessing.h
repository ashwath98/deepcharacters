//==============================================================================================//
// Classname:
//      StringProcessing
//
//==============================================================================================//
// Description:
//      Contains useful string operations such as splitting etc.
//
//==============================================================================================//

#pragma once

//==============================================================================================//

#include <math.h>
#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <iomanip>
#include <float.h>
#include <assert.h>
#include <limits>
#include <string>
#include <algorithm>
#include <vector>
#include <cstring>

//==============================================================================================//

void splitString(std::vector<std::string>& tokenVector, const  std::string& input, const  std::string& delimiter = " ", bool trim = true);

//==============================================================================================//

bool getTokens(std::ifstream& fh, std::vector<std::string>& tokenVector, const std::string& delimiter = " ", bool trim = true);

//==============================================================================================//

template <class T>
bool fromString(T& t, const std::string& s, std::ios_base & (*f)(std::ios_base&) = std::dec)
{
	std::istringstream iss(s);
	return !(iss >> f >> t).fail();
}

//==============================================================================================//

template <class T>
std::string toString(const T& t)
{
	std::ostringstream oss;
	oss << t;
	return oss.str();
}

//==============================================================================================//
