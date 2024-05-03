#include "StringProcessing.h"

//==============================================================================================//

void splitString(std::vector<std::string>& tokenVector, const std::string& input, const  std::string& delimiter, bool trim)
{
	tokenVector.clear();
	// don't do anything if we have an empty delimiter
	if (delimiter == "")
	{
		tokenVector.push_back(input);
		return;
	}

	size_t  start = 0, end = 0;

	while (end != std::string::npos)
	{
		end = input.find(delimiter, start);
		const std::string token = input.substr(start, (end == std::string::npos) ? std::string::npos : end - start);
		if (!trim || (token.length() > 0 && token.compare(delimiter) != 0))
			tokenVector.push_back(token);
		start = ((end > (std::string::npos - delimiter.size())) ? std::string::npos : end + delimiter.size());
	}

	if (tokenVector.size() == 0)
		tokenVector.push_back(std::string(""));
}

//==============================================================================================//

bool getTokens(std::ifstream& fh, std::vector<std::string>& tokenVector, const std::string& delimiter, bool trim)
{
#define BUFFERSIZE 16000 //8000 was too small for 200 dofs 2048
	static char buffer[BUFFERSIZE];
	tokenVector.clear();

	// check if the file is still valid
	if (!fh.good())
		return false;

	// read line
	fh.getline(buffer, BUFFERSIZE);
	
	
	// check if we reached the end of the file or an error arised
	if (!fh.good())
		return false;

#if defined(unix) || defined(__unix) || defined(__unix__) || defined(__linux__) || defined(linux) || defined(__linux)
	buffer[strlen(buffer) - 1] = '\0';
#endif


	// split the string
	std::string line(buffer);
	splitString(tokenVector, line, delimiter, trim);
	return true;
}

//==============================================================================================//