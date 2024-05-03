from sys import platform

def check_platform():

    if platform == "linux" or platform == "linux2":
        PLATFORM_PATH_PREFIX = "/HPS/"
    elif platform == "win32" or platform == "win64":
        PLATFORM_PATH_PREFIX = "Z:/"

    return PLATFORM_PATH_PREFIX



def win_api_path(dosPath):

	#empty path just return it
	if(dosPath ==""):
		return ""

	#replace the forward slashes with backward
	dosPath = dosPath.replace('/', '\\')
	return dosPath
	if dosPath.startswith(u"\\\\"):
		dosPath = u"\\\\?\\UNC\\" + dosPath[2:]
		return dosPath
	else:
		dosPath= u"\\\\?\\" + dosPath
		return dosPath