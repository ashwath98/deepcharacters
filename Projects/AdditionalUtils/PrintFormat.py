from colorama import Fore, Back, Style

def print1Level(inputString):
    print(Fore.LIGHTGREEN_EX + '---' + inputString + Fore.WHITE, flush=True)

def print2Level(inputString):
    print('   ' + inputString, flush=True)

def print3Level(inputString):
    print('      ' + inputString, flush=True)

def printWarning(inputString):
    print(Fore.LIGHTYELLOW_EX + inputString + Fore.WHITE, flush=True)

def printError(inputString):
    print(Fore.RED + inputString + Fore.WHITE, flush=True)

def prRed(skk): print("\033[91m {}\033[00m".format(skk))
def prGreen(skk): print("\033[92m {}\033[00m".format(skk))
def prYellow(skk): print("\033[93m {}\033[00m".format(skk))
def prLightPurple(skk): print("\033[94m {}\033[00m".format(skk))
def prPurple(skk): print("\033[95m {}\033[00m".format(skk))
def prCyan(skk): print("\033[96m {}\033[00m".format(skk))
def prLightGray(skk): print("\033[97m {}\033[00m".format(skk))
def prBlack(skk): print("\033[98m {}\033[00m".format(skk))