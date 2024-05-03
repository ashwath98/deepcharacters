
########################################################################################################################
# Imports
########################################################################################################################

import sys
from Utils.ArgParser import config_parser

sys.path.append("../")

import HoloportedCharacters.HPCMeshDL as RunDeepDynamicCharactersSRTexArch
import HoloportedCharacters.Utils.Settings as Settings

########################################################################################################################

print('--Parse args', flush=True)
args = config_parser().parse_args()

print('--Setup the settings', flush=True)
stgs = Settings.Settings(args)

print('--Write the settings log', flush=True)
stgs.write_settings()

print('--Check if training is already finished!', flush=True)
stgs.check_exit()

print('--Initialize the runner', flush=True)
runner = RunDeepDynamicCharactersSRTexArch.DeepDynamicCharacterRunner(stgs)

print('--Run the loop', flush=True)
runner.run_loop()