import numpy as np

########################################################################################################################
# Skeleton Reader
########################################################################################################################

class SkeletonReader:

    ########################################################################################################################

    def __init__(self, filename):

        self.filename = filename
        self.limits = []
        file = open(filename, 'r')

        for line in file:

            splittedLine = line.split()

            if len(splittedLine) > 0:
                if splittedLine[0] == 'joints:':
                    self.num_joints = int(splittedLine[1])
                if splittedLine[0] == 'markers:':
                    self.num_markers = int(splittedLine[1])
                if splittedLine[0] == 'dofs:':
                    self.num_dofs = int(splittedLine[1])

                if splittedLine[0] == 'limits':
                    self.limits.append([float(splittedLine[1]), splittedLine[2]])
                if splittedLine[0] == 'nolimits':
                    self.limits.append([-10000000.0, 10000000.0])