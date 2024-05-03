
########################################################################################################################
# Imports
########################################################################################################################

import AdditionalUtils.ObjTools as OBJWriter

########################################################################################################################
# OutputWriter
########################################################################################################################

class OutputWriter:

    ################

    def __init__(self, outputPath, training):

        self._deleted   = False
        self.outputPath = outputPath
        self.training   = training

        outputObjFinal  = open(outputPath + 'final.meshes', 'w')
        outputObjFinal.write('Skeletool Meshes file v2.1 \n')
        self.outputObjFinal = outputObjFinal

        outputObjEGOnly = open(outputPath + 'egOnly.meshes', 'w')
        outputObjEGOnly.write('Skeletool Meshes file v2.1 \n')
        self.outputObjEGOnly = outputObjEGOnly

        outputObjGraph = open(outputPath + 'graph.meshes', 'w')
        outputObjGraph.write('Skeletool Meshes file v2.1 \n')
        self.outputObjGraph = outputObjGraph

        outputObjGraphNormalized = open(outputPath + 'graphNormalized.meshes', 'w')
        outputObjGraphNormalized.write('Skeletool Meshes file v2.1 \n')
        self.outputObjGraphNormalized = outputObjGraphNormalized

        outputObjDisplacementsNormalized = open(outputPath + 'displacementsNormalized.meshes', 'w')
        outputObjDisplacementsNormalized.write('Skeletool Meshes file v2.1 \n')
        self.outputObjDisplacementsNormalized = outputObjDisplacementsNormalized

        outputObjPoseOnly = open(outputPath + 'poseOnly.meshes', 'w')
        outputObjPoseOnly.write('Skeletool Meshes file v2.1 \n')
        self.outputObjPoseOnly = outputObjPoseOnly

    ################

    def profile(self, i, frameIdsData, finalVertexPositions, egOnlyVertexPositions, graphPositions, egGraphNodesNormalized, displacementsNormalized, verticesPoseOnly):

        if not self._deleted:

            if self.training:
                OBJWriter.exportVerticesTensorToMeshSequenceV21(i, finalVertexPositions[0],     self.outputObjFinal)
                OBJWriter.exportVerticesTensorToMeshSequenceV21(i, egOnlyVertexPositions[0],    self.outputObjEGOnly)
                OBJWriter.exportVerticesTensorToMeshSequenceV21(i, graphPositions[0],           self.outputObjGraph)
                OBJWriter.exportVerticesTensorToMeshSequenceV21(i, egGraphNodesNormalized[0],   self.outputObjGraphNormalized)
                OBJWriter.exportVerticesTensorToMeshSequenceV21(i, displacementsNormalized[0],  self.outputObjDisplacementsNormalized)
                OBJWriter.exportVerticesTensorToMeshSequenceV21(i, verticesPoseOnly[0],         self.outputObjPoseOnly)
            else:
                OBJWriter.exportVerticesTensorToMeshSequenceV21(frameIdsData[0][0], finalVertexPositions[0],    self.outputObjFinal)
                OBJWriter.exportVerticesTensorToMeshSequenceV21(frameIdsData[0][0], egOnlyVertexPositions[0],   self.outputObjEGOnly)
                OBJWriter.exportVerticesTensorToMeshSequenceV21(frameIdsData[0][0], graphPositions[0],          self.outputObjGraph)
                OBJWriter.exportVerticesTensorToMeshSequenceV21(frameIdsData[0][0], egGraphNodesNormalized[0],  self.outputObjGraphNormalized)
                OBJWriter.exportVerticesTensorToMeshSequenceV21(frameIdsData[0][0], displacementsNormalized[0], self.outputObjDisplacementsNormalized)
                OBJWriter.exportVerticesTensorToMeshSequenceV21(frameIdsData[0][0], verticesPoseOnly[0],        self.outputObjPoseOnly)

    ################

    def closeAllFiles(self):

        self.outputObjFinal.close()
        self.outputObjEGOnly.close()
        self.outputObjGraph.close()
        self.outputObjGraphNormalized.close()
        self.outputObjDisplacementsNormalized.close()
        self.outputObjPoseOnly.close()

        self._deleted = True

########################################################################################################################
#
########################################################################################################################