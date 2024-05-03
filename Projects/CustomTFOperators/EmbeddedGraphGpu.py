
########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
from tensorflow.python.framework import ops
import CustomTFOperators.CppPath as CppPath
#import CppPath
#import Pose2EmbeddedGraphCpu
########################################################################################################################
# Load custom operators
########################################################################################################################

customOperators = tf.load_op_library(CppPath.getCustomOperatorPath())

########################################################################################################################
# EmbeddedGraphGpu class
########################################################################################################################

class EmbeddedGraphGpu:

    def __init__(self,
                 characterFilePath = '',
                 graphFilePath = '',
                 numberOfBatches = 0,
                 deltaT = None,
                 deltaR = None,
                 skinnedT=None,
                 skinnedR=None,
                 displacements=None,
                 refinement = None,
                 nodeName=''):

        self.characterFilePath = characterFilePath
        self.graphFilePath = graphFilePath
        self.numberOfBatches = numberOfBatches
        self.deltaT = deltaT
        self.deltaR = deltaR
        self.skinnedT = skinnedT
        self.skinnedR = skinnedR
        self.displacements = displacements
        self.refinement = refinement
        self.nodeName = nodeName

        self.embeddedGraphGpuOperator = None

        if(characterFilePath != ''
                and graphFilePath != ''
                and numberOfBatches != 0
                and deltaT is not None
                and deltaR is not None
                and skinnedT is not None
                and skinnedR is not None
                and displacements is not None
                and nodeName != ''and refinement is not None):

            self.embeddedGraphGpuOperator = customOperators.embedded_graph_gpu( deltaT,
                                                                                deltaR,
                                                                                skinnedT,
                                                                                skinnedR,
                                                                                displacements,
                                                                                character_file_path_eg = characterFilePath,
                                                                                graph_file_path = graphFilePath,
                                                                                number_of_batches_eg= numberOfBatches,
                                                                                refinement=refinement)

        else:

            raise ValueError('Invalid argument during the construction of the embedded graph operator!')

    # return the node for building a tf graph
    #outputs deformed vertices | deformed normals | deformed markers | deformed graph nodes
    def getNode(self):

        return [self.embeddedGraphGpuOperator[0],self.embeddedGraphGpuOperator[1], self.embeddedGraphGpuOperator[2],self.embeddedGraphGpuOperator[3]]

########################################################################################################################
# Register gradients
########################################################################################################################

@ops.RegisterGradient("EmbeddedGraphGpu")
def embedded_graph_gpu_grad(op, deformedVerticesGrad, deformedNormalsGrad, deformedMarkersGrad, deformedGraphNodes, deltaAGrad, skinnedAGrad):

    nodesTGrad, nodesRGrad, nodesSkinnedTGrad, nodesSkinnedRGrad, displacementsGrad = customOperators.embedded_graph_gpu_grad(deformedVerticesGrad,
                                                                               deformedMarkersGrad,
                                                                               op.outputs[4],
                                                                               op.outputs[5],
                                                                               character_file_path_eg_grad = op.get_attr('character_file_path_eg'),
                                                                               graph_file_path_grad = op.get_attr('graph_file_path'),
                                                                               refinement_grad = op.get_attr('refinement')
                                                                               )

    return nodesTGrad, nodesRGrad, nodesSkinnedTGrad, nodesSkinnedRGrad, displacementsGrad

########################################################################################################################
#
########################################################################################################################
# if __name__=="__main__":
#     #character_path='/CT/ashwath/work/DDC_DATA/dummy_data/Oleks/actor.character'
#     #graph_path='/CT/ashwath/work/DDC_DATA/dummy_data/Oleks/actorSimplified.obj'
#     #dofs="1.69283 1.02942 -0.642821 0.843042 0.045369 -0.037208 9.5e-05 0.045904 -0.309312 0.0 0.386346 0.213993 -0.186151 0.065221 -0.033913 0.109963 -1.1225 -0.482952 -1.86913 0.000636 -0.220721 -0.151632 0.590049 0.19951 0.039773 0.464762 -1.14301 -0.521553 -1.66916 0.000544 -0.307723 0.093787 0.38957 0.281455 -0.008929 0.432278 -0.298251 0.007886 -0.141433 -0.046364 6.8e-05 -0.190481 1.6667 -0.2765 0.108344 -0.081235 -0.728987 0.005483 0.177906 -0.098116 -0.000131 0.063027 1.64245 0.00052"
#     #dofs=tf.constant([float(dof) for dof in dofs.split(' ')])
#     #dofs=tf.reshape(dofs,[1,54])
#     #pose2EmbeddedGraphCpu = Pose2EmbeddedGraphCpu.Pose2EmbeddedGraphCpu(
#                     characterFilePath=character_path,
#                     graphFilePath=graph_path,
#                     dofs=dofs,
#                     nodeName='pose2EmbeddedGraphCpu')
    
#     embeddedGraphOutputPoseOnly = EmbeddedGraphGpu(
#                     characterFilePath   = character_path,
#                     graphFilePath       = graph_path,
#                     numberOfBatches     = 1,
#                     deltaT              = tf.zeros((1,487,3)),
#                     deltaR              = tf.zeros((1,487,3)),
#                     skinnedT            = pose2EmbeddedGraphCpu.getNode()[0],
#                     skinnedR            = pose2EmbeddedGraphCpu.getNode()[1],
#                     displacements       = tf.zeros([1, 4847, 3]),
#                     refinement          = False,
#                     nodeName            = 'embedded_graph_operator_pose_only')
#     import pdb
#     pdb.set_trace()

