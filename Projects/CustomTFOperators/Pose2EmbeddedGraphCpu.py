
########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf
from tensorflow.python.framework import ops
import CustomTFOperators.CppPath as CppPath

########################################################################################################################
# Load custom operators
########################################################################################################################

customOperators = tf.load_op_library(CppPath.getCustomOperatorPath())

########################################################################################################################
# EmbeddedGraphGpu class
########################################################################################################################

class Pose2EmbeddedGraphCpu:

    def __init__(self,
                 characterFilePath = '',
                 graphFilePath = '',
                 dofs = None,
                 nodeName=''):

        self.characterFilePath = characterFilePath
        self.graphFilePath = graphFilePath
        self.dofs = dofs
        self.nodeName = nodeName

        self.pose2EmbeddedGraphCpuOperator = None

        if(characterFilePath != '' and graphFilePath != '' and dofs is not None and nodeName != ''):
            self.pose2EmbeddedGraphCpuOperator = customOperators.pose2_embedded_graph_cpu(dofs,
                                                                                character_file_path_eg = characterFilePath,
                                                                                graph_file_path = graphFilePath,
                                                                                name=nodeName)

        else:

            raise ValueError('Invalid argument during the construction of the pose2embeddedgraph graph operator!')

    # return the node for building a tf graph
    #outputs deformed vertices | deformed normals | deformed markers | deformed graph nodes
    def getNode(self):

        return [self.pose2EmbeddedGraphCpuOperator[0],self.pose2EmbeddedGraphCpuOperator[1]]

########################################################################################################################
# Register gradients
########################################################################################################################

@ops.RegisterGradient("Pose2EmbeddedGraphCpu")
def pose2_embedded_graph_cpu_grad(op, skinnedTGrad, skinnedAGrad):

    dofsGrad = tf.zeros(tf.shape(op.inputs[0]), tf.float32)

    return  dofsGrad

########################################################################################################################
#
########################################################################################################################
