########################################################################################################################
# Imports
########################################################################################################################

import tensorflow as tf

########################################################################################################################
# Create summary
########################################################################################################################

def create_summary(egdefnet,
                   deltaNet,
                   texNet,
                   SRNet,
                   lossFinal,
                   lossSilhouette,
                   lossArap,
                   lossRender,
                   lossSR,
                   lossSpatial,
                   lossChamfer,
                   lossIso,
                   createEpochSummary,
                   iteration,
                   stgs,
                   masklossSR=None
                   ):

    ##########################################################################################################
    # loss

    ################################################
    # Scalars
# ---------------------
    
    # ---------------------
    # L2 final loss
    # ---------------------

    if (stgs.EG_NETWORK_MODE == 'training' or stgs.DELTA_NETWORK_MODE == 'training'or stgs.TEX_NETWORK_MODE == 'training'):
        tf.summary.scalar('training/lossFinal', lossFinal, step=iteration)

    # ---------------------
    # L2 silhouette loss
    # ---------------------

    if (stgs.EG_NETWORK_MODE == 'training' or stgs.DELTA_NETWORK_MODE == 'training'):
        tf.summary.scalar('training/lossSilhouette', lossSilhouette, step=iteration)

    # ---------------------
    # L2 ARAP loss
    # ---------------------

    if (stgs.EG_NETWORK_MODE == 'training' or stgs.DELTA_NETWORK_MODE == 'training'):
        tf.summary.scalar('training/lossArap', lossArap, step=iteration)

    # ---------------------
    # L2 Render loss
    # ---------------------

    if (stgs.LIGHTING_MODE == 'training' or stgs.EG_NETWORK_MODE == 'training' or stgs.DELTA_NETWORK_MODE == 'training' or stgs.TEX_NETWORK_MODE == 'training'):
        tf.summary.scalar('training/lossRender', lossRender, step=iteration)

    # ---------------------
    # Spatial loss
    # ---------------------

    if (stgs.EG_NETWORK_MODE == 'training' or stgs.DELTA_NETWORK_MODE == 'training' or stgs.TEX_NETWORK_MODE == 'training'):
        tf.summary.scalar('training/lossSpatial', lossSpatial, step=iteration)

    # ---------------------
    # L2 Chamfer loss
    # ---------------------

    if (stgs.LIGHTING_MODE == 'training' or stgs.EG_NETWORK_MODE == 'training' or stgs.DELTA_NETWORK_MODE == 'training' or stgs.TEX_NETWORK_MODE == 'training'):
        tf.summary.scalar('training/lossChamfer', lossChamfer, step=iteration)

    # ---------------------
    # L2 Iso loss
    # ---------------------

    if (stgs.LIGHTING_MODE == 'training' or stgs.EG_NETWORK_MODE == 'training' or stgs.DELTA_NETWORK_MODE == 'training' or stgs.TEX_NETWORK_MODE == 'training'):
        tf.summary.scalar('training/lossIso', lossIso, step=iteration)

    ##########################################################################################################
    # Per epoch
    # SR final loss
    # ---------------------

    if (stgs.SR_NETWORK_MODE == 'training'):
        tf.summary.scalar('training/lossSR', lossSR, step=iteration)
        if masklossSR is not None:
            tf.summary.scalar('training/masklossSR', masklossSR, step=iteration)
            
    if (createEpochSummary):

        # ---------------------
        # Trainable weights
        # ---------------------

        if (stgs.EG_NETWORK_MODE == 'training'):
            for variable in egdefnet.model.trainable_weights:
                tf.summary.histogram('egnet/' + variable.name, variable, step=iteration)

        if (stgs.DELTA_NETWORK_MODE == 'training'):
            for variable in deltaNet.model.trainable_weights:
                tf.summary.histogram('deltanet/' + variable.name, variable, step=iteration)

        if (stgs.TEX_NETWORK_MODE == 'training' and texNet is not None):
            for variable in texNet.model.trainable_weights:
                tf.summary.histogram('texnet/' + variable.name, variable, step=iteration)
        if (stgs.SR_NETWORK_MODE == 'training' and SRNet is not None):
            for variable in SRNet.model.trainable_weights:
                tf.summary.histogram('SRnet/' + variable.name, variable, step=iteration)
########################################################################################################################
#
########################################################################################################################