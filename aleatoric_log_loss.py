"""
    This custom loss is design for uncertainty prediction in
    image classification and segmentation tasks. It is based on
    [Kendall2017](https://arxiv.org/pdf/1703.04977.pdf), Section 3.3.
"""

from collections.abc import Callable
import tensorflow as tf
import tensorflow.keras.backend as K

def sample_logit(y_pred : tuple):
    noise_weight = K.random_normal(y_pred[1].shape, mean=0, stddev=1)
    return y_pred[0] + noise_weight * y_pred[1]

def aleatoric_log_loss(y_true, y_pred : tuple, n_samples : int):
  """
      y_true: tf.Tensor ground truth for logit labels
              dims := (batch, width, height, n_channel)

      y_pred: tuple of length 2
              y_pred[0] tf.Tensor model estimation of logit labels
                        dims := (batch, width, height, n_channel)
              y_pred[1] model estimation of aleatoric stddev
                        dims := (batch, width, height, n_channel)
  """

  # init loss
  loss = tf.zeros((1,1))

  # sum up loss for samples logits
  for t in range(n_samples):
    # sample logit from predicted uncertainty
    sample = sample_logit(y_pred)
    # sum up class labels
    temp = K.sum(K.exp(y_true), axis=-1, keepdims=True)
    temp = K.exp(sample - K.log(temp))
    loss += temp

  # convert to log mean
  loss /= n_samples
  loss = K.log(loss)

  # sum up spatial dimensions (batch, height, width, 1) -> (batch, 1)
  loss = K.sum(loss, axis=-1)
  loss = K.sum(loss, axis=-1)
  loss = K.sum(loss, axis=-1)

  return loss

def AleatoricLogLoss(n_samples : int = 100) -> Callable:
  """
     Constructs aleatoric loss function for classification tasks.
     n_samples: number of samples s_t to draw from s_t = f + eta_t * sigma
  """
  return lambda y_true, y_pred : aleatoric_log_loss(y_true, y_pred, n_samples)
