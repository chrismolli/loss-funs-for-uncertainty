from collections.abc import Callable
import tensorflow as tf
import tensorflow.keras.backend as K

def aleatoric_reg_loss(y_true, y_pred):
    """
      y_true: tf.Tensor ground truth for regression task
              dims := (batch, *dims)

      y_pred: tuple of length 2
              y_pred[0] tf.Tensor model estimation of regression
                        dims := (batch, *dims)
              y_pred[1] model estimation of aleatoric log variance
                        dims := (batch, *dims)
    """

    # calc aleatoric mse
    loss = K.pow(y_true - y_pred[0], 2)
    loss = K.exp(-y_pred[1]) * loss + y_pred[1]
    loss /= 2

    # mean loss (batch, *dims) -> (batch, )
    n_dims = len(y_true.shape) - 1
    for d in range(n_dims):
        loss = K.mean(loss, axis=-1)

    return loss

def AleatoricRegLoss() -> Callable:
    """
        Constructs aleatoric loss function for regression task
    """
    return lambda y_true, y_pred : aleatoric_reg_loss(y_true, y_pred)
