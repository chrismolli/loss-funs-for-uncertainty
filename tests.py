import tensorflow as tf
import tensorflow.keras.backend as K
import unittest
import pytest

from aleatoric_log_loss import AleatoricLogLoss

class LogLossTests(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def init_params(self):
        self.params = {
            "batch_size" : 16,
            "width" : 32,
            "height" : 32,
            "n_classes" : 2,
            "n_samples" : 20,
        }
        self.loss_fun = AleatoricLogLoss(self.params["n_samples"])

    def create_rand_norm(self):
        return K.random_normal((self.params["batch_size"],
                                self.params["width"],
                                self.params["height"],
                                self.params["n_classes"]))

    def test1_loss_shape(self):
        y_true = self.create_rand_norm()
        y_pred = (y_true, y_true)
        loss = self.loss_fun(y_true, y_pred)
        assert loss.shape == self.params["batch_size"]

    def test2_loss_comparison(self):
        y_true = self.create_rand_norm()
        y_pred = (y_true, y_true)
        loss1 = K.sum(self.loss_fun(y_true, y_pred))

        y_pred = (1-y_true,
                  y_true)
        loss2 = K.sum(self.loss_fun(y_true, y_pred))
        assert loss1 < loss2
