# loss-funs-for-uncertainty
Custom loss functions for Keras models to estimate predictive uncertainty. The loss implementations are based on [[Kendall2017](https://arxiv.org/pdf/1703.04977.pdf)] and contain a logistic loss for classification and a mse-like regression loss function.

## Usage
Import the wanted loss constructor from its source and pass the y_true and y_pred tensors to it.
```
  from aleatoric_log_loss import AleatoricLogLoss

  lfun = AleatoricLogLoss()
  loss = lfun(y_true, y_pred)
```
Alternatively, it can also be used upon model compilation.
```
  from tensorflow import keras

  model = keras.Sequential()
  ...

  model.compile(loss = AleatoricLogLoss())

```
Make sure that the prediction tensor is a tuple containing `(y_pred, std_pred)`.
