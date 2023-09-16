import tensorflow as tf

def SetZeroLearningRate(torch_optimizer, tf_optimizer):
  new_lr = 0.
  for param_group in torch_optimizer.param_groups:
    param_group['lr'] = new_lr
  tf.keras.backend.set_value(tf_optimizer.learning_rate, new_lr)