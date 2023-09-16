from .copy_weights import W_Torch2TF
from .covert_dataloader import Torch_loader2TF_loader, TF_loader2Torch_loader
import torch
import tensorflow as tf
from .trainer_checker import TrainerChecker
from .utils import SetZeroLearningRate

def test_trainer(
    tf_trainer, torch_trainer, 
    tf_model, torch_model,
    tf_optimizer, torch_optimizer,
    tf_train_loader = None, torch_train_loader = None,
    tf_test_loader = None, torch_test_loader = None,
    train_metrics = ['loss'], test_metrics = ['full'],
    batch_size = 16, loader_length=10,
    ):
  is_train = tf_train_loader or torch_train_loader
  is_test = tf_test_loader or torch_test_loader
  W_Torch2TF(torch_model, tf_model)
  print("Info: Copy weights TF2Torch passed.\n")
  if torch.cuda.is_available():
    torch_model = torch_model.to('cuda')

  SetZeroLearningRate(torch_optimizer, tf_optimizer)
  print("Info: SetZeroLearningRate done.\n")

  # Prepare dataloader
  if torch_train_loader: 
    tf_train_loader = Torch_loader2TF_loader(torch_train_loader, batch_size, loader_length)
  if tf_train_loader: 
    torch_train_loader = TF_loader2Torch_loader(tf_train_loader, batch_size, loader_length)
    tf_train_loader = tf_train_loader.unbatch().batch(batch_size)
  if torch_test_loader: 
    tf_test_loader = Torch_loader2TF_loader(torch_test_loader, batch_size, loader_length)
  if tf_test_loader: 
    torch_test_loader = TF_loader2Torch_loader(tf_test_loader, batch_size, loader_length)
    tf_test_loader = tf_test_loader.unbatch().batch(batch_size)
  print("Info: Prepare dataloader done.\n")
  # tf_train_loader = tf_train_loader.unbatch().batch(batch_size)
  trainer_checker = TrainerChecker(
    torch_trainer, tf_trainer, 
    train_metrics, test_metrics,
    loader_length, batch_size
    )
  if is_train:
    if not trainer_checker.check_train_step(tf_train_loader, torch_train_loader):
      return False
  if is_test:
    if not trainer_checker.chek_test_step(tf_test_loader, torch_test_loader):
      return False
  return True