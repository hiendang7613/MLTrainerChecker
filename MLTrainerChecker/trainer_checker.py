from tqdm import tqdm
import numpy as np

class TrainerChecker:
    def __init__(self, torch_trainer, tf_trainer,
                  train_metrics, test_metrics,
                  length, batch_size):
        self.torch_trainer = torch_trainer
        self.tf_trainer = tf_trainer
        self.train_metrics = train_metrics
        self.test_metrics = test_metrics
        self.length = length
        self.batch_size = batch_size       
        pass
    def check_train_step(self, tf_dataloader, torch_dataloader):
        torch_iter = iter(torch_dataloader)
        tf_iter = iter(tf_dataloader)
        for i in tqdm(range(self.length), total=self.length):
          torch_data, tf_data = next(torch_iter), next(tf_iter)
          tf_metrics = self.tf_trainer.train_step(tf_data)#['loss'].numpy()
          torch_metrics = self.torch_trainer.train_step(torch_data)#['loss'].detach().numpy()
          for metric_name in self.train_metrics:
            if not np.isclose(torch_metrics[metric_name], tf_metrics[metric_name]):
              print("Error: Different train_step !!\n")
              return False
        print("Info: Check train_step passed !!\n")
        return True

    def chek_test_step(self, tf_dataloader, torch_dataloader):  
        tf_metrics_names = [ metric.name for metric in self.tf_trainer.metrics]
        tf_metrics = self.tf_trainer.evaluate(tf_dataloader, steps=self.length)
        tf_metrics = dict(zip(tf_metrics_names , tf_metrics))
        torch_metrics = self.torch_trainer.evaluate(torch_dataloader, steps=self.length)
        if self.test_metrics[0] == 'full': # type: ignore
          self.test_metrics = set(torch_metrics.keys()) & set(tf_metrics.keys())
        for metric_name in self.test_metrics:
            if not np.isclose(torch_metrics[metric_name], tf_metrics[metric_name], rtol=1e-3, atol=1e-3):
              print("Error: Different test_step !!\n")
              print(metric_name, torch_metrics[metric_name], tf_metrics[metric_name])
              return False
        print("Info: Check test_step passed !!\n")
        return True