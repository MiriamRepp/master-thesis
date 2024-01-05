import numpy as np
import torch
from matplotlib import pyplot as plt
from nemo.collections.asr.models import EncDecClassificationModel
from nemo.collections.common.metrics import TopKClassificationAccuracy
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torchmetrics import ConfusionMatrix, Accuracy, F1Score


class LoggingEncDecClassificationModel(EncDecClassificationModel):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)

        self.confusion_matrix_save_path = None

    def _setup_metrics(self):
        num_classes = len(self.cfg.labels)
        self._accuracy = TopKClassificationAccuracy(dist_sync_on_step=True, top_k=[1, 2, 3])

        self._weighted_accuracy_train = Accuracy(task="multiclass", num_classes=num_classes, average='weighted')
        self._weighted_accuracy_val = Accuracy(task="multiclass", num_classes=num_classes, average='weighted')
        self._macro_accuracy_train = Accuracy(task="multiclass", num_classes=num_classes, average='macro')
        self._macro_accuracy_val = Accuracy(task="multiclass", num_classes=num_classes, average='macro')

        self._weighted_f1_train = F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self._weighted_f1_val = F1Score(task="multiclass", num_classes=num_classes, average='weighted')
        self._macro_f1_train = F1Score(task="multiclass", num_classes=num_classes, average='macro')
        self._macro_f1_val = F1Score(task="multiclass", num_classes=num_classes, average='macro')

        self._confusion_matrix_train = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize='true')
        self._confusion_matrix_val = ConfusionMatrix(task="multiclass", num_classes=num_classes, normalize='true')

    def training_step(self, batch, batch_nb):
        audio_signal, audio_signal_len, labels, labels_len = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss(logits=logits, labels=labels)

        return self._log_train_metrics(logits, labels, loss_value)

    def _log_train_metrics(self, logits, labels, loss_value):
        self.log('learning_rate', self._optimizer.param_groups[0]['lr'])
        self.log('global_step', self.trainer.global_step)

        predictions = torch.argmax(logits, dim=-1)
        predictions_list = predictions.tolist()
        labels_list = labels.tolist()

        print('pred/label: ', list(zip(predictions_list, labels_list)))

        self.log('train_loss', loss_value)

        # accuracies
        self._accuracy(logits=logits, labels=labels)
        topk_scores = self._accuracy.compute()
        self._accuracy.reset()

        for top_k, score in zip(self._accuracy.top_k, topk_scores):
            self.log('training_batch_accuracy_top_{}'.format(top_k), score)

        # confusion matrix
        self._confusion_matrix_train.update(preds=predictions, target=labels)
        self._weighted_accuracy_train(preds=predictions, target=labels)
        self._macro_accuracy_train(preds=predictions, target=labels)
        self._weighted_f1_train(preds=predictions, target=labels)
        self._macro_f1_train(preds=predictions, target=labels)

        return {
            'loss': loss_value,
        }

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.val_or_test_step(batch, dataloader_idx, 'val')

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.val_or_test_step(batch, dataloader_idx, 'test')

    def val_or_test_step(self, batch, dataloader_idx, prefix):
        audio_signal, audio_signal_len, labels, labels_len = batch

        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)
        loss_value = self.loss(logits=logits, labels=labels)

        return self._log_val_or_test_metrics(logits, labels, loss_value, dataloader_idx, prefix)

    def _log_val_or_test_metrics(self, logits, labels, loss_value, dataloader_idx, prefix):
        acc = self._accuracy(logits=logits, labels=labels)
        correct_counts, total_counts = self._accuracy.correct_counts_k, self._accuracy.total_counts_k

        logs = {
            f'{prefix}_loss': loss_value,
            f'{prefix}_correct_counts': correct_counts,
            f'{prefix}_total_counts': total_counts,
            f'{prefix}_acc': acc,
        }
        if type(self.trainer.val_dataloaders) == list and len(self.trainer.val_dataloaders) > 1:
            self.validation_step_outputs[dataloader_idx].append(logs)
        else:
            self.validation_step_outputs.append(logs)

        # confusion matrix
        predictions = torch.argmax(logits, dim=-1)
        self._confusion_matrix_val.update(preds=predictions, target=labels)
        self._weighted_accuracy_val(preds=predictions, target=labels)
        self._macro_accuracy_val(preds=predictions, target=labels)
        self._weighted_f1_val(preds=predictions, target=labels)
        self._macro_f1_val(preds=predictions, target=labels)

        return logs

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

        self.log_confusion_matrix_and_reset("train_confusion_matrix", self._confusion_matrix_train)
        self.log_metrics_and_reset('train', self._weighted_accuracy_train, self._macro_accuracy_train, self._weighted_f1_train, self._macro_f1_train)

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        logs = super().multi_validation_epoch_end(outputs, dataloader_idx)

        self.log_confusion_matrix_and_reset("val_confusion_matrix", self._confusion_matrix_val)
        self.log_metrics_and_reset('val', self._weighted_accuracy_val, self._macro_accuracy_val, self._weighted_f1_val, self._macro_f1_val)

        return logs

    def multi_test_epoch_end(self, outputs, dataloader_idx: int = 0):
        logs = super().multi_test_epoch_end(outputs, dataloader_idx)

        self.log_confusion_matrix_and_reset("test_confusion_matrix", self._confusion_matrix_val)
        self.log_metrics_and_reset('test', self._weighted_accuracy_val, self._macro_accuracy_val, self._weighted_f1_val, self._macro_f1_val)

        self.log_dict(logs)

        return logs

    def log_confusion_matrix_and_reset(self, name, confusion_matrix):
        computed_confusion = 100 * confusion_matrix.compute().detach().cpu().numpy()
        confusion_matrix.reset()

        image = self._render_confusion_matrix(computed_confusion)
        self.logger.experiment.add_image(name, image, global_step=self.trainer.global_step, dataformats='HWC')
        return computed_confusion, image

    def _render_confusion_matrix(self, confusion_matrix):
        class_names = self.cfg.labels

        figure, ax = plt.subplots(figsize=(6, 5))
        cax = ax.matshow(confusion_matrix, cmap=plt.cm.Blues)

        # Add colorbar
        plt.colorbar(cax)

        # Set ticks and labels
        ax.set_xticks(np.arange(len(class_names)))
        ax.set_yticks(np.arange(len(class_names)))
        ax.set_xticklabels(class_names)
        ax.set_yticklabels(class_names)

        # Rotate the tick labels for better readability
        plt.xticks(rotation=45)

        # Label axes
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')

        # Loop over data dimensions and create text annotations.
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                ax.text(j, i, f'{confusion_matrix[i, j]:.1f}', ha="center", va="center", color="white" if confusion_matrix[i, j] > confusion_matrix.max() / 2. else "black")

        plt.tight_layout()

        if self.confusion_matrix_save_path is not None:
            plt.savefig(self.confusion_matrix_save_path)
            plt.show()

        figure.canvas.draw()
        rgba = np.asarray(figure.canvas.buffer_rgba())
        return rgba

    def log_metrics_and_reset(self, prefix, weighted_accuracy, macro_accuracy, weighted_f1, macro_f1):
        self.log_metric_and_reset(f'{prefix}_weighted_accuracy', weighted_accuracy)
        self.log_metric_and_reset(f'{prefix}_macro_accuracy', macro_accuracy)
        self.log_metric_and_reset(f'{prefix}_weighted_f1', weighted_f1)
        self.log_metric_and_reset(f'{prefix}_macro_f1', macro_f1)

    def log_metric_and_reset(self, name, metric):
        weighted_accuracy = metric.compute()
        metric.reset()

        self.log(name, weighted_accuracy, sync_dist=True)
