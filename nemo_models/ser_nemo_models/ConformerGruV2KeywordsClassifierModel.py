import torch
from omegaconf import DictConfig
from pytorch_lightning import Trainer
from torch.nn import BCEWithLogitsLoss
from torchmetrics import Accuracy, F1Score, MetricCollection

from nemo_models.ser_nemo_models.EmotionKeywordsClassiferModel import EmotionKeywordsClassifierModel


class ConformerGruV2KeywordsClassifierModel(EmotionKeywordsClassifierModel):

    def __init__(self, cfg: DictConfig, trainer: Trainer = None):
        super().__init__(cfg, trainer)

        self._

        if cfg.decoder.num_keyword_classes is not None:
            num_keyword_classes = len(cfg.keyword_labels)
            self._keywords_weighted_accuracy_train = Accuracy(task="multilabel", num_labels=num_keyword_classes, average='weighted')
            self._keywords_weighted_accuracy_val = Accuracy(task="multilabel", num_labels=num_keyword_classes, average='weighted')
            self._keywords_macro_accuracy_train = Accuracy(task="multilabel", num_labels=num_keyword_classes, average='macro')
            self._keywords_macro_accuracy_val = Accuracy(task="multilabel", num_labels=num_keyword_classes, average='macro')

            self._keywords_weighted_f1_train = F1Score(task="multilabel", num_labels=num_keyword_classes, average='weighted')
            self._keywords_weighted_f1_val = F1Score(task="multilabel", num_labels=num_keyword_classes, average='weighted')
            self._keywords_macro_f1_train = F1Score(task="multilabel", num_labels=num_keyword_classes, average='macro')
            self._keywords_macro_f1_val = F1Score(task="multilabel", num_labels=num_keyword_classes, average='macro')

    def _setup_loss(self):
        self._keywords_loss = BCEWithLogitsLoss()
        return super()._setup_loss()

    def training_step(self, batch, batch_nb):
        audio_signal, audio_signal_len, labels, labels_len, keywords = batch
        emotion_logits, keyword_logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)

        emotion_loss_value = self.loss(logits=emotion_logits, labels=labels)
        keywords_loss_value = self._keywords_loss(keyword_logits, keywords)

        loss_value = emotion_loss_value + keywords_loss_value

        self.log('train_emotion_loss', emotion_loss_value)
        self.log('train_keywords_loss', keywords_loss_value)
        self._keywords_weighted_accuracy_train(preds=keyword_logits, target=keywords)
        self._keywords_macro_accuracy_train(preds=keyword_logits, target=keywords)
        self._keywords_weighted_f1_train(preds=keyword_logits, target=keywords)
        self._keywords_macro_f1_train(preds=keyword_logits, target=keywords)

        return self._log_train_metrics(emotion_logits, labels, loss_value)

    def val_or_test_step(self, batch, dataloader_idx, prefix):
        audio_signal, audio_signal_len, labels, labels_len, keywords = batch
        emotion_logits, keyword_logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len)

        emotion_loss_value = self.loss(logits=emotion_logits, labels=labels)
        keywords_loss_value = self._keywords_loss(keyword_logits, keywords)

        loss_value = emotion_loss_value + keywords_loss_value

        self._keywords_weighted_accuracy_val(preds=keyword_logits, target=keywords)
        self._keywords_macro_accuracy_val(preds=keyword_logits, target=keywords)
        self._keywords_weighted_f1_val(preds=keyword_logits, target=keywords)
        self._keywords_macro_f1_val(preds=keyword_logits, target=keywords)

        logs = self._log_val_or_test_metrics(emotion_logits, labels, loss_value, dataloader_idx, prefix)
        logs[f'{prefix}_emotion_loss'] = emotion_loss_value
        logs[f'{prefix}_keywords_loss'] = keywords_loss_value

        return logs

    def on_train_epoch_end(self):
        super().on_train_epoch_end()

        self.log_metrics_and_reset('train_keywords', self._keywords_weighted_accuracy_train, self._keywords_macro_accuracy_train, self._keywords_weighted_f1_train,
                                   self._keywords_macro_f1_train)

    def multi_validation_epoch_end(self, outputs, dataloader_idx: int = 0):
        logs = super().multi_validation_epoch_end(outputs, dataloader_idx)

        self.log_metrics_and_reset('val_keywords', self._keywords_weighted_accuracy_val, self._keywords_macro_accuracy_val, self._keywords_weighted_f1_val,
                                   self._keywords_macro_f1_val)

        val_emotion_loss_mean = torch.stack([x['val_emotion_loss'] for x in outputs]).mean()
        val_keywords_loss_mean = torch.stack([x['val_keywords_loss'] for x in outputs]).mean()

        self.log('val_emotion_loss', val_emotion_loss_mean, sync_dist=True)
        self.log('val_keywords_loss', val_keywords_loss_mean, sync_dist=True)

        return logs
