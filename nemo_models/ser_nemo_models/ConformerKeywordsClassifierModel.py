from nemo_models.ser_nemo_models.EmotionKeywordsClassiferModel import EmotionKeywordsClassifierModel


class ConformerKeywordsClassifierModel(EmotionKeywordsClassifierModel):

    def training_step(self, batch, batch_nb):
        audio_signal, audio_signal_len, labels, labels_len, keywords = batch
        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len, keywords=keywords)
        loss_value = self.loss(logits=logits, labels=labels)

        return self._log_train_metrics(logits, labels, loss_value)

    def val_or_test_step(self, batch, dataloader_idx, prefix):
        audio_signal, audio_signal_len, labels, labels_len, keywords = batch

        logits = self.forward(input_signal=audio_signal, input_signal_length=audio_signal_len, keywords=keywords)
        loss_value = self.loss(logits=logits, labels=labels)

        return self._log_val_or_test_metrics(logits, labels, loss_value, dataloader_idx, prefix)
