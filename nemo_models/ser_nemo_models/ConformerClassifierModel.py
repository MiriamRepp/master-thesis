from nemo_models.ser_nemo_models.ConformerEncDecClassificationModel import ConformerEncDecClassificationModel


class ConformerClassifierModel(ConformerEncDecClassificationModel):

    def forward(self, input_signal=None, input_signal_length=None, processed_signal=None, processed_signal_length=None, **decoder_kwargs):
        has_input_signal = input_signal is not None and input_signal_length is not None
        has_processed_signal = processed_signal is not None and processed_signal_length is not None

        if not (has_input_signal ^ has_processed_signal):
            raise ValueError(
                f"{self} Arguments ``input_signal`` and ``input_signal_length`` are mutually exclusive "
                " with ``processed_signal`` and ``processed_signal_length`` arguments."
            )

        if not has_processed_signal:
            processed_signal, processed_signal_length = self.preprocessor(input_signal=input_signal, length=input_signal_length)

        # Crop or pad is always applied
        if self.crop_or_pad is not None:
            processed_signal, processed_signal_length = self.crop_or_pad(input_signal=processed_signal, length=processed_signal_length)

        # Spec augment is not applied during evaluation/testing
        if self.spec_augmentation is not None and self.training:
            processed_signal = self.spec_augmentation(input_spec=processed_signal, length=processed_signal_length)

        encoded, encoded_len = self.encoder(audio_signal=processed_signal, length=processed_signal_length)
        logits = self.decoder(encoder_output=encoded, encoded_lengths=encoded_len, **decoder_kwargs)

        return logits
