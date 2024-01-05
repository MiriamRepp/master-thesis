import argparse

import pytorch_lightning as pl
import torch
from nemo.utils.exp_manager import exp_manager, ExpManagerConfig, CallbackParams
from omegaconf import OmegaConf

from nemo_models.ser_nemo_models.scripts.evaluate_model import load_class


def train_model(pre_trained_model_name: str, dataset_name: str, model_type: str, config_name: str, experiment_name: str, freeze_encoder: bool):
    super_experiment_name = f"{dataset_name}/{config_name}"

    model_config_path = f'/master-thesis/masterarbeit/nemo_models/ser_nemo_models/config/{dataset_name}/{config_name}.yaml'

    loader_config_yaml = f"""
    init_from_pretrained_model:
      model0:
        name: {pre_trained_model_name}
        include: [ "encoder" ]
    """

    pretrained_model_load_config = OmegaConf.create(loader_config_yaml)

    model_config = OmegaConf.load(model_config_path)

    trainer = pl.Trainer(**model_config.trainer)

    model_class = load_class(model_type)
    model = model_class(model_config.model, trainer)
    model.maybe_init_from_pretrained_checkpoint(pretrained_model_load_config)

    def enable_bn_se(m):
        if type(m) == torch.nn.BatchNorm1d:
            m.train()
            for param in m.parameters():
                param.requires_grad_(True)

        if 'SqueezeExcite' in type(m).__name__:
            m.train()
            for param in m.parameters():
                param.requires_grad_(True)

    if freeze_encoder:
        model.encoder.freeze()
        model.encoder.apply(enable_bn_se)
        print("Model encoder has been frozen")
    else:
        model.encoder.unfreeze()
        print("Model encoder has been un-frozen")

    # setup export manager
    exp_manager_config = ExpManagerConfig(
        exp_dir=f'../data/experiments/{super_experiment_name}',
        name=experiment_name,
        create_tensorboard_logger=True,
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        checkpoint_callback_params=CallbackParams(
            always_save_nemo=False,
            monitor='val_weighted_f1',
            mode='max',
            save_top_k=10,
        )
    )

    exp_manager(trainer, OmegaConf.structured(exp_manager_config))

    trainer.fit(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-trained-model-name", default='stt_en_fastconformer_ctc_large', type=str, choices=['stt_en_fastconformer_ctc_large', 'stt_en_conformer_ctc_small'],
                        help='Name of the pre-trained model to be utilized. Please note that the selected config needs to be compatible with this model.')
    parser.add_argument("--model-type", required=True, type=str,
                        choices=['ConformerClassifierModel', 'ConformerKeywordsClassifierModel', 'ConformerKeywordsPredictingClassifierModel'],
                        help='Class of the model to be used. Please note the model class needs to be compatible with the selected config.')
    parser.add_argument("--dataset-name", required=True, type=str, choices=['emoDB', 'meld', 'meld-balanced', 'meld-mixed', 'iemocap-6', 'iemocap-4'],
                        help='Name of the dataset to be used. This name is used to select the config directory.')
    parser.add_argument("--config-name", required=True, type=str,
                        help='The name of the config. E.g. \"stt_en_fastconformer_ctc_large_classification\"')
    parser.add_argument("--experiment-name", required=True, type=str,
                        help='The name of the experiment. We usually use a timestamp for this. This name is used to define the path where the model is saved.')
    parser.add_argument("--freeze-encoder", type=bool, default=True,
                        help='Freeze the encoder or not.')
    args = parser.parse_args()

    train_model(args.pre_trained_model_name, args.dataset_name, args.model_type, args.config_name, args.experiment_name, args.freeze_encoder)
