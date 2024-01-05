import argparse
import glob
import importlib
import logging
import os.path
import shutil

import pandas as pd
from pytorch_lightning import Trainer


def load_class(class_name):
    module = importlib.import_module(f'nemo_models.ser_nemo_models.{class_name}')
    return getattr(module, class_name)


def get_best_model(model_directory):
    checkpoint_path_pattern = os.path.join(model_directory, 'checkpoints', '*.ckpt')
    all_checkpoints = glob.glob(checkpoint_path_pattern)
    logging.info(f'Found {all_checkpoints}')

    checkpoints_with_val_scores = {checkpoint.split("=")[1].split("-")[0]: checkpoint for checkpoint in all_checkpoints}
    max_value = max(checkpoints_with_val_scores.keys())

    max_checkpoint = checkpoints_with_val_scores[max_value]
    logging.info(f'Found checkpoint {max_checkpoint} with highest val score {max_value}')

    return max_checkpoint


def make_data_frame(train_results, val_results, test_results):
    train_results['dataset'] = 'train'
    val_results['dataset'] = 'val'
    test_results['dataset'] = 'test'

    df = pd.DataFrame([train_results, val_results, test_results])
    cols = df.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    return df[cols]


def evaluate_model(model_base_directory: str, model_directory: str, model_type: str, results_base_directory: str):
    model_path = get_best_model(os.path.join(model_base_directory, model_directory))

    logging.info(f"Loading model from checkpoint: ${model_path}...")

    model_class = load_class(model_type)
    model = model_class.load_from_checkpoint(model_path)

    logging.info("Loading Dataloaders...")
    model.setup_test_data(model.cfg.test_ds)
    logging.info(f"Test Dataset Data: {model.cfg.test_ds.manifest_filepath}")
    model.setup_validation_data(model.cfg.validation_ds)
    logging.info(f"Validation Dataset Data: {model.cfg.validation_ds.manifest_filepath}")
    model.setup_training_data(model.cfg.train_ds)
    logging.info(f"Training Dataset Data: {model.cfg.train_ds.manifest_filepath}")

    trainer = Trainer(devices=1)

    logging.info(f'Evaluating model: {model_path}')

    logging.info("Running Test Dataset...")
    model.confusion_matrix_save_path = os.path.join(results_base_directory, 'confusion_matrix_test.png')
    test_results = trainer.validate(model, dataloaders=model.test_dataloader(), verbose=True)[0]
    logging.info(f'Test Results: {test_results}')

    logging.info("Running Validation Dataset...")
    model.confusion_matrix_save_path = os.path.join(results_base_directory, 'confusion_matrix_val.png')
    val_results = trainer.validate(model, dataloaders=model.val_dataloader(), verbose=True)[0]
    logging.info(f'Val Results: {val_results}')

    logging.info("Running Training Dataset...")
    model.confusion_matrix_save_path = os.path.join(results_base_directory, 'confusion_matrix_train.png')
    train_results = trainer.validate(model, dataloaders=model.train_dataloader(), verbose=True)[0]
    logging.info('Train Results: {train_results}')

    return make_data_frame(train_results, val_results, test_results)


def configure_logging(file_path):
    logging.basicConfig(level=logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(logging.INFO)

    logger = logging.getLogger()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def main(model_base_directory: str, model_directory: str, model_type: str, results_base_directory: str):
    # set up & clean results directory
    model_results_directory = os.path.join(results_base_directory, str(model_directory.replace("/", "_")))
    if os.path.exists(model_results_directory):  # delete previous eval if it already exists
        shutil.rmtree(model_results_directory)

    os.mkdir(model_results_directory)

    # set up logging
    configure_logging(os.path.join(model_results_directory, 'evaluate_model_log.txt'))

    # evaluate the model
    dataframe = evaluate_model(model_base_directory, model_directory, model_type, model_results_directory)

    # persist results as csv
    # dataframe.to_csv(os.path.join(model_results_directory, 'results.csv'), index=False)
    dataframe.to_excel(os.path.join(model_results_directory, 'results.xlsx'), index=False, engine='openpyxl')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-directory", required=True, default=None, type=str)
    parser.add_argument("--model-type", required=True, default=None, type=str,
                        choices=['ConformerClassifierModel', 'ConformerKeywordsPredictingClassifierModel', 'ConformerKeywordsClassifierModel'])
    parser.add_argument("--model-base-directory", default='/master-thesis/data/experiments', type=str)
    parser.add_argument("--results-base-directory", default='/master-thesis/data/results', type=str)
    args = parser.parse_args()

    main(args.model_base_directory, args.model_directory, args.model_type, args.results_base_directory)
