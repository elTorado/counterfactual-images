#!/bin/bash
# Break on any error
set -e

DATASET_DIR=/home/user/heizmann/data/


# Hyperparameters
GAN_EPOCHS=30
CLASSIFIER_EPOCHS=3
CF_COUNT=50
GENERATOR_MODE=open_set


# Train the intial generative model (E+G+D) and the initial classifier (C_K)
python generativeopenset/train_gan.py --epochs $GAN_EPOCHS

# Baseline: Evaluate the standard classifier (C_k+1)
python src/evaluate_classifier.py --result_dir . --mode baseline
python src/evaluate_classifier.py --result_dir . --mode weibull

cp checkpoints/classifier_k_epoch_00${GAN_EPOCHS}.pth checkpoints/classifier_kplusone_epoch_00${GAN_EPOCHS}.pth

# Generate a number of counterfactual images (in the K+2 by K+2 square grid format)
python src/generate_${GENERATOR_MODE}.py --result_dir . --count $CF_COUNT

# Automatically label the rightmost column in each grid (ignore the others)
python src/auto_label.py --output_filename generated_images_${GENERATOR_MODE}.dataset

# Train a new classifier, now using the aux_dataset containing the counterfactuals
python src/train_classifier.py --epochs $CLASSIFIER_EPOCHS --aux_dataset generated_images_${GENERATOR_MODE}.dataset

# Evaluate the C_K+1 classifier, trained with the augmented data
python src/evaluate_classifier.py --result_dir . --mode fuxin

./print_results.sh
