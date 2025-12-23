import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

from config.vars import (
    set_seed,
    DEVICE,
    EXPERIMENTS,
    FIGS_DIR,
)
from data.data import get_datasets_and_loaders
from models.models import SimpleCNNGAP, SimpleCNNFC, SimpleTransformerClassifier, HybridConvTransformer
from training.train import run_experiment
from utils.viz import visualize_dataset_samples, plot_losses_accs, plot_sample_predictions, plot_runtimes


def build_model(model_type: str):
    if model_type == "cnn_gap":
        return SimpleCNNGAP()
    if model_type == "cnn_fc":
        return SimpleCNNFC()
    if model_type == "transformer":
        return SimpleTransformerClassifier()
    if model_type == "hybrid":
        return HybridConvTransformer()
    raise ValueError(f"Unknown model_type: {model_type}")


def main():
    set_seed()

    # data
    train_ds, test_ds, train_loader, test_loader = get_datasets_and_loaders()

    # visualize data
    data_viz_path = os.path.join(FIGS_DIR, "data.png")
    visualize_dataset_samples(dataset=train_ds, num_samples=16, out_path=data_viz_path)

    histories = {}
    trained_models = {}

    # train all experiments
    for exp_name, cfg in EXPERIMENTS.items():
        model_type = cfg["model_type"]
        num_epochs = cfg["num_epochs"]

        model = build_model(model_type)
        history = run_experiment(
            exp_name=exp_name,
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=num_epochs,
            device=DEVICE,
        )

        histories[exp_name] = history
        trained_models[exp_name] = model  # keep in memory for viz

    # visualizations
    losses_path = os.path.join(FIGS_DIR, "losses_accs.png")
    preds_path = os.path.join(FIGS_DIR, "predictions.png")
    runtimes_path = os.path.join(FIGS_DIR, "runtimes.png")

    plot_losses_accs(histories, losses_path)
    plot_runtimes(histories, runtimes_path)
    plot_sample_predictions(trained_models, test_ds, preds_path, device=DEVICE)


if __name__ == "__main__":
    main()
