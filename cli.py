"""
Command-Line Interface (CLI) for NeuroVision Deep Learning Suite.
Execute model training, inference, explainability (Grad-CAM/Saliency), and adversarial attacks.
"""

import os
import argparse
import sys
import torch
import numpy as np
from PIL import Image

from src.models import ConvNetClassifier, NeuroResNet, ConvVAE
from src.data import get_data_loaders, get_preset_samples
from src.training import ModelTrainer
from src.explainability import GradCAM, SaliencyMap, FeatureMapExtractor
from src.adversarial import FGSMAttack
from src.utils import (
    preprocess_drawn_image,
    plot_prediction_bars,
    plot_gradcam_breakdown,
    plot_saliency_breakdown,
    plot_adversarial_breakdown,
)


def load_model(model_type: str = "cnn", weights_path: str = None, device: str = "cpu"):
    if model_type.lower() == "resnet":
        model = NeuroResNet()
        default_path = "./weights/best_resnet.pt"
    else:
        model = ConvNetClassifier()
        default_path = "./weights/best_cnn.pt"

    path_to_use = weights_path or default_path
    if os.path.exists(path_to_use):
        checkpoint = torch.load(path_to_use, map_location=device)
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print(f"[*] Loaded {model_type.upper()} weights from: {path_to_use}")
    else:
        print(f"[!] Warning: Weights file {path_to_use} not found. Using untrained weights.")

    model.to(device)
    model.eval()
    return model


def cmd_predict(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.weights, device=device)

    # Load image
    if not os.path.exists(args.image):
        print(f"[Error] Image file not found: {args.image}")
        sys.exit(1)

    pil_img = Image.open(args.image)
    tensor, _ = preprocess_drawn_image(pil_img)
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

    top_idx = int(np.argmax(probs))
    confidence = float(probs[top_idx])

    print("\n" + "=" * 45)
    print("NeuroVision Prediction Result")
    print("=" * 45)
    print(f"Predicted Class: Digit {top_idx}")
    print(f"Confidence:      {confidence * 100:.2f}%")
    print("\nTop Candidates:")
    sorted_indices = np.argsort(probs)[::-1][:5]
    for rank, idx in enumerate(sorted_indices, 1):
        print(f"  {rank}. Digit {idx}: {probs[idx] * 100:.2f}%")
    print("=" * 45)


def cmd_explain(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.weights, device=device)

    if not os.path.exists(args.image):
        print(f"[Error] Image file not found: {args.image}")
        sys.exit(1)

    pil_img = Image.open(args.image)
    tensor, norm_img = preprocess_drawn_image(pil_img)
    tensor = tensor.to(device)

    method = args.method.lower()
    out_file = args.output or f"explanation_{method}.png"

    if method == "gradcam":
        cam = GradCAM(model)
        heatmap, pred_class, conf = cam.generate(tensor, target_class=args.target_class)
        overlay = cam.overlay_on_image(norm_img, heatmap)
        fig = plot_gradcam_breakdown(norm_img, heatmap, overlay, f"Digit {pred_class}", conf)
        fig.savefig(out_file, bbox_inches="tight")
        print(f"[*] Grad-CAM explanation saved to: {out_file}")
    elif method in ["saliency", "smoothgrad"]:
        sal = SaliencyMap(model)
        is_smooth = (method == "smoothgrad")
        saliency_map, pred_class, conf = sal.generate(
            tensor, target_class=args.target_class, smoothgrad=is_smooth
        )
        vanilla_map, _, _ = sal.generate(tensor, target_class=args.target_class, smoothgrad=False)
        fig = plot_saliency_breakdown(norm_img, vanilla_map, saliency_map, f"Digit {pred_class}")
        fig.savefig(out_file, bbox_inches="tight")
        print(f"[*] {method.capitalize()} explanation saved to: {out_file}")
    else:
        print(f"[Error] Unsupported explanation method: {method}")


def cmd_attack(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.weights, device=device)

    if not os.path.exists(args.image):
        print(f"[Error] Image file not found: {args.image}")
        sys.exit(1)

    pil_img = Image.open(args.image)
    tensor, _ = preprocess_drawn_image(pil_img)
    tensor = tensor.to(device)

    attacker = FGSMAttack(model)
    results = attacker.attack(tensor, epsilon=args.epsilon, target_class=args.target_class)

    print("\n" + "=" * 50)
    print("FGSM Adversarial Attack Evaluation")
    print("=" * 50)
    print(f"Perturbation Magnitude (epsilon): {args.epsilon}")
    print(f"Clean Prediction:            Digit {results['orig_prediction']} ({results['orig_confidence']*100:.2f}%)")
    print(f"Adversarial Prediction:      Digit {results['adv_prediction']} ({results['adv_confidence']*100:.2f}%)")
    print(f"Model Fooled:                {'YES' if results['is_fooled'] else 'NO'}")
    print(f"L_inf Max Distortion:        {results['l_inf_distortion']:.4f}")
    print(f"L2 Euclidean Distortion:     {results['l_2_distortion']:.4f}")
    print("=" * 50)

    out_file = args.output or "adversarial_result.png"
    fig = plot_adversarial_breakdown(results)
    fig.savefig(out_file, bbox_inches="tight")
    print(f"[*] Attack comparison plot saved to: {out_file}")


def cmd_train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training model: {args.model.upper()} on device {device}")

    train_loader, val_loader, _ = get_data_loaders(
        data_dir=args.data_dir, batch_size=args.batch_size
    )

    if args.model.lower() == "resnet":
        model = NeuroResNet()
    else:
        model = ConvNetClassifier()

    trainer = ModelTrainer(
        model, device=device, learning_rate=args.lr, optimizer_type=args.optimizer
    )
    save_path = args.save_path or f"./weights/best_{args.model.lower()}.pt"

    def progress(epoch, total, stats):
        print(
            f"Epoch [{epoch}/{total}] "
            f"Train Loss: {stats['train_loss']:.4f} | "
            f"Val Loss: {stats['val_loss']:.4f} | "
            f"Val Acc: {stats['val_acc']*100:.2f}% | "
            f"Val F1: {stats['val_f1']:.4f} ({stats['elapsed']:.1f}s)"
        )

    results = trainer.fit(
        train_loader, val_loader, epochs=args.epochs, save_path=save_path, callback=progress
    )
    print(f"\n[*] Training Complete. Peak Validation Accuracy: {results['best_val_acc']*100:.2f}%")
    print(f"[*] Checkpoint saved at: {save_path}")


def cmd_evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(args.model, args.weights, device=device)
    _, _, test_loader = get_data_loaders(data_dir=args.data_dir, batch_size=128)

    trainer = ModelTrainer(model, device=device)
    loss, metrics = trainer.evaluate(test_loader)

    print("\n" + "=" * 55)
    print(f"Evaluation Metrics on Test Partition ({metrics['total_samples']} samples)")
    print("=" * 55)
    print(f"Test Loss:        {loss:.4f}")
    print(f"Top-1 Accuracy:   {metrics['accuracy'] * 100:.2f}%")
    print(f"Macro F1 Score:   {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:      {metrics['weighted_f1']:.4f}")
    print("-" * 55)
    print("Per-Class Breakdown:")
    for c in range(10):
        cls_stat = metrics["per_class"][c]
        print(
            f"  Class {c}: Precision={cls_stat['precision']*100:.1f}% | "
            f"Recall={cls_stat['recall']*100:.1f}% | "
            f"F1={cls_stat['f1']:.3f} | Support={cls_stat['support']}"
        )
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="NeuroVision CLI: Deep Learning Computer Vision, Explainability & Robustness Studio"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Predict
    p_pred = subparsers.add_parser("predict", help="Predict class of an image")
    p_pred.add_argument("--image", required=True, help="Path to input image")
    p_pred.add_argument("--model", default="cnn", choices=["cnn", "resnet"], help="Model architecture")
    p_pred.add_argument("--weights", default=None, help="Custom weights file")
    p_pred.set_defaults(func=cmd_predict)

    # Explain
    p_exp = subparsers.add_parser("explain", help="Generate Explainable AI attribution map")
    p_exp.add_argument("--image", required=True, help="Path to input image")
    p_exp.add_argument("--method", default="gradcam", choices=["gradcam", "saliency", "smoothgrad"])
    p_exp.add_argument("--model", default="cnn", choices=["cnn", "resnet"])
    p_exp.add_argument("--weights", default=None)
    p_exp.add_argument("--target-class", type=int, default=None)
    p_exp.add_argument("--output", default=None, help="Output image file path")
    p_exp.set_defaults(func=cmd_explain)

    # Attack
    p_atk = subparsers.add_parser("attack", help="Generate FGSM adversarial perturbation")
    p_atk.add_argument("--image", required=True, help="Path to input image")
    p_atk.add_argument("--epsilon", type=float, default=0.15, help="Perturbation magnitude")
    p_atk.add_argument("--target-class", type=int, default=None)
    p_atk.add_argument("--model", default="cnn", choices=["cnn", "resnet"])
    p_atk.add_argument("--weights", default=None)
    p_atk.add_argument("--output", default=None, help="Output plot path")
    p_atk.set_defaults(func=cmd_attack)

    # Train
    p_trn = subparsers.add_parser("train", help="Train a neural network model")
    p_trn.add_argument("--model", default="cnn", choices=["cnn", "resnet"])
    p_trn.add_argument("--epochs", type=int, default=5)
    p_trn.add_argument("--lr", type=float, default=1e-3)
    p_trn.add_argument("--batch-size", type=int, default=64)
    p_trn.add_argument("--optimizer", default="adam", choices=["adam", "adamw", "sgd"])
    p_trn.add_argument("--data-dir", default="./data_cache")
    p_trn.add_argument("--save-path", default=None)
    p_trn.set_defaults(func=cmd_train)

    # Evaluate
    p_eval = subparsers.add_parser("evaluate", help="Evaluate model performance on test set")
    p_eval.add_argument("--model", default="cnn", choices=["cnn", "resnet"])
    p_eval.add_argument("--weights", default=None)
    p_eval.add_argument("--data-dir", default="./data_cache")
    p_eval.set_defaults(func=cmd_evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
