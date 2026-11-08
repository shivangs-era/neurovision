"""
One-click training script to train and save pre-trained checkpoints for CNN, ResNet, and VAE.
Operates 100% locally and completes in under a minute on CPU.
"""

import os
import sys
import time
import torch

from src.models import ConvNetClassifier, NeuroResNet, ConvVAE
from src.data import get_data_loaders
from src.training import ModelTrainer, VAETrainer


def main():
    print("=" * 65)
    print("NeuroVision Deep Learning Suite: Training Default Models")
    print("=" * 65)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Compute Device: {device}")

    # Prepare data loaders (batch_size 128 for efficient CPU throughput)
    print("[*] Initializing DataLoaders from local cache...")
    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir="./data_cache", batch_size=128
    )
    print(f"    Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")

    os.makedirs("./weights", exist_ok=True)

    # 1. Train ConvNetClassifier
    print("\n[1/3] Training ConvNetClassifier...")
    cnn = ConvNetClassifier()
    print(f"      Architecture parameters: {cnn.count_parameters()['trainable']:,} trainable")
    cnn_trainer = ModelTrainer(cnn, device=device, learning_rate=1e-3, optimizer_type="adam")

    t0 = time.time()
    def cnn_cb(epoch, total_epochs, stats):
        print(
            f"      Epoch [{epoch}/{total_epochs}] "
            f"Train Loss: {stats['train_loss']:.4f} | "
            f"Val Loss: {stats['val_loss']:.4f} | "
            f"Val Acc: {stats['val_acc']*100:.2f}% | "
            f"Time: {stats['elapsed']:.1f}s"
        )

    cnn_results = cnn_trainer.fit(
        train_loader, val_loader, epochs=3, save_path="./weights/best_cnn.pt", callback=cnn_cb
    )
    print(f"      Best Val Accuracy: {cnn_results['best_val_acc']*100:.2f}% (Finished in {time.time()-t0:.1f}s)")

    # 2. Train NeuroResNet
    print("\n[2/3] Training NeuroResNet (Residual Network)...")
    resnet = NeuroResNet()
    print(f"      Architecture parameters: {resnet.count_parameters()['trainable']:,} trainable")
    resnet_trainer = ModelTrainer(resnet, device=device, learning_rate=1e-3, optimizer_type="adamw")

    t0 = time.time()
    def resnet_cb(epoch, total_epochs, stats):
        print(
            f"      Epoch [{epoch}/{total_epochs}] "
            f"Train Loss: {stats['train_loss']:.4f} | "
            f"Val Loss: {stats['val_loss']:.4f} | "
            f"Val Acc: {stats['val_acc']*100:.2f}% | "
            f"Time: {stats['elapsed']:.1f}s"
        )

    resnet_results = resnet_trainer.fit(
        train_loader, val_loader, epochs=3, save_path="./weights/best_resnet.pt", callback=resnet_cb
    )
    print(f"      Best Val Accuracy: {resnet_results['best_val_acc']*100:.2f}% (Finished in {time.time()-t0:.1f}s)")

    # 3. Train ConvVAE (Latent Space Generator)
    print("\n[3/3] Training ConvVAE (Variational Autoencoder)...")
    vae = ConvVAE(latent_dim=2)
    vae_trainer = VAETrainer(vae, device=device, learning_rate=1e-3, beta=1.0)

    t0 = time.time()
    def vae_cb(epoch, total_epochs, stats):
        print(
            f"      Epoch [{epoch}/{total_epochs}] "
            f"Total Loss: {stats['total_loss']:.1f} | "
            f"Recon: {stats['recon_loss']:.1f} | "
            f"KLD: {stats['kld']:.1f}"
        )

    vae_results = vae_trainer.fit(
        train_loader, epochs=3, save_path="./weights/best_vae.pt", callback=vae_cb
    )
    print(f"      Finished VAE training in {time.time()-t0:.1f}s")

    print("\n" + "=" * 65)
    print("All models trained and checkpoints saved successfully in ./weights!")
    print("=" * 65)


if __name__ == "__main__":
    main()
