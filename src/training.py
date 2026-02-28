import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        logits, _ = model(images, labels)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100. * correct / total:.1f}%"})

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            logits, _ = model(images, labels)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100. * correct / total:.1f}%"})

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def extract_embeddings(model, loader, device):
    model.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Embedding", leave=False):
            images = images.to(device)
            emb = model.get_embeddings(images).cpu().numpy()
            embeddings_list.append(emb)
            labels_list.append(labels.numpy())

    embeddings = np.vstack(embeddings_list)
    labels = np.concatenate(labels_list)
    return embeddings, labels


def extract_head_embeddings(model, loader, device):
    model.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for features, labels in tqdm(loader, desc="Embedding", leave=False):
            features = features.to(device)
            emb = model.get_embeddings(features).cpu().numpy()
            embeddings_list.append(emb)
            labels_list.append(labels.numpy())

    embeddings = np.vstack(embeddings_list)
    labels = np.concatenate(labels_list)
    return embeddings, labels


def compute_validation_map(model, loader, device):
    embeddings, labels = extract_embeddings(model, loader, device)
    return compute_map_from_embeddings(embeddings, labels)


def compute_validation_map_from_embeddings(model, loader, device):
    embeddings, labels = extract_head_embeddings(model, loader, device)
    return compute_map_from_embeddings(embeddings, labels)


def compute_map_from_embeddings(embeddings, labels):
    sim_matrix = cosine_similarity(embeddings)
    np.fill_diagonal(sim_matrix, -1)

    query_aps = {}
    labels = np.asarray(labels)

    for query_idx in range(len(labels)):
        query_label = labels[query_idx]
        similarities = sim_matrix[query_idx]

        is_match = (labels == query_label).astype(int)
        is_match[query_idx] = 0

        sorted_indices = np.argsort(-similarities)
        sorted_matches = is_match[sorted_indices]

        n_positives = sorted_matches.sum()
        if n_positives == 0:
            continue

        cumsum = np.cumsum(sorted_matches)
        precision_at_k = cumsum / np.arange(1, len(sorted_matches) + 1)
        ap = np.sum(precision_at_k * sorted_matches) / n_positives
        query_aps[query_idx] = (query_label, ap)

    identity_aps = {}
    for _, (label, ap) in query_aps.items():
        identity_aps.setdefault(label, []).append(ap)

    identity_mean_aps = [np.mean(aps) for aps in identity_aps.values()]
    return float(np.mean(identity_mean_aps)) if identity_mean_aps else 0.0


def train_epoch_embeddings(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for embeddings, labels in pbar:
        embeddings, labels = embeddings.to(device), labels.to(device)

        logits, _ = model(embeddings, labels)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100. * correct / total:.1f}%"})

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def validate_epoch_embeddings(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for embeddings, labels in pbar:
            embeddings, labels = embeddings.to(device), labels.to(device)
            logits, _ = model(embeddings, labels)
            loss = criterion(logits, labels)

            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            pbar.set_postfix({"loss": f"{loss.item():.4f}", "acc": f"{100. * correct / total:.1f}%"})

    avg_loss = total_loss / len(loader)
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def fit(
    model,
    train_loader,
    val_loader,
    config,
    device,
    criterion,
    optimizer,
    scheduler,
    label_encoder=None,
    wandb_run=None,
    checkpoint_name="best_model.pth",
):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_map": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_map = 0.0
    patience_counter = 0
    best_epoch = 0

    print(f"Starting training for {config['num_epochs']} epochs...")
    print("=" * 70)

    for epoch in range(config["num_epochs"]):
        print()
        print(f"Epoch {epoch+1}/{config['num_epochs']}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_map = compute_validation_map(model, val_loader, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_map"].append(val_map)
        history["lr"].append(current_lr)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_map": val_map,
                    "learning_rate": current_lr,
                }
            )

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.1f}%")
        print(f"  Val mAP:    {val_map:.4f} | LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_map = val_map
            best_epoch = epoch + 1
            patience_counter = 0

            checkpoint_path = config["checkpoint_dir"] / checkpoint_name
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_map": val_map,
                    "config": {k: str(v) if hasattr(v, "__fspath__") else v for k, v in config.items()},
                    "label_encoder_classes": label_encoder.classes_.tolist() if label_encoder else None,
                    "num_classes": len(label_encoder.classes_) if label_encoder else None,
                },
                checkpoint_path,
            )

            print("  [New best model saved]")
        else:
            patience_counter += 1
            print(f"  No improvement. Patience: {patience_counter}/{config['patience']}")

        if patience_counter >= config["patience"]:
            print()
            print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print()
    print("=" * 70)
    print("Training complete!")
    print(f"Best epoch: {best_epoch} (Val Loss: {best_val_loss:.4f}, Val mAP: {best_map:.4f})")

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_map": best_map,
        "best_epoch": best_epoch,
        "epochs_ran": len(history["train_loss"]),
    }


def fit_embeddings(
    model,
    train_loader,
    val_loader,
    config,
    device,
    criterion,
    optimizer,
    scheduler,
    label_encoder=None,
    wandb_run=None,
    checkpoint_name="best_model.pth",
):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_map": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_map = 0.0
    patience_counter = 0
    best_epoch = 0

    print(f"Starting training for {config['num_epochs']} epochs...")
    print("=" * 70)

    for epoch in range(config["num_epochs"]):
        # print()
        # print(f"Epoch {epoch+1}/{config['num_epochs']}")

        train_loss, train_acc = train_epoch_embeddings(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch_embeddings(model, val_loader, criterion, device)
        val_map = compute_validation_map_from_embeddings(model, val_loader, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_map"].append(val_map)
        history["lr"].append(current_lr)

        if wandb_run is not None:
            wandb_run.log(
                {
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "val_map": val_map,
                    "learning_rate": current_lr,
                }
            )

        # print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        # print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.1f}%")
        # print(f"  Val mAP:    {val_map:.4f} | LR: {current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_map = val_map
            best_epoch = epoch + 1
            patience_counter = 0

            checkpoint_path = config["checkpoint_dir"] / checkpoint_name
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_map": val_map,
                    "config": {k: str(v) if hasattr(v, "__fspath__") else v for k, v in config.items()},
                    "label_encoder_classes": label_encoder.classes_.tolist() if label_encoder else None,
                    "num_classes": len(label_encoder.classes_) if label_encoder else None,
                },
                checkpoint_path,
            )

            # print("  [New best model saved]")
        else:
            patience_counter += 1
            # print(f"  No improvement. Patience: {patience_counter}/{config['patience']}")

        if patience_counter >= config["patience"]:
            # print()
            # print(f"Early stopping triggered after {epoch+1} epochs")
            break

    print()
    print("=" * 70)
    print("Training complete!")
    print(f"Best epoch: {best_epoch} (Val Loss: {best_val_loss:.4f}, Val mAP: {best_map:.4f})")

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_map": best_map,
        "best_epoch": best_epoch,
        "epochs_ran": len(history["train_loss"]),
    }
