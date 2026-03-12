import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_similarity
from tqdm.notebook import tqdm
from src.reranking import rerank_embeddings


def build_optimizer(model, config):
    head_lr = config.get("head_learning_rate", config.get("learning_rate", 1e-4))
    backbone_lr = config.get("backbone_learning_rate", head_lr * 0.1)
    weight_decay = config.get("weight_decay", 1e-4)

    if hasattr(model, "backbone"):
        backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]
        head_params = [
            p
            for name, p in model.named_parameters()
            if p.requires_grad and not name.startswith("backbone.")
        ]

        param_groups = []
        if backbone_params:
            param_groups.append({"params": backbone_params, "lr": backbone_lr})
        if head_params:
            param_groups.append({"params": head_params, "lr": head_lr})

        if not param_groups:
            raise ValueError("No trainable parameters found for optimizer.")

        return torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("No trainable parameters found for optimizer.")
    return torch.optim.AdamW(params, lr=head_lr, weight_decay=weight_decay)


def build_eval_score_matrix(embeddings, use_rerank=False, k1=20, k2=6, lambda_value=0.3):
    sim_matrix = cosine_similarity(embeddings)
    if not use_rerank:
        return sim_matrix

    final_dist = rerank_embeddings(
        embeddings,
        gallery_embeddings=embeddings,
        k1=k1,
        k2=k2,
        lambda_value=lambda_value,
    )
    return -final_dist


def get_rerank_config(config):
    rerank_config = config.get("rerank", {})
    return {
        "enabled": rerank_config.get("enabled", rerank_config.get("use_rerank", False)),
        "k1": rerank_config.get("k1", 20),
        "k2": rerank_config.get("k2", 6),
        "lambda_value": rerank_config.get("lambda_value", 0.3),
    }


def get_checkpoint_metric(val_map, val_map_rerank):
    if val_map_rerank is not None:
        return val_map_rerank, "val_map_rerank"
    return val_map, "val_map"


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


def compute_validation_map(model, loader, device, use_rerank=False, k1=20, k2=6, lambda_value=0.3):
    embeddings, labels = extract_embeddings(model, loader, device)
    return compute_map_from_embeddings(
        embeddings,
        labels,
        use_rerank=use_rerank,
        k1=k1,
        k2=k2,
        lambda_value=lambda_value,
    )


def compute_validation_map_from_embeddings(model, loader, device, use_rerank=False, k1=20, k2=6, lambda_value=0.3):
    embeddings, labels = extract_head_embeddings(model, loader, device)
    return compute_map_from_embeddings(
        embeddings,
        labels,
        use_rerank=use_rerank,
        k1=k1,
        k2=k2,
        lambda_value=lambda_value,
    )


def compute_map_from_embeddings(embeddings, labels, use_rerank=False, k1=20, k2=6, lambda_value=0.3):
    score_matrix = build_eval_score_matrix(
        embeddings,
        use_rerank=use_rerank,
        k1=k1,
        k2=k2,
        lambda_value=lambda_value,
    )
    np.fill_diagonal(score_matrix, -np.inf)

    stats = compute_ap_details_from_similarity(score_matrix, labels)
    return stats["map"]


def compute_ap_details_from_similarity(sim_matrix, labels):
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
        query_aps[query_idx] = (query_label, float(ap))

    identity_aps = {}
    for _, (label, ap) in query_aps.items():
        identity_aps.setdefault(int(label), []).append(float(ap))

    identity_mean_ap = {label: float(np.mean(aps)) for label, aps in identity_aps.items()}
    map_score = float(np.mean(list(identity_mean_ap.values()))) if identity_mean_ap else 0.0

    return {
        "query_aps": query_aps,
        "identity_aps": identity_aps,
        "identity_mean_ap": identity_mean_ap,
        "map": map_score,
    }


def compute_ap_details_from_embeddings(embeddings, labels, use_rerank=False, k1=20, k2=6, lambda_value=0.3):
    score_matrix = build_eval_score_matrix(
        embeddings,
        use_rerank=use_rerank,
        k1=k1,
        k2=k2,
        lambda_value=lambda_value,
    )
    np.fill_diagonal(score_matrix, -np.inf)
    return compute_ap_details_from_similarity(score_matrix, labels)


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
        "val_map_rerank": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_map = 0.0
    best_map_rerank = None
    best_checkpoint_metric = float("-inf")
    patience_counter = 0
    best_epoch = 0
    rerank_config = get_rerank_config(config)
    best_checkpoint_metric_name = "val_map_rerank" if rerank_config["enabled"] else "val_map"

    print(f"Starting training for {config['num_epochs']} epochs...")
    print("=" * 70)

    for epoch in range(config["num_epochs"]):
        print()
        print(f"Epoch {epoch+1}/{config['num_epochs']}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
        val_embeddings, val_labels = extract_embeddings(model, val_loader, device)
        val_map = compute_map_from_embeddings(val_embeddings, val_labels)
        val_map_rerank = None
        if rerank_config["enabled"]:
            val_map_rerank = compute_map_from_embeddings(
                val_embeddings,
                val_labels,
                use_rerank=True,
                k1=rerank_config["k1"],
                k2=rerank_config["k2"],
                lambda_value=rerank_config["lambda_value"],
            )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_map"].append(val_map)
        history["val_map_rerank"].append(val_map_rerank)
        history["lr"].append(current_lr)

        if wandb_run is not None:
            log_payload = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_map": val_map,
                "learning_rate": current_lr,
            }
            if val_map_rerank is not None:
                log_payload["val_map_rerank"] = val_map_rerank
            wandb_run.log(log_payload)

        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.1f}%")
        print(f"  Val mAP:    {val_map:.4f} | LR: {current_lr:.2e}")
        if val_map_rerank is not None:
            print(f"  Val mAP RR: {val_map_rerank:.4f} | k1={rerank_config['k1']} k2={rerank_config['k2']} lambda={rerank_config['lambda_value']}")

        checkpoint_metric, checkpoint_metric_name = get_checkpoint_metric(val_map, val_map_rerank)
        if checkpoint_metric > best_checkpoint_metric:
            best_checkpoint_metric = checkpoint_metric
            best_checkpoint_metric_name = checkpoint_metric_name
            best_val_loss = val_loss
            best_map = val_map
            best_map_rerank = val_map_rerank
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
                    "val_map_rerank": val_map_rerank,
                    "config": {k: str(v) if hasattr(v, "__fspath__") else v for k, v in config.items()},
                    "label_encoder_classes": label_encoder.classes_.tolist() if label_encoder else None,
                    "num_classes": len(label_encoder.classes_) if label_encoder else None,
                },
                checkpoint_path,
            )

            print(f"  [New best model saved by {checkpoint_metric_name}={checkpoint_metric:.4f}]")
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
    print(
        f"Best epoch: {best_epoch} "
        f"({best_checkpoint_metric_name}: {best_checkpoint_metric:.4f}, "
        f"Val Loss: {best_val_loss:.4f}, Val mAP: {best_map:.4f})"
    )

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_map": best_map,
        "best_map_rerank": best_map_rerank,
        "best_checkpoint_metric": best_checkpoint_metric,
        "best_checkpoint_metric_name": best_checkpoint_metric_name,
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
    restore_best=True,
):
    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_map": [],
        "val_map_rerank": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_map = 0.0
    best_map_rerank = None
    best_checkpoint_metric = float("-inf")
    patience_counter = 0
    best_epoch = 0
    best_state_dict = None
    rerank_config = get_rerank_config(config)
    best_checkpoint_metric_name = "val_map_rerank" if rerank_config["enabled"] else "val_map"

    print(f"Starting training for {config['num_epochs']} epochs...")
    print("=" * 70)

    for epoch in range(config["num_epochs"]):
        # print()
        # print(f"Epoch {epoch+1}/{config['num_epochs']}")

        train_loss, train_acc = train_epoch_embeddings(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate_epoch_embeddings(model, val_loader, criterion, device)
        val_embeddings, val_labels = extract_head_embeddings(model, val_loader, device)
        val_map = compute_map_from_embeddings(val_embeddings, val_labels)
        val_map_rerank = None
        if rerank_config["enabled"]:
            val_map_rerank = compute_map_from_embeddings(
                val_embeddings,
                val_labels,
                use_rerank=True,
                k1=rerank_config["k1"],
                k2=rerank_config["k2"],
                lambda_value=rerank_config["lambda_value"],
            )

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_map"].append(val_map)
        history["val_map_rerank"].append(val_map_rerank)
        history["lr"].append(current_lr)

        if wandb_run is not None:
            log_payload = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "val_map": val_map,
                "learning_rate": current_lr,
            }
            if val_map_rerank is not None:
                log_payload["val_map_rerank"] = val_map_rerank
            wandb_run.log(log_payload)

        # print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.1f}%")
        # print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.1f}%")
        # print(f"  Val mAP:    {val_map:.4f} | LR: {current_lr:.2e}")

        checkpoint_metric, checkpoint_metric_name = get_checkpoint_metric(val_map, val_map_rerank)
        if checkpoint_metric > best_checkpoint_metric:
            best_checkpoint_metric = checkpoint_metric
            best_checkpoint_metric_name = checkpoint_metric_name
            best_val_loss = val_loss
            best_map = val_map
            best_map_rerank = val_map_rerank
            best_epoch = epoch + 1
            patience_counter = 0
            best_state_dict = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

            checkpoint_path = config["checkpoint_dir"] / checkpoint_name
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_map": val_map,
                    "val_map_rerank": val_map_rerank,
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
    print(
        f"Best epoch: {best_epoch} "
        f"({best_checkpoint_metric_name}: {best_checkpoint_metric:.4f}, "
        f"Val Loss: {best_val_loss:.4f}, Val mAP: {best_map:.4f})"
    )

    if restore_best and best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    return {
        "history": history,
        "best_val_loss": best_val_loss,
        "best_map": best_map,
        "best_map_rerank": best_map_rerank,
        "best_checkpoint_metric": best_checkpoint_metric,
        "best_checkpoint_metric_name": best_checkpoint_metric_name,
        "best_epoch": best_epoch,
        "epochs_ran": len(history["train_loss"]),
    }
