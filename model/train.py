"""Train the ranker on data/meals.csv.

Target probability for a row is grade / sum(grades) over every recipe.
A grade of 0 has target probability 0 and still sits in the softmax.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from generate_dataset import load_catalog
from model import MealRanker
from schema import feature_vector, grade_column, input_columns

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "meals.csv"
DEFAULT_CHECKPOINT = ROOT / "model" / "artifacts" / "ranker.pt"


def load_table(path: Path, recipe_ids: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    grades = []
    grade_names = [grade_column(recipe_id) for recipe_id in recipe_ids]
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            features.append(feature_vector(row))
            grades.append([int(row[name]) for name in grade_names])
    feature_tensor = torch.tensor(features, dtype=torch.float32)
    grade_tensor = torch.tensor(grades, dtype=torch.float32)
    if feature_tensor.shape[1] != 47:
        raise ValueError(f"expected 47 features, found {feature_tensor.shape[1]}")
    return feature_tensor, grade_tensor


def soft_targets(grades: torch.Tensor) -> torch.Tensor:
    totals = grades.sum(dim=-1, keepdim=True)
    if torch.any(totals <= 0):
        raise ValueError("a row has no positive grade")
    return grades / totals


def batch_loss(model: MealRanker, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(model(features), dim=-1)
    return -(targets * log_probs).sum(dim=-1).mean()


@torch.no_grad()
def evaluate(model: MealRanker, features: torch.Tensor, grades: torch.Tensor) -> dict[str, float]:
    model.eval()
    targets = soft_targets(grades)
    loss = batch_loss(model, features, targets).item()
    probs = torch.softmax(model(features), dim=-1)
    predicted = probs.argmax(dim=-1)
    best_grade = grades.max(dim=-1).values.unsqueeze(1)
    top_hit = grades.gather(1, predicted.unsqueeze(1)).squeeze(1) == best_grade.squeeze(1)
    zero_mask = grades == 0
    zero_mass = (probs * zero_mask).sum(dim=-1).mean().item()
    return {
        "loss": loss,
        "top_grade_hit": top_hit.float().mean().item(),
        "zero_mass": zero_mass,
    }


def train(
    data_path: Path,
    checkpoint_path: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> dict[str, float]:
    torch.manual_seed(seed)
    recipes = load_catalog()
    recipe_ids = [recipe["id"] for recipe in recipes]
    features, grades = load_table(data_path, recipe_ids)
    targets = soft_targets(grades)

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(features.shape[0], generator=generator)
    split = int(features.shape[0] * 0.9)
    train_index = permutation[:split]
    val_index = permutation[split:]

    model = MealRanker(recipe_count=len(recipe_ids))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, epochs + 1):
        model.train()
        order = train_index[torch.randperm(train_index.shape[0], generator=generator)]
        total = 0.0
        seen = 0
        for start in range(0, order.shape[0], batch_size):
            batch = order[start : start + batch_size]
            loss = batch_loss(model, features[batch], targets[batch])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * batch.shape[0]
            seen += batch.shape[0]
        if epoch == 1 or epoch == epochs or epoch % 10 == 0:
            metrics = evaluate(model, features[val_index], grades[val_index])
            print(
                f"epoch {epoch:02d} train_loss {total / seen:.4f} "
                f"val_loss {metrics['loss']:.4f} "
                f"top_grade_hit {metrics['top_grade_hit']:.3f} "
                f"zero_mass {metrics['zero_mass']:.3f}"
            )

    metrics = evaluate(model, features[val_index], grades[val_index])
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "recipe_ids": recipe_ids,
            "model_state": model.state_dict(),
            "input_columns": input_columns(),
            "val_metrics": metrics,
        },
        checkpoint_path,
    )
    (checkpoint_path.parent / "recipe_ids.json").write_text(json.dumps(recipe_ids, indent=2))
    print(f"wrote {checkpoint_path}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the meal ranker on the grade table.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=14)
    args = parser.parse_args()
    train(args.data, args.checkpoint, args.epochs, args.batch_size, args.lr, args.seed)


if __name__ == "__main__":
    main()
