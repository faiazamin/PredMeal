"""Write phase-1 pantry rows and 0-5 recipe grades.

Authoring checks live here only. They fill the grade table. The watch model
does not import this module.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from schema import (
    CARBS,
    MEATS,
    OVERNIGHT,
    TIMES_OF_DAY,
    VEGETABLES,
    carb_column,
    feature_dim,
    grade_column,
    input_columns,
    meat_column,
    scale_heart_rate,
    score_column,
    veg_column,
)

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = Path(__file__).with_name("catalog.json")
DEFAULT_OUTPUT = ROOT / "data" / "meals.csv"

SHORT_COOK_MINUTES = 20
LOW_SCORE_FRACTION = 0.15


def load_catalog(path: Path = CATALOG_PATH) -> list[dict]:
    recipes = json.loads(path.read_text())
    _validate_catalog(recipes)
    return recipes


def _validate_catalog(recipes: list[dict]) -> None:
    if len(recipes) != 24:
        raise ValueError(f"expected 24 recipes, found {len(recipes)}")
    known = set(MEATS) | set(VEGETABLES) | set(CARBS)
    seen: set[str] = set()
    for recipe in recipes:
        recipe_id = recipe["id"]
        if recipe_id in seen:
            raise ValueError(f"duplicate recipe id {recipe_id}")
        seen.add(recipe_id)
        if not recipe["ingredients"]:
            raise ValueError(f"{recipe_id} has no ingredients")
        unknown = [item for item in recipe["ingredients"] if item not in known]
        if unknown:
            raise ValueError(f"{recipe_id} has unknown ingredients {unknown}")
        if recipe["cook_minutes"] <= 0:
            raise ValueError(f"{recipe_id} has a non-positive cook time")
        bad_periods = [item for item in recipe["meal_periods"] if item not in TIMES_OF_DAY]
        if bad_periods:
            raise ValueError(f"{recipe_id} has unknown meal periods {bad_periods}")
        missing_affinity = [item for item in TIMES_OF_DAY if item not in recipe["time_affinity"]]
        if missing_affinity:
            raise ValueError(f"{recipe_id} is missing affinity for {missing_affinity}")
        if not 0 < float(recipe["popularity"]) <= 1:
            raise ValueError(f"{recipe_id} popularity must be in (0, 1]")


def empty_pantry() -> dict:
    row = {column: 0 for column in input_columns() if column not in ("heart_rate", "time_of_day")}
    row["heart_rate"] = 0
    row["time_of_day"] = "evening"
    return row


def mark_present(row: dict, ingredient: str) -> None:
    if ingredient in MEATS:
        row[meat_column(ingredient)] = 1
    elif ingredient == "potato":
        row[carb_column("potato")] = 1
        row[veg_column("potato")] = 1
    elif ingredient in CARBS:
        row[carb_column(ingredient)] = 1
    elif ingredient in VEGETABLES:
        row[veg_column(ingredient)] = 1
    else:
        raise KeyError(ingredient)


def has_ingredient(row: dict, ingredient: str) -> bool:
    if ingredient in MEATS:
        return row[meat_column(ingredient)] == 1
    if ingredient == "potato":
        return row[carb_column("potato")] == 1 or row[veg_column("potato")] == 1
    if ingredient in CARBS:
        return row[carb_column(ingredient)] == 1
    if ingredient in VEGETABLES:
        return row[veg_column(ingredient)] == 1
    raise KeyError(ingredient)


def period_allows(recipe: dict, time_of_day: str) -> bool:
    periods = set(recipe["meal_periods"])
    if time_of_day in periods:
        return True
    return time_of_day in OVERNIGHT and "evening" in periods


def authoring_zero(recipe: dict, row: dict) -> bool:
    """True when this recipe is a no-go for reasons other than weak preference."""
    if any(not has_ingredient(row, ingredient) for ingredient in recipe["ingredients"]):
        return True
    if row["recent_event"] == 1 and recipe["cook_minutes"] > SHORT_COOK_MINUTES:
        return True
    if row["diabetic"] == 1 and not recipe["low_sugar"]:
        return True
    if row["hypertension"] == 1 and not recipe["low_sodium"]:
        return True
    if not period_allows(recipe, row["time_of_day"]):
        return True
    return False


def preference_score(recipe: dict, row: dict) -> float:
    """Rank a recipe that already passed the authoring checks."""
    numerator = 0.0
    denominator = 0.0
    for ingredient in recipe["ingredients"]:
        weight = 2.0 if ingredient in MEATS or ingredient in CARBS else 1.0
        denominator += weight
        if has_ingredient(row, ingredient):
            numerator += weight
    coverage = numerator / denominator
    affinity = float(recipe["time_affinity"][row["time_of_day"]])
    score = coverage * affinity * float(recipe["popularity"])
    if scale_heart_rate(int(row["heart_rate"])) > 1:
        score *= 1.0 / recipe["cook_minutes"]
    return score


def grade_row(recipes: list[dict], row: dict) -> tuple[dict[str, int], dict[str, float]]:
    scores = {recipe["id"]: 0.0 for recipe in recipes}
    passing = []
    for recipe in recipes:
        if authoring_zero(recipe, row):
            continue
        score = preference_score(recipe, row)
        scores[recipe["id"]] = score
        passing.append((recipe["id"], score))

    grades = {recipe["id"]: 0 for recipe in recipes}
    if not passing:
        return grades, scores

    best = max(score for _, score in passing)
    if best <= 0:
        return grades, scores

    for recipe_id, score in passing:
        if score < LOW_SCORE_FRACTION * best:
            continue
        raw = int(5 * score / best + 0.5)
        grades[recipe_id] = min(5, max(1, raw))
    return grades, scores


def _sample_heart_rate(rng: random.Random) -> int:
    if rng.random() < 0.05:
        return 0
    if rng.random() < 0.2:
        heart_rate = rng.gauss(110, 12)
    else:
        heart_rate = rng.gauss(70, 8)
    return int(max(48, min(160, round(heart_rate))))


def _add_random_ingredients(row: dict, rng: random.Random, count: int) -> None:
    pool = list(MEATS) + list(VEGETABLES) + list(CARBS)
    for ingredient in rng.sample(pool, k=min(count, len(pool))):
        mark_present(row, ingredient)


def _clear_ingredient(row: dict, ingredient: str) -> None:
    if ingredient in MEATS:
        row[meat_column(ingredient)] = 0
    elif ingredient == "potato":
        row[carb_column("potato")] = 0
        row[veg_column("potato")] = 0
    elif ingredient in CARBS:
        row[carb_column(ingredient)] = 0
    else:
        row[veg_column(ingredient)] = 0


def sample_row(recipes: list[dict], rng: random.Random) -> dict:
    row = empty_pantry()
    time_of_day = rng.choice(TIMES_OF_DAY)
    row["time_of_day"] = time_of_day
    if rng.random() < 0.8:
        pool = [recipe for recipe in recipes if period_allows(recipe, time_of_day)]
        group_size = min(len(pool), rng.choice((2, 2, 3, 3, 1)))
        chosen = rng.sample(pool, k=group_size) if pool else []
        for recipe in chosen:
            for ingredient in recipe["ingredients"]:
                mark_present(row, ingredient)
        if rng.random() < 0.25 and chosen:
            victim = rng.choice(chosen)
            _clear_ingredient(row, rng.choice(victim["ingredients"]))
        _add_random_ingredients(row, rng, rng.randint(0, 3))
    else:
        for ingredient in rng.sample(list(MEATS), k=rng.randint(0, 2)):
            mark_present(row, ingredient)
        for ingredient in rng.sample(list(CARBS), k=rng.randint(0, 2)):
            mark_present(row, ingredient)
        for ingredient in rng.sample(list(VEGETABLES), k=rng.randint(1, 5)):
            mark_present(row, ingredient)

    row["heart_rate"] = _sample_heart_rate(rng)
    row["diabetic"] = int(rng.random() < 0.2)
    row["hypertension"] = int(rng.random() < 0.2)
    row["recent_event"] = int(rng.random() < 0.35)
    return row


def generate_rows(recipes: list[dict], count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    attempts = 0
    limit = count * 20
    while len(rows) < count and attempts < limit:
        attempts += 1
        pantry = sample_row(recipes, rng)
        grades, scores = grade_row(recipes, pantry)
        if all(grade == 0 for grade in grades.values()):
            continue
        record = dict(pantry)
        for recipe in recipes:
            record[grade_column(recipe["id"])] = grades[recipe["id"]]
            record[score_column(recipe["id"])] = round(scores[recipe["id"]], 6)
        rows.append(record)
    if len(rows) < count:
        raise RuntimeError(f"only kept {len(rows)} rows after {attempts} attempts")
    return rows


def write_csv(rows: list[dict], recipes: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = input_columns()
    fieldnames += [grade_column(recipe["id"]) for recipe in recipes]
    fieldnames += [score_column(recipe["id"]) for recipe in recipes]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def check_authoring_rules(recipes: list[dict]) -> None:
    by_id = {recipe["id"]: recipe for recipe in recipes}

    fried = empty_pantry()
    for ingredient in by_id["chicken_fried_rice"]["ingredients"]:
        mark_present(fried, ingredient)
    fried["time_of_day"] = "evening"
    fried["heart_rate"] = 70
    grades, _ = grade_row(recipes, fried)
    if grades["chicken_fried_rice"] != 5:
        raise AssertionError("a full chicken-fried-rice pantry in the evening should grade 5")
    if grades["goat_curry"] != 0:
        raise AssertionError("goat curry must be 0 when goat is absent")

    missing_chicken = dict(fried)
    missing_chicken["meat_chicken"] = 0
    grades, _ = grade_row(recipes, missing_chicken)
    if grades["chicken_fried_rice"] != 0:
        raise AssertionError("missing chicken must zero chicken fried rice")

    rushed = dict(fried)
    mark_present(rushed, "beef")
    mark_present(rushed, "broccoli")
    rushed["recent_event"] = 1
    grades, _ = grade_row(recipes, rushed)
    if grades["beef_rice_bowl"] != 0:
        raise AssertionError("a cook over 20 minutes must be 0 when recent_event is set")
    if grades["midnight_chicken_rice"] == 0:
        raise AssertionError("a 12 minute dish with its ingredients should survive recent_event")

    diabetic = dict(fried)
    mark_present(diabetic, "pork")
    mark_present(diabetic, "cabbage")
    diabetic["diabetic"] = 1
    grades, _ = grade_row(recipes, diabetic)
    if grades["pork_fried_rice"] != 0:
        raise AssertionError("pork fried rice is not low_sugar and must be 0 for diabetic")
    if grades["chicken_fried_rice"] == 0:
        raise AssertionError("chicken fried rice is low_sugar and should stay graded")

    midnight = dict(fried)
    for ingredient in ("spinach", "mushroom", "herbs"):
        mark_present(midnight, ingredient)
    midnight["time_of_day"] = "midnight"
    grades, _ = grade_row(recipes, midnight)
    if grades["chicken_fried_rice"] == 0:
        raise AssertionError("evening recipes stay gradable overnight")
    if grades["spinach_scramble"] != 0:
        raise AssertionError("a morning-only dish must be 0 at midnight")

    if feature_dim() != 47:
        raise AssertionError(f"feature width drifted to {feature_dim()}")


def summarize(rows: list[dict], recipes: list[dict]) -> None:
    positive_counts = []
    mixed = 0
    for row in rows:
        grades = [row[grade_column(recipe["id"])] for recipe in recipes]
        positive = [grade for grade in grades if grade > 0]
        positive_counts.append(len(positive))
        if len(set(positive)) > 1:
            mixed += 1
    print(f"rows {len(rows)}")
    print(f"feature dim {feature_dim()}")
    print(
        "positive grades per row: "
        f"min {min(positive_counts)} mean {sum(positive_counts) / len(positive_counts):.2f} "
        f"max {max(positive_counts)}"
    )
    print(f"rows with more than one positive grade value: {mixed}")
    for recipe in recipes:
        column = grade_column(recipe["id"])
        tally = [sum(row[column] == grade for row in rows) for grade in range(6)]
        print(f"{recipe['id']}: " + " ".join(f"{grade}:{count}" for grade, count in enumerate(tally)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the phase-1 meal grade table.")
    parser.add_argument("--rows", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=14)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    recipes = load_catalog()
    check_authoring_rules(recipes)
    rows = generate_rows(recipes, args.rows, args.seed)
    write_csv(rows, recipes, args.output)
    summarize(rows, recipes)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
