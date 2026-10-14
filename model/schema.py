"""Fixed pantry and context columns for the meal dataset.

The CSV stores heart rate as a raw integer and time of day as a label.
`feature_vector` is the ~47-d layout the network will read later.
"""

from __future__ import annotations

MEATS = ("chicken", "pork", "beef", "turkey", "alligator", "lamb", "goat")

VEGETABLES = (
    "tomato",
    "celery",
    "onion",
    "garlic",
    "potato",
    "carrot",
    "bell_pepper",
    "spinach",
    "broccoli",
    "cabbage",
    "lettuce",
    "cucumber",
    "mushroom",
    "zucchini",
    "corn",
    "peas",
    "green_bean",
    "eggplant",
    "ginger",
    "chili",
    "lemon",
    "herbs",
    "kale",
    "cauliflower",
)

CARBS = ("flour", "rice", "pasta", "potato")

TIMES_OF_DAY = (
    "early_morning",
    "morning",
    "noon",
    "afternoon",
    "evening",
    "night",
    "midnight",
    "latenight",
)

# Overnight slots also accept a recipe tagged evening when grades are written.
OVERNIGHT = frozenset({"night", "midnight", "latenight", "early_morning"})

CONTEXT_COLUMNS = ("heart_rate", "diabetic", "hypertension", "recent_event", "time_of_day")


def meat_column(name: str) -> str:
    return f"meat_{name}"


def veg_column(name: str) -> str:
    return f"veg_{name}"


def carb_column(name: str) -> str:
    return f"carb_{name}"


def grade_column(recipe_id: str) -> str:
    return f"grade_{recipe_id}"


def score_column(recipe_id: str) -> str:
    return f"score_{recipe_id}"


def input_columns() -> list[str]:
    columns = [meat_column(name) for name in MEATS]
    columns += [veg_column(name) for name in VEGETABLES]
    columns += [carb_column(name) for name in CARBS]
    columns += list(CONTEXT_COLUMNS)
    return columns


def feature_dim() -> int:
    # Time of day is one column in the CSV and eight one-hot dims in the vector.
    return len(MEATS) + len(VEGETABLES) + len(CARBS) + 1 + 3 + len(TIMES_OF_DAY)


def scale_heart_rate(heart_rate: int) -> float:
    """Map a raw bpm to roughly [-2, 2]. Missing heart rate is stored as 0."""
    if heart_rate <= 0:
        return 0.0
    scaled = (heart_rate - 70) / 30
    return max(-2.0, min(2.0, scaled))


def feature_vector(row: dict) -> list[float]:
    vector = [float(row[meat_column(name)]) for name in MEATS]
    vector += [float(row[veg_column(name)]) for name in VEGETABLES]
    vector += [float(row[carb_column(name)]) for name in CARBS]
    vector.append(scale_heart_rate(int(row["heart_rate"])))
    vector.append(float(row["diabetic"]))
    vector.append(float(row["hypertension"]))
    vector.append(float(row["recent_event"]))
    time_of_day = row["time_of_day"]
    vector += [1.0 if time_of_day == name else 0.0 for name in TIMES_OF_DAY]
    if len(vector) != feature_dim():
        raise ValueError(f"expected {feature_dim()} features, got {len(vector)}")
    return vector
