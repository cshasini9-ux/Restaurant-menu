"""
model.py
--------
AI Recommendation Engine for the Smart Restaurant Menu Assistant.

This module encapsulates all the machine learning logic used to suggest
menu items to a customer based on their stated preferences (cuisine,
spice level, veg/non-veg type and budget range).

Algorithm used : K-Nearest Neighbors (KNN)
Library used   : scikit-learn (NearestNeighbors)

The idea:
1. Every menu item is converted into a small numeric feature vector
   (Cuisine, Spice Level, Type, Normalized Price).
2. The customer's preferences are converted into the same kind of
   feature vector.
3. KNN finds the menu items whose vectors are "closest" to the
   customer's preference vector.
4. The distance is converted into a friendly "match percentage" score
   that is shown to the user in the UI.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import MinMaxScaler


class MenuRecommender:
    """A simple KNN based recommender for restaurant menu items."""

    # Fixed encoding maps so that results are consistent every run.
    CUISINE_MAP = {"Indian": 0, "Chinese": 1, "Italian": 2, "Mexican": 3, "Continental": 4}
    SPICE_MAP = {"Low": 0, "Medium": 1, "High": 2}
    TYPE_MAP = {"Veg": 0, "Non-Veg": 1}

    # Theoretical maximum euclidean distance across our 4 encoded
    # dimensions -> used to convert raw distance into a 0-100% match score.
    # sqrt((4)^2 + (2)^2 + (1)^2 + (1)^2)
    MAX_DISTANCE = np.sqrt(4 ** 2 + 2 ** 2 + 1 ** 2 + 1 ** 2)

    def __init__(self, data_path: str = "menu_data.csv"):
        """Load the menu dataset and pre-compute encoded feature columns."""
        self.df = pd.read_csv(data_path)
        self._prepare_features()

    def _prepare_features(self):
        """Encode categorical columns into numbers the model can use."""
        self.df["Cuisine_enc"] = self.df["Cuisine"].map(self.CUISINE_MAP)
        self.df["Spice_enc"] = self.df["Spice_Level"].map(self.SPICE_MAP)
        self.df["Type_enc"] = self.df["Type"].map(self.TYPE_MAP)

        # Scale price to a 0-1 range so it does not dominate the
        # distance calculation compared to the other small-range features.
        self.price_scaler = MinMaxScaler()
        self.df["Price_scaled"] = self.price_scaler.fit_transform(self.df[["Price"]])

    def get_categories(self):
        """Return the unique menu categories (Starters, Main Course, etc.)."""
        return sorted(self.df["Category"].unique().tolist())

    def get_cuisines(self):
        """Return the list of cuisines available on the menu."""
        return sorted(self.df["Cuisine"].unique().tolist())

    def filter_menu(self, search_term="", category="All"):
        """Used by the interactive menu page for search + category filter."""
        filtered = self.df.copy()
        if category != "All":
            filtered = filtered[filtered["Category"] == category]
        if search_term:
            mask = (
                filtered["Item_Name"].str.contains(search_term, case=False, na=False)
                | filtered["Description"].str.contains(search_term, case=False, na=False)
            )
            filtered = filtered[mask]
        return filtered

    def recommend(
        self,
        cuisine: str,
        spice_level: str,
        veg_type: str,
        budget_min: float,
        budget_max: float,
        category: str = "All",
        top_n: int = 5,
    ):
        """
        Core AI recommendation function.

        Steps:
        1. Filter the menu by hard constraints the user cannot compromise
           on: veg/non-veg preference, budget range and (optionally)
           category.
        2. Fit a fresh KNN model on the *filtered* candidate items.
        3. Encode the user's soft preferences (cuisine, spice level) into
           a query vector and ask KNN for the closest matching items.
        4. Convert distance -> match percentage for a friendly UI.

        Returns a DataFrame of recommended items with an added
        'Match_Score' column, sorted by best match first.
        """
        candidates = self.df.copy()

        # --- Hard filters -------------------------------------------------
        if veg_type != "Both":
            candidates = candidates[candidates["Type"] == veg_type]

        candidates = candidates[
            (candidates["Price"] >= budget_min) & (candidates["Price"] <= budget_max)
        ]

        if category != "All":
            candidates = candidates[candidates["Category"] == category]

        if candidates.empty:
            return pd.DataFrame()

        # --- Build feature matrix for the candidates ----------------------
        feature_cols = ["Cuisine_enc", "Spice_enc", "Type_enc", "Price_scaled"]
        X = candidates[feature_cols].values

        n_neighbors = min(top_n, len(candidates))
        knn = NearestNeighbors(n_neighbors=n_neighbors, metric="euclidean")
        knn.fit(X)

        # --- Build the user's query vector --------------------------------
        target_price = (budget_min + budget_max) / 2
        target_price_scaled = self.price_scaler.transform([[target_price]])[0][0]

        user_vector = np.array(
            [[
                self.CUISINE_MAP.get(cuisine, 0),
                self.SPICE_MAP.get(spice_level, 0),
                self.TYPE_MAP.get(veg_type, 0) if veg_type != "Both" else 0,
                target_price_scaled,
            ]]
        )

        distances, indices = knn.kneighbors(user_vector)

        results = candidates.iloc[indices[0]].copy()
        raw_distances = distances[0]

        # Convert distance to an intuitive 0-100% match score.
        match_scores = np.clip(100 - (raw_distances / self.MAX_DISTANCE) * 100, 0, 100)
        results["Match_Score"] = np.round(match_scores, 1)

        # Rank by match score first, then by rating as a tiebreaker.
        results = results.sort_values(
            by=["Match_Score", "Rating"], ascending=[False, False]
        ).reset_index(drop=True)

        return results

    def category_distribution(self):
        """Item count per category — used in the analytics dashboard."""
        return self.df["Category"].value_counts().reset_index(
            name="Count"
        ).rename(columns={"index": "Category"})

    def cuisine_distribution(self):
        """Item count per cuisine — used in the analytics dashboard."""
        return self.df["Cuisine"].value_counts().reset_index(
            name="Count"
        ).rename(columns={"index": "Cuisine"})
