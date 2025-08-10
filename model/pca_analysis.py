import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import json

def _select_numeric_features(df):
    # prefer common sports stat names if present
    prefer = ["Goals", "Assists", "Shots", "PassAccuracy", "Speed", "Stamina", "Tackles", "Saves"]
    numeric = df.select_dtypes(include=[np.number]).columns.tolist()
    chosen = [c for c in prefer if c in numeric]
    if chosen:
        return chosen
    # fallback to top 6 numeric features excluding ID-like columns
    filtered = [c for c in numeric if "id" not in c.lower() and "name" not in c.lower()]
    return filtered[:6]

def run_pca_and_prepare(csv_path, features=None, n_components=2):
    df = pd.read_csv(csv_path)
    if df.shape[0] < 3:
        raise ValueError("CSV must contain at least 3 rows of data.")

    # identify ID/name column if present
    id_col = None
    for col in df.columns:
        if col.lower() in ("playerid","player_id","id","name","player"):
            id_col = col
            break
    if id_col is None:
        id_col = "PlayerID"
        df[id_col] = [f"PL_{i+1}" for i in range(len(df))]

    # choose features
    if features:
        used = [f for f in features if f in df.columns]
        if not used:
            raise ValueError("None of the selected feature names are present in the CSV.")
    else:
        used = _select_numeric_features(df)
        if len(used) < 2:
            raise ValueError("Need at least two numeric features for PCA. Found: " + ", ".join(df.select_dtypes(include=[np.number]).columns))

    X = df[used].fillna(df[used].mean())
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    pca = PCA(n_components=min(n_components, Xs.shape[1]))
    pcs = pca.fit_transform(Xs)
    pcs_df = pd.DataFrame(pcs, columns=[f"PC{i+1}" for i in range(pcs.shape[1])])
    out_df = pd.concat([df[[id_col]].reset_index(drop=True), pcs_df], axis=1)

    # prepare plot data: return list of points with id and PC coordinates
    plot_points = []
    for _, row in out_df.iterrows():
        pt = {"id": str(row[id_col])}
        for i in range(pcs.shape[1]):
            pt[f"PC{i+1}"] = float(row[f"PC{i+1}"])
        plot_points.append(pt)

    plot_data_json = json.dumps({
        "points": plot_points
    })

    explained_variance = [float(round(x, 4)) for x in pca.explained_variance_ratio_]

    preview_html = df[[id_col] + used].head(10).to_html(classes="table table-sm", index=False, float_format="{:0.3f}".format)

    return {
        "pca_scores": out_df,
        "plot_data_json": plot_data_json,
        "explained_variance": explained_variance,
        "used_features": used,
        "preview_html": preview_html
    }
