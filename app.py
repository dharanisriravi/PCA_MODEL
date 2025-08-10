import os
import io
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
import pandas as pd
from model.pca_analysis import run_pca_and_prepare
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {"csv"}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB

app = Flask(__name__)
app.secret_key = "replace-with-a-strong-secret"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Home / upload
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            flash("No file part")
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            flash("No selected file")
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            unique_name = f"{uuid.uuid4().hex}_{filename}"
            path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(path)

            # get form options
            features = request.form.getlist("features")
            n_components = int(request.form.get("n_components", 2))

            try:
                result = run_pca_and_prepare(path, features=features if features else None, n_components=n_components)
            except Exception as e:
                flash(f"Error processing file: {e}")
                return redirect(request.url)

            # store result in session-like map by id (simple, ephemeral)
            result_id = uuid.uuid4().hex
            # store CSV in-memory for download later
            csv_bytes = result["pca_scores"].to_csv(index=False).encode("utf-8")
            request.environ.setdefault("pca_cache", {})[result_id] = csv_bytes

            return render_template(
                "result.html",
                plot_json=result["plot_data_json"],
                explained_variance=result["explained_variance"],
                columns=result["used_features"],
                preview_html=result["preview_html"],
                result_id=result_id,
                original_filename=filename,
                n_components=n_components
            )
        else:
            flash("Invalid file type. Only CSV allowed.")
            return redirect(request.url)

    return render_template("index.html")

# download PCA scores
@app.route("/download/<result_id>")
def download(result_id):
    cache = request.environ.get("pca_cache", {})
    data = cache.get(result_id)
    if not data:
        # fallback: produce meaningful error
        flash("Download expired or not found. Re-run analysis.")
        return redirect(url_for("index"))
    return send_file(io.BytesIO(data),
                     mimetype="text/csv",
                     as_attachment=True,
                     download_name="pca_scores.csv")

if __name__ == "__main__":
    app.run(debug=True)
