#!/usr/bin/env -S uv run --script
# /// script
# requires-python = "==3.12.*"
# dependencies = [
#   "open-clip-torch==2.31.0",
#   "torch~=2.5.0",
#   "pillow~=10.2.0",
#   "scikit-learn~=1.4.1",
#   "loguru~=0.7.2",
#   "joblib",
#   "polars",
#   "pdf2image",
#   "pillow-heif==0.18.0",
#   "sqlite-vec==0.1.1",
#   "imbalanced-learn==0.12.4",
#   "appdirs==1.4.4",
#   "numpy>2.0.0",
#   "loguru",
#   "imbalanced-learn",
#   "rich",
#   "polars",
#   "tqdm",
#   "pydantic",
# ]
# [tool.uv.sources]
# torch = [
#  { index = "pytorch-rocm", marker = "sys_platform == 'linux'" },
# ]
# torchvision = [
#  { index = "pytorch-rocm", marker = "sys_platform == 'linux'" },
# ]
# [[tool.uv.index]]
# name = "pytorch-rocm"
# url = "https://download.pytorch.org/whl/rocm6.2"
# explicit = false
# [tool.uv]
# index-strategy = "unsafe-best-match"
# ///

import argparse
import hashlib
import json
import os
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime
from os.path import basename
from pathlib import Path
from typing import Generator

import appdirs
import joblib
import lightning as L
import numpy as np
import open_clip
import polars as pl
import sqlite_vec
import torch
from imblearn.over_sampling import RandomOverSampler
from loguru import logger
from PIL import Image
from pillow_heif import register_heif_opener
from sklearn import ensemble, neural_network, svm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV
from tqdm import tqdm

APP_FOLDER = Path(appdirs.user_data_dir("imageclassifier", "openpaul"))
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Database:
    def __init__(self, db_path: Path, use_vec: bool = False, dimension: int = 512):
        self.db_path = db_path
        self.use_vec = use_vec
        self.dimension = dimension
        if not db_path.exists():
            db_path.parent.mkdir(parents=True, exist_ok=True)

        self.connection = sqlite3.connect(db_path)
        if self.use_vec:
            self.connection.enable_load_extension(True)
            sqlite_vec.load(self.connection)
            self.connection.enable_load_extension(False)
        self.setup_tables()

    def setup_tables(self):
        if self.use_vec:
            # Create virtual table for image embeddings
            self.connection.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS embeddings USING vec0(
                embedding float[{self.dimension}],
            )
            """)
            self.connection.execute("""
            CREATE TABLE IF NOT EXISTS image_embeddings (
                rowid INTEGER PRIMARY KEY AUTOINCREMENT,
                hash TEXT,
                model_name TEXT,
                UNIQUE(hash, model_name)
            )
            """)

        else:
            # Create a regular table for image embeddings
            self.connection.execute("""
            CREATE TABLE IF NOT EXISTS image_embeddings (
                embedding BLOB,
                hash TEXT,
                model_name TEXT,
                UNIQUE(hash, model_name)
            )
            """)

        # Create table for image metadata
        self.connection.execute("""
        CREATE TABLE IF NOT EXISTS image_metadata (
            path TEXT,
            hash TEXT UNIQUE
        )
        """)
        self.connection.commit()

    def insert_embedding(self, embedding: torch.Tensor, img_hash: str, model_name: str):
        cursor = self.connection.cursor()
        if self.use_vec:
            # add into image embedding then get the row id of the just emebedded data and add it into the emebddings tabe
            cursor.execute(
                """
            INSERT OR IGNORE INTO image_embeddings (hash, model_name)
            VALUES (?, ?)
            """,
                (img_hash, model_name),  # Store as BLOB
            )
            rowid = cursor.lastrowid
            cursor.execute(
                "INSERT INTO embeddings(rowid, embedding) VALUES (?, ?)",
                (rowid, embedding.cpu().numpy()),
            )
        else:
            cursor.execute(
                """
            INSERT OR IGNORE INTO image_embeddings (embedding, hash, model_name)
            VALUES (?, ?, ?)
            """,
                (embedding.numpy().tobytes(), img_hash, model_name),  # Store as BLOB
            )
        self.connection.commit()

    def insert_metadata(self, path: Path, img_hash: str):
        cursor = self.connection.cursor()
        cursor.execute(
            """
        INSERT OR IGNORE INTO image_metadata (path, hash)
        VALUES (?, ?)
        """,
            (str(path), img_hash),
        )
        self.connection.commit()

    def get_embedding(self, img_hash: str, model_name: str) -> torch.Tensor:
        cursor = self.connection.cursor()

        if self.use_vec:
            # join on rowid
            cursor.execute(
                """
            SELECT e.embedding
            FROM image_embeddings ie
            JOIN embeddings e ON ie.rowid = e.rowid
            WHERE ie.hash = ? AND ie.model_name = ?
            """,
                (img_hash, model_name),
            )

        else:
            cursor.execute(
                """
            SELECT embedding FROM image_embeddings
            WHERE hash = ? AND model_name = ?
            """,
                (img_hash, model_name),
            )
        result = cursor.fetchone()
        if result:
            return torch.tensor(np.frombuffer(result[0], dtype=np.float32))
        return None

    def close(self):
        self.connection.close()

    def _delete_db(self):
        self.connection.close()
        os.remove(self.db_path)

    def get_all_embeddings(self):
        cursor = self.connection.cursor()
        if self.use_vec:
            cursor.execute("""
                SELECT ie.hash, ie.model_name, e.embedding, im.path
                FROM image_embeddings ie
                JOIN embeddings e ON ie.rowid = e.rowid
                JOIN image_metadata im ON ie.hash = im.hash
            """)
        else:
            cursor.execute("""
                SELECT ie.hash, ie.model_name, ie.embedding, im.path
                FROM image_embeddings ie
                JOIN image_metadata im ON ie.hash = im.hash
            """)

        results = cursor.fetchall()
        embeddings = []
        for row in results:
            hash_value, model_name, embedding_blob, path = row
            embedding = torch.tensor(np.frombuffer(embedding_blob, dtype=np.float32))
            embeddings.append(
                {
                    "hash": hash_value,
                    "model": model_name,
                    "embedding": embedding,
                    "path": path,
                }
            )

        return embeddings

    def knn_query(self, query_embedding: torch.Tensor, limit: int = 5):
        if not self.use_vec:
            raise RuntimeError("KNN query is only available when using sqlite-vec.")

        cursor = self.connection.cursor()
        query_embedding_json = json.dumps(query_embedding.tolist())

        # Perform the KNN query
        cursor.execute(
            """
            SELECT rowid, distance
            FROM embeddings
            WHERE embedding MATCH ?
            ORDER BY distance
            LIMIT ?
            """,
            (query_embedding_json, limit),
        )

        results = cursor.fetchall()

        # Extract rowids and distances from the results
        rowids = [row[0] for row in results]  # Get the rowid from each result
        distances = [row[1] for row in results]  # Get the distance from each result

        if not rowids:
            return []  # Return an empty list if no results found

        # Using the row ids, fetch the image_embeddings and image_metadata data
        placeholders = ", ".join(
            "?" for _ in rowids
        )  # Create placeholders for the rowids
        cursor.execute(
            f"""
            SELECT ie.hash, ie.model_name, e.embedding, im.path
                FROM image_embeddings ie
                JOIN embeddings e ON ie.rowid = e.rowid
                JOIN image_metadata im ON ie.hash = im.hash
                WHERE ie.rowid IN ({placeholders})
            """,
            rowids,
        )

        metadata_results = cursor.fetchall()

        # Combine the results into the desired format
        embeddings = []
        for i, (hash_value, model_name, embedding_blob, path) in enumerate(
            metadata_results
        ):
            distance = distances[i]  # Get the corresponding distance
            embedding = torch.tensor(np.frombuffer(embedding_blob, dtype=np.float32))

            embedding_data = {
                "hash": hash_value,
                "model": model_name,
                "embedding": embedding,
                "path": path,
                "distance": distance,
            }
            embeddings.append(embedding_data)

        return embeddings


class Picture:
    use_vec = True
    # _model_name, _dataset_name, dimension = "ViT-B-32", "laion2b_s34b_b79k", 512
    _model_name, _dataset_name, dimension = "ViT-H-14-quickgelu", "dfn5b", 1024
    model, _, preprocess = open_clip.create_model_and_transforms(
        _model_name, pretrained=_dataset_name
    )
    model.eval()
    model.to(DEVICE)

    tokenizer = open_clip.get_tokenizer(_model_name)

    # Initialize the database, store file in filesystem location
    # according to best practice for python packages
    _db_path = (
        APP_FOLDER / f"image_embeddings{_model_name}{'_vec' if use_vec else ''}.db"
    )

    db = Database(_db_path, use_vec=use_vec, dimension=dimension)

    register_heif_opener()

    def __init__(
        self,
        path: Path,
    ):
        self.path: Path = path
        self._date = None

    @property
    def file_type(self) -> str:
        """Determine the image type based on the file extension."""
        return self.path.suffix[1:].lower()

    def train_label(self, parent_folder: Path) -> str:
        # remove parent_folder from self.Path
        # then get the first folder
        return self.path.relative_to(parent_folder).parts[0]

    @staticmethod
    def embedd_image(image: Image) -> torch.Tensor:
        image = Picture.preprocess(image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            image_features = Picture.model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
        return image_features[0]

    @property
    def embedding(self) -> torch.Tensor:
        # First check if the image is already in the database
        img_hash = self.get_hash()
        embedding = self.db.get_embedding(img_hash, self._model_name)
        if embedding is None:
            # Create and store the embedding
            embedding = self.embedd_image(self.load_image())
            self.save_embedding(embedding, img_hash)
        return embedding.reshape(1, -1).cpu()

    @property
    def date(self) -> datetime | None:
        if self._date is None:
            self._date = self._get_image_date()
        return self._date

    def load_image(self) -> Image:
        try:
            image = Image.open(self.path)
            return image
        except Exception as e:
            raise ValueError(f"Could not load image {self.path}") from e

    def get_hash(self) -> str:
        hasher = hashlib.sha256()
        with open(self.path, "rb") as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def save_embedding(self, embedding: torch.Tensor, img_hash: str):
        self.db.insert_embedding(embedding, img_hash, self._model_name)
        self.db.insert_metadata(self.path, img_hash)

    def _get_image_date(self) -> datetime | None:
        """
        Attempts to determine when an image was taken by checking EXIF data first,
        then searching the filename for dates in European format (day before month).
        """
        # Check EXIF data first
        try:
            with Image.open(self.path) as img:
                exif_data = img._getexif()
                if exif_data:
                    for tag in (36867, 36868, 306):
                        value = exif_data.get(tag)
                        if value:
                            try:
                                return datetime.strptime(value, "%Y:%m:%d %H:%M:%S")
                            except ValueError:
                                continue  # Invalid format, try next tag
        except (IOError, AttributeError):
            pass  # Handle invalid images or missing EXIF

        # If EXIF failed, check filename for dates
        filename = basename(self.path)
        date_patterns = [
            (r"\d{4}_\d{2}_\d{2}", "%Y_%m_%d"),  # YYYY-MM-DD
            (r"\d{4}-\d{2}-\d{2}", "%Y-%m-%d"),  # YYYY-MM-DD
            (r"\d{2}-\d{2}-\d{4}", "%d-%m-%Y"),  # DD-MM-YYYY
            (r"\d{2}\.\d{2}\.\d{4}", "%d.%m.%Y"),  # DD.MM.YYYY
            (r"\d{2}/\d{2}/\d{4}", "%d/%m/%Y"),  # DD/MM/YYYY
            (r"\d{8}", "%Y%m%d"),  # YYYYMMDD
            (r"\d{8}", "%d%m%Y"),  # DDMMYYYY
        ]

        for pattern, fmt in date_patterns:
            regex = re.compile(pattern)
            for match in regex.finditer(filename):
                date_str = match.group()
                try:
                    return datetime.strptime(date_str, fmt)
                except ValueError:
                    continue  # Invalid date, try next format

        return None


def date_folder(date: datetime) -> str:
    year = date.year
    month = date.month
    return f"{year}/{month:02d}"


class Video(Picture):
    def __init__(self, path: Path, tempfolder: Path, *args, **kwargs):
        super().__init__(path, *args, **kwargs)

        self.tempfolder = tempfolder

    def load_image(self) -> Image:
        raise NotImplementedError("Video does not support loading images")

    def sample_frames(self, n: int = 5) -> Generator[Picture, None, None]:
        logger.debug(f"Creating {n} frames for {self.path}")
        for frame in self.extract_n_frames(n=n):
            yield Picture(frame)

    @property
    def embedding(self):
        raise NotImplementedError(
            "Video does not support embeddings, sample frames first"
        )

    @property
    def embeddings(self, n: int = 5) -> Generator[torch.Tensor, None, None]:
        for frame in self.sample_frames(n=n):
            yield Picture(frame).embedding

    @property
    def filename_hash(self) -> str:
        hasher = hashlib.sha256()
        hasher.update(self.path.name.encode("utf-8"))
        return hasher.hexdigest()

    def extract_n_frames(self, n: int = 5) -> Generator[Path, None, None]:
        logger.debug(f"Extracting {n} frames from {self.path}")
        for i in range(n):
            out_file = self.tempfolder / f"frame_{self.filename_hash}_{i}.jpg"
            subprocess.run(
                [
                    "ffmpeg",
                    "-i",
                    str(self.path),
                    "-vf",
                    f"select='eq(n\\,{i})'",
                    "-vsync",
                    "vfr",
                    out_file,
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            yield out_file


def majority_vote(labels: list[int | str]) -> int | str:
    return max(set(labels), key=labels.count)


def find_files(
    path: Path, suffixes: list[str] = [".jpg", ".jpeg", ".png", ".heic"]
) -> Generator[Path, None, None]:
    for root, _dirs, files in os.walk(path):
        for file in files:
            if file.lower().endswith(tuple(suffixes)):
                # Nextcloud artifacts
                if file.startswith(".pending"):
                    continue
                yield Path(root) / file


def pictures(
    path: Path, desc: str = "Finding images"
) -> Generator[Picture, None, None]:
    for image in tqdm(list(find_files(path)), desc=desc):
        yield Picture(image)


def videos(
    path: Path, tempfolder: Path, desc: str = "Finding videos"
) -> Generator[Video, None, None]:
    for image in tqdm(list(find_files(path, suffixes=[".mp4", ".mov"])), desc=desc):
        yield Video(image, tempfolder)


class data_loader(L.LightningDataModule):
    def __init__(
        self,
        folder: Path,
        batch_size: int = 32,
        train_split: float = 0.8,
        shuffle: bool = True,
        frames_for_video: int = 5,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        oversampler = RandomOverSampler()
        self.label_map = None
        with tempfile.TemporaryDirectory() as video_tempfiles:
            self.images = list(pictures(folder))
            self.videos = list(videos(folder, tempfolder=Path(video_tempfiles)))

            self.batch_size = batch_size
            data = self.videos + self.images
            if shuffle:
                np.random.shuffle(data)

            labels = []
            self.embeddings = []
            for media_file in tqdm(data, desc="Embedding images/videos"):
                if isinstance(media_file, Video):
                    try:
                        labels.extend(
                            [media_file.train_label(folder)] * frames_for_video
                        )
                        self.embeddings.extend(
                            [
                                frame.embedding[0]
                                for frame in list(
                                    media_file.sample_frames(n=frames_for_video)
                                )
                            ]
                        )
                    except FileNotFoundError as e:
                        logger.error(f"Could not process {media_file.path}")
                        raise e
                elif isinstance(media_file, Picture):
                    labels.append(media_file.train_label(folder))
                    self.embeddings.append(media_file.embedding[0])
                else:
                    raise ValueError("Unknown media file type")

            all_labels = list(set(labels))
            all_labels.sort()
            self.label_map = {label: i for i, label in enumerate(all_labels)}
            self.label_ids = [self.label_map[label] for label in labels]

            # shuffle self.label_ids and embeddings in the same way
            if shuffle:
                index = np.random.permutation(len(self.label_ids))
                self.label_ids = [self.label_ids[i] for i in index]
                self.embeddings = [self.embeddings[i] for i in index]

            self.train_label_ids = self.label_ids[
                : int(train_split * len(self.label_ids))
            ]
            self.train_embeddings = self.embeddings[
                : int(train_split * len(self.embeddings))
            ]

            if True:  # oversample
                self.train_embeddings, self.train_label_ids = oversampler.fit_resample(
                    np.array(self.train_embeddings), np.array(self.train_label_ids)
                )
                # convert to torch
                self.train_embeddings = [
                    torch.tensor(embedding) for embedding in self.train_embeddings
                ]
                self.train_label_ids = torch.tensor(self.train_label_ids)
                # shuffle again
                index = np.random.permutation(len(self.train_label_ids))
                self.train_label_ids = self.train_label_ids[index]
                self.train_embeddings = [self.train_embeddings[i] for i in index]

            self.val_label_ids = self.label_ids[
                int(train_split * len(self.label_ids)) :
            ]
            self.val_embeddings = self.embeddings[
                int(train_split * len(self.embeddings)) :
            ]
            if len(self.val_label_ids) != len(self.val_embeddings):
                logger.error("Validation label length does not match")
                exit(1)
            if len(self.train_label_ids) != len(self.train_embeddings):
                logger.error("Training label length does not match")
                exit(1)

    @staticmethod
    def collate_fn(batch: list[tuple[torch.Tensor, str]]):
        embeddings, labels = zip(*batch)
        return torch.stack(embeddings).to(DEVICE), torch.tensor(labels).to(DEVICE)

    def train_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            list(zip(self.train_embeddings, self.train_label_ids)),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=True,
        )

    def val_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            list(zip(self.val_embeddings, self.val_label_ids)),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
        )

    def predict_dataloader(self) -> torch.utils.data.DataLoader:
        return torch.utils.data.DataLoader(
            list(zip(self.embeddings, [-1] * len(self.embeddings))),
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            shuffle=False,
        )


class Classifier:
    def __init__(
        self,
        mode: str = "RandomForest",
    ):
        self.mode = mode
        self.model_file = APP_FOLDER / f"{mode}.joblib"
        self.label_map = None

    def _load_classifier(self):
        if self.model_file.exists():
            data = joblib.load(self.model_file)
            self.classifier = data["classifier"]
            self.label_map = data["label_map"]
            self.reverse_label_map = {v: k for k, v in self.label_map.items()}
        else:
            raise FileNotFoundError(f"Could not find model file {self.model_file}")

    @classmethod
    def load(cls, mode: str = "RandomForest") -> "Classifier":
        """Class method to create an instance and load the classifier."""
        instance = cls(mode)
        instance._load_classifier()  # Load the classifier
        return instance

    def validate(
        self, dataset: torch.utils.data.Dataset
    ) -> tuple[np.ndarray, np.ndarray]:
        X, y = self._get_data(dataset)
        return self.classifier.predict(X), y

    def _get_data(self, dataset: torch.utils.data.Dataset):
        X = []
        y = []
        for embeddings, labels in dataset:
            X.extend(embeddings.cpu().numpy())
            y.extend(labels.cpu().numpy())
        return np.array(X), np.array(y)

    def train(self, dataset: torch.utils.data.Dataset, label_map: dict | None):
        if label_map is None:
            logger.warning("You should really pass a label map")
        self.label_map = label_map
        if self.mode == "RandomForest":
            self.classifier = ensemble.RandomForestClassifier()
        elif self.mode == "SVM":
            self.classifier = svm.SVC()
        elif self.mode == "GradientBoosting":
            self.classifier = ensemble.GradientBoostingClassifier()
        elif self.mode == "MLP":
            self.classifier = neural_network.MLPClassifier()
        else:
            raise ValueError(f"Unknown {self.mode}")

        X, y = self._get_data(dataset)
        self._train_and_crossvalidate(X, y)

    def _train_and_crossvalidate(self, X: np.ndarray, y: np.ndarray):
        logger.debug(f"Training {self.mode}. Shape of input data: {X.shape} {y.shape}")
        # Define parameter grid for GridSearchCV
        param_grid = {}
        if self.mode == "SVM":
            param_grid = {"kernel": ["linear", "rbf"], "C": [0.1, 1, 10, 20]}
        elif self.mode == "RandomForest":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "max_depth": [None, 10, 20, 50],
            }
        elif self.mode == "LogisticRegression":
            param_grid = {"C": [0.1, 1, 10, 20], "solver": ["liblinear", "lbfgs"]}
        elif self.mode == "GradientBoosting":
            param_grid = {"n_estimators": [5, 10, 50], "learning_rate": [0.1, 0.01]}
        elif self.mode == "MLP":
            param_grid = {
                "hidden_layer_sizes": [(50,), (100,), (50, 50)],
                "activation": ["relu"],
                "alpha": [0.0001, 0.001, 0.01],
                "max_iter": [250, 500],
            }

        # Perform GridSearchCV
        grid_search = GridSearchCV(
            estimator=self.classifier, param_grid=param_grid, cv=5
        )
        grid_search.fit(X, y)
        print(grid_search.best_params_)
        self.classifier = grid_search.best_estimator_
        logger.debug("Fitted model")

    def save(self):
        joblib.dump(
            {"classifier": self.classifier, "label_map": self.label_map},
            self.model_file,
        )
        logger.debug(f"Saved model to {self.model_file}")

    def predict(self, dataset: torch.utils.data.Dataset) -> np.ndarray:
        X, _ = self._get_data(dataset)
        return self.classifier.predict(X)

    def resolve_labels(self, predictions: np.ndarray) -> list[str]:
        if self.reverse_label_map is None:
            raise ValueError("No label map stored")
        predicted_labels = [self.reverse_label_map[p] for p in predictions]
        return predicted_labels

    def resolve_label(self, prediction) -> str:
        return self.reverse_label_map.get(prediction, str(prediction))


def process_file(
    model: Classifier,
    media_file: Picture | Video,
    output: Path,
    move: bool = False,
    dry_run: bool = False,
    force: bool = False,
    frames_for_video: int = 5,
):
    predicted_label = "UNDEFINED"
    if isinstance(media_file, Video):
        try:
            video_predictions = model.classifier.predict(
                np.array(
                    [
                        frame.embedding[0]
                        for frame in list(media_file.sample_frames(n=frames_for_video))
                    ]
                )
            )
            predicted_label = majority_vote(model.resolve_labels(video_predictions))
        except Exception as e:
            logger.warning(f"Error processing video '{media_file.path}'")
            logger.debug(e)

    elif isinstance(media_file, Picture):
        try:
            predicted_label = model.resolve_label(
                model.classifier.predict(media_file.embedding)[0]
            )
        except Exception as e:
            logger.warning(f"Error processing picture '{media_file.path}'")
            logger.debug(e)
    else:
        logger.error(f"Unsupported file type {type(media_file)}")
        return

    if predicted_label == "UNDEFINED":
        logger.error("Label Not predicted")
        return
    date_prefix = date_folder(media_file.date) if media_file.date else "unknown"
    folder = output / date_prefix / predicted_label
    folder.mkdir(parents=True, exist_ok=True)
    target_path = folder / media_file.path.name

    if target_path == media_file.path:
        logger.debug("Input and output location are the same")
    elif force or not target_path.exists():
        if dry_run:
            logger.debug(f"Dry run: {media_file.path} to {target_path}")
        elif move:
            logger.debug(f"Moving {media_file.path} to {target_path}")
            shutil.move(media_file.path, target_path)
        else:
            logger.debug(f"Copying {media_file.path} to {target_path}")
            shutil.copy(media_file.path, target_path)
    else:
        logger.trace(f"File '{media_file.path}' already exists in {target_path}")


def delete_empty_subfolders(folder_path: Path):
    folder = Path(folder_path)
    if not folder.is_dir():
        logger.warning(f"The path {folder_path} is not a valid directory.")
        return

    for subfolder in folder.rglob("*"):
        if subfolder.is_dir() and not any(subfolder.iterdir()):
            try:
                subfolder.rmdir()  # Remove the empty directory
                logger.info(f"Deleted empty folder: {subfolder}")
            except OSError as e:
                logger.error(f"Error deleting folder {subfolder}: {e}")


def classify(args: argparse.Namespace):
    logger.info("Classifying images")
    model = Classifier.load("RandomForest")
    with tempfile.TemporaryDirectory() as tempfolder:
        for file in pictures(args.input, desc="Classifying pictures"):
            process_file(
                model,
                media_file=file,
                output=args.output,
                move=args.move,
                dry_run=args.dry_run,
                force=args.force,
            )
        for file in videos(args.input, Path(tempfolder), desc="Classifying videos"):
            process_file(
                model,
                media_file=file,
                output=args.output,
                move=args.move,
                dry_run=args.dry_run,
                force=args.force,
            )
    logger.info(f"Finished classifying your folder '{args.input}`")
    # cleanup empty folders
    delete_empty_subfolders(folder_path=args.input)


def _ml_evaluate(predicted: list, labels: list):
    predicted = [str(p) for p in predicted]
    labels = [str(l) for l in labels]
    categories = list(set(labels))
    sorted(categories)
    overall_accuracy = accuracy_score(labels, predicted)
    overall_precision, overall_recall, overall_f1, overall_support = (
        precision_recall_fscore_support(labels, predicted, average="macro")
    )

    overall_metrics = {
        "Category": ["Overall"],
        "Accuracy": [overall_accuracy],
        "Precision": [overall_precision],
        "Recall": [overall_recall],
        "F1 Score": [overall_f1],
        "Support": [overall_support],
    }
    overall_df = pl.DataFrame(overall_metrics)

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, average=None, labels=categories
    )

    category_metrics = {
        "Category": categories,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "Support": support,
    }
    category_df = pl.DataFrame(category_metrics)
    combined_df = pl.concat([overall_df, category_df], how="diagonal_relaxed")

    return combined_df.sort("Category")


def train(args: argparse.Namespace):
    # here we train a classical ML model
    logger.info("Random Forest ML Model")
    data = data_loader(folder=args.folder, batch_size=16, train_split=0.8)
    classifier = Classifier(mode="RandomForest")
    classifier.train(data.train_dataloader(), label_map=data.label_map)
    classifier.save()
    p, y = classifier.validate(data.val_dataloader())
    print(_ml_evaluate(p, y))


def cli():
    parser = argparse.ArgumentParser(
        description="Image classification and sorting tool"
    )
    parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase verbosity"
    )

    # subparser
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser(
        "train", help="Train a random forest model on CLIP embeddings"
    )
    train_parser.add_argument(
        "folder", type=str, help="Folder containing training images"
    )
    train_parser.set_defaults(func=train)

    # inference
    inference_parser = subparsers.add_parser(
        "classify",
        help="Classify images using a trained model and copy or move them into a target folder, arranged by date and class",
    )
    inference_parser.add_argument(
        "-b", "--batch-size", type=int, default=4, help="Batch size for inference"
    )
    inference_parser.add_argument(
        "-k",
        help="Number of frames to sample from videos",
        type=int,
        default=5,
    )
    inference_parser.add_argument(
        "-i",
        "--input",
        type=Path,
        help="Folder containing inference images",
        required=True,
    )
    inference_parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output folder for classified images",
        required=True,
    )

    inference_parser.add_argument(
        "--move",
        help="Move file instead of copying it",
        default=False,
        action="store_true",
    )
    inference_parser.add_argument(
        "-n",
        "--dry-run",
        help="Dont copy or move",
        default=False,
        action="store_true",
    )
    inference_parser.add_argument(
        "-f",
        "--force",
        action="store_true",
        default=False,
        help="Force the copy or move of the files, ignore existing files",
    )
    inference_parser.set_defaults(func=classify)

    return parser.parse_args()


def main():
    arguments = cli()

    # setup loguru in the right verbosity
    logger.remove()
    logger.add(sys.stderr, level="WARNING")
    if arguments.verbose == 1:
        logger.add(sys.stderr, level="INFO")
    elif arguments.verbose == 2:
        logger.add(sys.stderr, level="DEBUG")
    elif arguments.verbose >= 3:
        logger.add(sys.stderr, level="TRACE")
    arguments.func(arguments)


if __name__ == "__main__":
    main()
