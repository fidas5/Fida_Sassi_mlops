import joblib
import numpy as np
import pandas as pd
import pytest

from model_pipeline import (
    load_data,
    prepare_data,
    train_model,
    evaluate_model,
    save_model,
    load_model,
    predict_data,
)

# ==========================================
# DONNEES DE TEST
# ==========================================


@pytest.fixture
def sample_data():
    return pd.DataFrame(
        {
            "RowNumber": range(1, 11),
            "CustomerId": range(1001, 1011),
            "Surname": ["Client"] * 10,
            "CreditScore": [600, 650, 700, 750, 680, 620, 710, 690, 640, 720],
            "Geography": ["France"] * 10,
            "Gender": ["Male", "Female"] * 5,
            "Age": [25, 30, 35, 40, 45, 28, 33, 38, 42, 50],
            "Tenure": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "Balance": [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
            "NumOfProducts": [1, 2] * 5,
            "HasCrCard": [1, 0] * 5,
            "IsActiveMember": [1, 0] * 5,
            "EstimatedSalary": [
                20000,
                30000,
                40000,
                50000,
                60000,
                70000,
                80000,
                90000,
                100000,
                110000,
            ],
            "Exited": [0, 1] * 5,
        }
    )


@pytest.fixture
def training_data():
    x_train = np.array([[25, 50000], [40, 80000], [30, 60000], [50, 90000]])

    y_train = np.array([0, 1, 0, 1])

    return x_train, y_train


# ==========================================
# 1. TESTS LOAD_DATA
# ==========================================


def test_load_data_normal(tmp_path, sample_data):
    file_path = tmp_path / "test.csv"
    sample_data.to_csv(file_path, index=False)

    result = load_data(file_path)

    assert isinstance(result, pd.DataFrame)
    assert len(result) == 10


def test_load_data_file_not_found(tmp_path):
    file_path = tmp_path / "inexistant.csv"

    with pytest.raises(FileNotFoundError):
        load_data(file_path)


# ==========================================
# 2. TESTS PREPARE_DATA
# ==========================================


def test_prepare_data_normal(sample_data, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    x_train, x_test, y_train, y_test = prepare_data(sample_data)

    assert len(x_train) == 8
    assert len(x_test) == 2
    assert len(y_train) == 8
    assert len(y_test) == 2

    assert (tmp_path / "scaler.joblib").exists()


def test_prepare_data_empty(sample_data, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    data = sample_data.copy()
    data["Age"] = np.nan

    with pytest.raises(ValueError):
        prepare_data(data)


# ==========================================
# 3. TESTS TRAIN_MODEL
# ==========================================


def test_train_model_normal(training_data):
    x_train, y_train = training_data

    model = train_model(x_train, y_train)

    predictions = model.predict(x_train)

    assert len(predictions) == len(y_train)
    assert set(predictions).issubset({0, 1})


def test_train_model_invalid_size(training_data):
    x_train, y_train = training_data

    with pytest.raises(ValueError):
        train_model(x_train, y_train[:2])


# ==========================================
# 4. TESTS EVALUATE_MODEL
# ==========================================


def test_evaluate_model_normal(training_data):
    x_train, y_train = training_data

    model = train_model(x_train, y_train)

    accuracy = evaluate_model(model, x_train, y_train)

    assert 0 <= accuracy <= 1


def test_evaluate_model_empty(training_data):
    x_train, y_train = training_data

    model = train_model(x_train, y_train)

    x_empty = np.empty((0, x_train.shape[1]))
    y_empty = np.array([])

    with pytest.raises(ValueError):
        evaluate_model(model, x_empty, y_empty)


# ==========================================
# 5. TESTS SAVE_MODEL
# ==========================================


def test_save_model_normal(training_data, tmp_path):
    x_train, y_train = training_data

    model = train_model(x_train, y_train)

    model_path = tmp_path / "model.joblib"

    save_model(model, model_path)

    assert model_path.exists()


def test_save_model_invalid_path(training_data, tmp_path):
    x_train, y_train = training_data

    model = train_model(x_train, y_train)

    invalid_path = tmp_path / "dossier_inexistant" / "model.joblib"

    with pytest.raises(FileNotFoundError):
        save_model(model, invalid_path)


# ==========================================
# 6. TESTS LOAD_MODEL
# ==========================================


def test_load_model_normal(training_data, tmp_path):
    x_train, y_train = training_data

    model = train_model(x_train, y_train)

    model_path = tmp_path / "model.joblib"

    joblib.dump(model, model_path)

    loaded_model = load_model(model_path)

    assert loaded_model is not None

    predictions = loaded_model.predict(x_train)

    assert len(predictions) == len(y_train)


def test_load_model_file_not_found(tmp_path):
    model_path = tmp_path / "inexistant.joblib"

    with pytest.raises(FileNotFoundError):
        load_model(model_path)


# ==========================================
# 7. TESTS PREDICT_DATA
# ==========================================


def test_predict_data_normal(sample_data, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Create a training dataset
    x_train, x_test, y_train, y_test = prepare_data(sample_data)

    # Train and save the model
    model = train_model(x_train, y_train)
    save_model(model)

    # Create the CSV file used for prediction
    sample_data.to_csv("test_prediction.csv", index=False)

    # Execute prediction
    results = predict_data("test_prediction.csv")

    assert len(results) == len(sample_data)
    assert "Prediction" in results.columns
    assert (tmp_path / "predictions.csv").exists()
    assert set(results["Prediction"].unique()).issubset({0, 1})


def test_predict_data_empty(sample_data, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    # Create an empty CSV with the expected columns
    empty_data = sample_data.iloc[0:0]
    empty_data.to_csv("empty.csv", index=False)

    # Verify that prediction rejects empty data
    with pytest.raises(ValueError):
        predict_data("empty.csv")
