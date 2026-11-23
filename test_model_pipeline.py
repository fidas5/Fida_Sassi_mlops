import numpy as np

from model_pipeline import train_model


def test_train_model():

    # Create a small artificial dataset
    x_train = np.array([
        [25, 50000],
        [40, 80000],
        [30, 60000],
        [50, 90000]
    ])

    y_train = np.array([0, 1, 0, 1])

    # Train the model
    model = train_model(x_train, y_train)

    # Make predictions
    predictions = model.predict(x_train)

    # Verify the results
    assert len(predictions) == len(y_train)

    assert set(predictions).issubset({0, 1})
