import pytest
from pydantic import ValidationError
from schemas.configs.MLP_config import MLPParams


def test_valid_mlp_params():
    params = MLPParams(
        input_size=10,
        hidden_layers=[32, 16],
        output_size=3,
        activation="tanh",
        learning_rate=0.01,
        optimizer="adam",
        loss="cross_entropy",
        epochs=5,
        batch_size=16,
        verbose=True,
    )

    assert params.input_size == 10
    assert params.hidden_layers == [32, 16]
    assert params.output_size == 3
    assert params.activation == "tanh"
    assert params.learning_rate == 0.01
    assert params.optimizer == "adam"
    assert params.loss == "cross_entropy"
    assert params.epochs == 5
    assert params.batch_size == 16
    assert params.verbose is True


def test_invalid_activation():
    with pytest.raises(ValidationError):
        MLPParams(
            input_size=10,
            hidden_layers=[32],
            output_size=1,
            activation="invalid",  # nieprawidłowa wartość typu Literal
            learning_rate=0.001,
            optimizer="adam",
            loss="mse",
            epochs=1
        )


def test_negative_input_size():
    with pytest.raises(ValidationError):
        MLPParams(
            input_size=-5,  # poniżej zera
            hidden_layers=[64],
            output_size=1,
            activation="relu",
            learning_rate=0.001,
            optimizer="adam",
            loss="mse",
            epochs=1
        )


def test_missing_required_fields():
    with pytest.raises(ValidationError):
        MLPParams()  # brak wymaganych pól: input_size, output_size


def test_invalid_dropout_range():
    with pytest.raises(ValidationError):
        MLPParams(
            input_size=10,
            hidden_layers=[32],
            output_size=1,
            activation="relu",
            learning_rate=0.001,
            optimizer="adam",
            loss="mse",
            dropout=1.5,  # poza dozwolonym zakresem
            epochs=5
        )
