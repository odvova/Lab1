import streamlit as st
from tensorflow import keras
import pandas as pd
import numpy as np
import os
import json
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import random


@st.cache_data
def load_data():
    fashion_mnist = keras.datasets.fashion_mnist
    (x_train_full, y_train_full), (x_test, y_test) = fashion_mnist.load_data()
    x_train_full = x_train_full / 255.0
    x_test = x_test / 255.0
    return (x_train_full, y_train_full), (x_test, y_test)


class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

loss_fn = keras.losses.SparseCategoricalCrossentropy()


def create_model(architecture):
    if architecture == "ReLU-LeakyReLU-ELU Model":
        model = keras.models.Sequential(
            [
                keras.layers.Flatten(input_shape=[28, 28]),
                keras.layers.Dense(100, activation="relu"),
                keras.layers.Dense(50, activation="leaky_relu"),
                keras.layers.Dense(20, activation="elu"),
                keras.layers.Dense(10, activation="softmax"),
            ]
        )
    elif architecture == "ReLU-ELU-GELU Model":
        model = keras.models.Sequential(
            [
                keras.layers.Flatten(input_shape=[28, 28]),
                keras.layers.Dense(100, activation="relu"),
                keras.layers.Dense(50, activation="elu"),
                keras.layers.Dense(20, activation="gelu"),
                keras.layers.Dense(10, activation="softmax"),
            ]
        )
    elif architecture == "GELU-ReLU-Tanh Model":
        model = keras.models.Sequential(
            [
                keras.layers.Flatten(input_shape=[28, 28]),
                keras.layers.Dense(100, activation="gelu"),
                keras.layers.Dense(50, activation="relu"),
                keras.layers.Dense(20, activation="tanh"),
                keras.layers.Dense(10, activation="softplus"),
            ]
        )

    model.compile(loss=loss_fn, optimizer="sgd", metrics=["accuracy"])
    return model


def train_model(_model, x_train, y_train, x_valid, y_valid):
    history = _model.fit(
        x_train, y_train, epochs=30, validation_data=(x_valid, y_valid), verbose=0
    )
    return history


def load_history(history_file):
    with open(history_file, "r") as f:
        history = json.load(f)
    return history


def save_history(history, history_file):
    with open(history_file, "w") as f:
        json.dump(history, f)


st.title("Fashion MNIST Model Trainer")

(x_train_full, y_train_full), (x_test, y_test) = load_data()

validation_size = 4000
x_valid, x_train = x_train_full[:validation_size], x_train_full[validation_size:]
y_valid, y_train = y_train_full[:validation_size], y_train_full[validation_size:]


@st.cache_resource
def pretrain_all_models():
    pretrained_results = {}

    for model_name in [
        "ReLU-LeakyReLU-ELU Model",
        "ReLU-ELU-GELU Model",
        "GELU-ReLU-Tanh Model",
    ]:
        weight_file = f"{model_name}_weights.h5"
        history_file = f"{model_name}_history.json"

        if os.path.exists(weight_file) and os.path.exists(history_file):
            model = create_model(model_name)
            model.load_weights(weight_file)
            history = load_history(history_file)
        else:
            model = create_model(model_name)
            history = train_model(model, x_train, y_train, x_valid, y_valid)
            model.save_weights(weight_file)
            history_dict = history.history
            save_history(history_dict, history_file)

        pretrained_results[model_name] = (
            history.history if isinstance(history, keras.callbacks.History) else history
        )

    return pretrained_results


pretrained_results = pretrain_all_models()

st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a model to view pretrained results",
    ["ReLU-LeakyReLU-ELU Model", "ReLU-ELU-GELU Model", "GELU-ReLU-Tanh Model"],
)

st.sidebar.subheader(f"Selected: {model_choice}")

selected_history = pretrained_results[model_choice]
st.subheader("Learning Curve")
st.line_chart(
    pd.DataFrame(selected_history)[["accuracy", "val_accuracy", "loss", "val_loss"]]
)

st.write(f"Final Training Accuracy: {selected_history['accuracy'][-1]:.4f}")
st.write(f"Final Validation Accuracy: {selected_history['val_accuracy'][-1]:.4f}")


grid_size = 5

if st.sidebar.button("Display Random Image Grid"):
    model = create_model(model_choice)
    model.load_weights(f"{model_choice}_weights.h5")

    random_indices = random.sample(range(0, 10000), 20)

    for i in range(0, 20, grid_size):
        cols = st.columns(grid_size)

        for j in range(grid_size):
            idx = random_indices[i + j]
            test_image = x_test[idx]
            test_label = y_test[idx]
            test_image_reshaped = np.expand_dims(test_image, axis=0)

            predictions = model.predict(test_image_reshaped)
            predicted_class = np.argmax(predictions, axis=1)[0]

            with cols[j]:
                st.image(
                    test_image,
                    caption=f"Actual: {class_names[test_label]}\nPredicted: {class_names[predicted_class]}",
                    width=100,
                    use_column_width=False,
                )


if st.sidebar.button("Classification Report"):
    model = create_model(model_choice)
    model.load_weights(f"{model_choice}_weights.h5")
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    st.text("Classification Report")
    st.text(classification_report(y_test, y_pred_classes, target_names=class_names))
