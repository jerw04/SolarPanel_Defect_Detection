import tensorflow as tf
import os

def test_model_file_exists():
    assert os.path.exists("saved_model/solar_panel_model_final.h5")

def test_model_load():
    model = tf.keras.models.load_model("saved_model/solar_panel_model_final.h5")
    assert model is not None
