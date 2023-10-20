import tensorflow as tf
import hydra
from omegaconf import DictConfig, OmegaConf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


@hydra.main(version_base=None, config_path="../../config", config_name="serving_input")
def main(cfg):
    # Importation du modèle
    model = tf.saved_model.load(
        "mlruns/3/b00d472425744a1c8384a9a0f79c0d99/artifacts/ADVANCED_LSTM/data/model"
    )
    # Conversion du modèle en concrete function
    concrete_func = model.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    concrete_func.inputs[0].set_shape([None, 30])
    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])

    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,  # enable TensorFlow Lite ops.
        tf.lite.OpsSet.SELECT_TF_OPS,  # enable TensorFlow ops.
    ]
    converter._experimental_lower_tensor_list_ops = False
    # Conversion de la concrete function en modèle TFLite
    tflite_model = converter.convert()
    # Exportation
    with open("deployment/backend/model.tflite", "wb") as handle:
        handle.write(tflite_model)


if __name__ == "__main__":
    main()
