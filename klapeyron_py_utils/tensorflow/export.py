import os
import shutil
from klapeyron_py_utils.tensorflow.imports import import_tensorflow
tf = import_tensorflow()


def export(model: tf.Module, signatures: dict, export_path: str):
    if os.path.isdir(export_path):
        shutil.rmtree(export_path)
    assert not os.path.isdir(export_path)
    os.mkdir(export_path)
    tf.saved_model.save(model, export_path, signatures=signatures)
