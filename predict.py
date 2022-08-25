import os
import shutil
import tempfile
from typing import List, Callable

import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from rich import print

TFLITE_FILE_PATH = "converted_model.tflite"
DEFAULT_SIGNATURE_KEY = "serving_default"

PRECISION = 4
EXP = 10**PRECISION


interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
siglist = interpreter.get_signature_list()
inputd = interpreter.get_input_details()
print(f"{siglist=}")
print(f"{inputd=}")

shape = inputd[0]["shape"]
WIDTH = shape[1]
HEIGHT = shape[2]
CHANNELS = shape[3]


INKEY = siglist[DEFAULT_SIGNATURE_KEY]["inputs"][0]
OUTKEY = siglist[DEFAULT_SIGNATURE_KEY]["outputs"][0]
runner = interpreter.get_signature_runner()
app = FastAPI()


def run_model(runner: Callable, input_path: str):
    prep = tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(input_path, target_size=(WIDTH, HEIGHT))
    )
    output = runner(**{INKEY: tf.constant(prep, shape=shape, dtype=tf.float32)})
    return int(round(output[OUTKEY][0][1], PRECISION) * EXP) / EXP


@app.post("/upload")
def upload(files: List[UploadFile] = File(...)):
    if runner == None:
        raise HTTPException(status_code=500, detail="Model was not loaded")

    results = {}
    for file in files:
        fd, path = tempfile.mkstemp()
        try:
            with os.fdopen(fd, "wb") as tmp:
                shutil.copyfileobj(file.file, tmp)
                results[file.filename] = run_model(runner, path)
        except Exception as e:
            is_format_error = isinstance(e, IOError)
            print(e)
            raise HTTPException(
                status_code=415 if is_format_error else 500,
                detail="Uploaded images were malformed"
                if is_format_error
                else "There was an error uploading the files",
            )
        finally:
            os.remove(path)
            file.file.close()

    return results
