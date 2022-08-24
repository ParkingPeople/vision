import numpy as np
import tensorflow as tf
from rich import print

from fastapi import FastAPI, File, UploadFile

UPLOAD_PATH = "uploads"

TFLITE_FILE_PATH = "converted_model.tflite"

INKEY = "input_1"
OUTKEY = "dense"

WIDTH = 224
HEIGHT = 224
CHANNELS = 3

PRECISION = 4
EXP = 10**PRECISION


app = FastAPI()
runner = None


def run_model(runner, input_path):
    prep = tf.keras.preprocessing.image.img_to_array(
        tf.keras.preprocessing.image.load_img(input_path, target_size=(WIDTH, HEIGHT))
    )
    output = runner(
        **{
            INKEY: tf.constant(
                prep, shape=(1, WIDTH, HEIGHT, CHANNELS), dtype=tf.float32
            )
        }
    )
    return int(round(output[OUTKEY][0][1], PRECISION) * EXP) / EXP


def main():
    # load model
    interpreter = tf.lite.Interpreter(TFLITE_FILE_PATH)
    siglist = interpreter.get_signature_list()
    print(f"{siglist=}")

    global runner
    runner = interpreter.get_signature_runner()


@app.get("/")
async def root():
    return {"message": "Hello World"}


if __name__ == "__main__":
    main()
