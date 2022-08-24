import tensorflow as tf

def main():
  model = tf.keras.models.load_model('model.h5')
  converter = tf.lite.TFLiteConverter.from_keras_model(model)
  tflite_model = converter.convert()
  open("converted_model.tflite", "wb").write(tflite_model)

if __name__ == "__main__":
  main()
