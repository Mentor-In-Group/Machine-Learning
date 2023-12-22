import tensorflow as tf

# Load the saved Keras model
model = tf.keras.models.load_model('my_model.h5')

# Convert the Keras model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('recommendation_model.tflite', 'wb') as f:
    f.write(tflite_model)
