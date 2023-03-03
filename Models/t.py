# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.4
# ---

import tensorflow as tf
from tensorflow.lite.python import interpreter as interpreter_wrapper

model_path="t.tflite"
model_input_shape=(608,608)
interpreter = interpreter_wrapper.Interpreter(model_path=model_path)
interpreter.allocate_tensors()
#model_format = 'TFLITE'

#Ehsan input shape correctness
input_details = interpreter.get_input_details()
input_shape = input_details[0]['shape']
input_shape[1] = model_input_shape[0]
input_shape[2] = model_input_shape[1]
interpreter.resize_tensor_input(0, input_shape)
interpreter.allocate_tensors()

# Convert the TensorFlow Lite model to a Keras .h5 model
input_tensors = interpreter.get_input_details()
output_tensors = interpreter.get_output_details()

# Define the input and output shapes of the Keras model
input_shape = (608, 608, 3)
output_shape = [(1, 1, 1, 255), (1, 19, 19, 255), (1, 1, 1, 255)]

@tf.function(input_signature=[tf.TensorSpec(shape=input_shape[i], dtype=tf.float32) for i in range(len(input_shape))])
def tflite_inference(*inputs):
    interpreter.set_tensor(interpreter.get_input_details()[0]['index'], inputs[0])
    interpreter.invoke()
    return tuple(interpreter.get_tensor(output_tensors[i]['index']) for i in range(len(output_tensors)))

# Convert the TensorFlow Lite model to a Keras .h5 model
def m():
    output_tensors = interpreter.get_output_details()
    keras_model = tf.keras.models.Sequential([tf.keras.layers.Lambda(tflite_inference, input_shape=input_shape, output_shape=output_shape)])
    keras_model.save("t.h5")

if __name__ == '__main__':
    m()
