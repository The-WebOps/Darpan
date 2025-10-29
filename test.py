import tensorflow as tf

gpu_available = tf.config.list_physical_devices('GPU')

if gpu_available:
    print("GPU is enabled and available.")
    print("GPU details:", gpu_available)
else:
    print("GPU is not enabled or available.")