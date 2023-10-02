import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
print("ok")
for gpu in gpus:
    print("Name:", gpu.name, "  Type:", gpu.device_type)

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))


import tensorflow as tf
visible_devices = tf.config.get_visible_devices()
for devices in visible_devices:
  print(devices)