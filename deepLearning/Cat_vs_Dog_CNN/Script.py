from pyspark.sql import SparkSession
import tensorflow as tf
import os

# -------------------------------
# Start Spark Session
# -------------------------------
spark = SparkSession.builder.appName("CatsVsDogs_TF").getOrCreate()

# Dataset paths
train_dir = "/kaggle/input/dogs-vs-cats/train/train"
test_dir  = "/kaggle/input/dogs-vs-cats/test/test"

# -------------------------------
# Load file paths with Spark
# -------------------------------
train_df = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.jpg") \
    .load(train_dir)

test_df = spark.read.format("binaryFile") \
    .option("pathGlobFilter", "*.jpg") \
    .load(test_dir)

print("Train count:", train_df.count())
print("Test count :", test_df.count())

# -------------------------------
# Convert Spark paths -> TensorFlow Dataset
# -------------------------------
def preprocess_image(path, label):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [128,128])
    img = img / 255.0
    return img, label

def get_label_from_filename(path):
    filename = tf.strings.split(path, os.sep)[-1]
    label = tf.strings.split(filename, ".")[0]
    return tf.cond(tf.equal(label, "cat"), lambda: 0, lambda: 1)

train_paths = [row.path for row in train_df.collect()]
test_paths  = [row.path for row in test_df.collect()]

train_ds = tf.data.Dataset.from_tensor_slices(train_paths)
train_ds = train_ds.map(lambda x: preprocess_image(x, get_label_from_filename(x)),
                        num_parallel_calls=tf.data.AUTOTUNE)
train_ds = train_ds.shuffle(1000).batch(32).prefetch(tf.data.AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices(test_paths)
val_ds = val_ds.map(lambda x: preprocess_image(x, get_label_from_filename(x)),
                    num_parallel_calls=tf.data.AUTOTUNE)
val_ds = val_ds.batch(32).prefetch(tf.data.AUTOTUNE)

# -------------------------------
# Define CNN model
# -------------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# -------------------------------
# Train the model
# -------------------------------
history = model.fit(train_ds, validation_data=val_ds, epochs=3)

print("✅ Training complete")

# -------------------------------
# Save model
# -------------------------------
model.save("cats_vs_dogs_model.h5")
print("✅ Model saved as cats_vs_dogs_model.h5")

spark.stop()
