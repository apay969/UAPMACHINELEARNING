import streamlit as st
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
import zipfile, os, shutil
import matplotlib.pyplot as plt

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="UAP Image Classification",
    page_icon="🧠",
    layout="wide"
)

# ===============================
# GLOBAL CONFIG
# ===============================
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
CHANNELS = 3
TEMP_DIR = "temp"

TRAIN_SPLIT = 0.7
VALIDATION_SPLIT = 0.2
TEST_SPLIT = 0.1

# ===============================
# HEADER
# ===============================
st.markdown("""
# 🧠 UAP — Klasifikasi Citra  
### CNN • ResNet50 • VGG16  
Aplikasi interaktif *training & evaluasi model deep learning*
""")
st.divider()

# ===============================
# SIDEBAR
# ===============================
st.sidebar.header("⚙️ Konfigurasi")

uploaded_zip = st.sidebar.file_uploader("📦 Upload Dataset (ZIP)", type="zip")

model_choice = st.sidebar.radio(
    "Pilih Model",
    ["CNN", "ResNet50", "VGG16"]
)

EPOCHS = st.sidebar.slider("Epoch", 1, 30, 5)
BATCH_SIZE = st.sidebar.selectbox("Batch Size", [8, 16, 32])

train_button = st.sidebar.button("🚀 Train Model", use_container_width=True)

# ===============================
# DATASET UTIL
# ===============================
def extract_zip(uploaded_zip):
    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)
    os.makedirs(TEMP_DIR)
    with zipfile.ZipFile(uploaded_zip, 'r') as zip_ref:
        zip_ref.extractall(TEMP_DIR)
    return TEMP_DIR

def get_dataset_partition_tf(
    ds,
    train_split=TRAIN_SPLIT,
    val_split=VALIDATION_SPLIT,
    test_split=TEST_SPLIT,
    shuffle=True,
    shuffle_size=10000
):
    ds_size = len(ds)
    if shuffle:
        ds = ds.shuffle(shuffle_size, seed=12)

    train_size = int(train_split * ds_size)
    val_size = int(val_split * ds_size)

    train_ds = ds.take(train_size)
    val_ds = ds.skip(train_size).take(val_size)
    test_ds = ds.skip(train_size).skip(val_size)

    return train_ds, val_ds, test_ds

def load_dataset_custom(dataset_path):
    full_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_path,
        shuffle=True,
        image_size=(IMAGE_HEIGHT, IMAGE_WIDTH),
        batch_size=BATCH_SIZE
    )

    class_names = full_ds.class_names
    train_ds, val_ds, test_ds = get_dataset_partition_tf(full_ds)

    train_ds = train_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    val_ds   = val_ds.cache().shuffle(1000).prefetch(tf.data.AUTOTUNE)
    test_ds  = test_ds.cache().prefetch(tf.data.AUTOTUNE)

    return train_ds, val_ds, test_ds, class_names

# ===============================
# PREPROCESSING
# ===============================
rescale = keras.Sequential([
    layers.Rescaling(1.0 / 255)
])

# ===============================
# MODEL FACTORY (TANPA AUGMENTATION)
# ===============================
def build_model(name, num_classes):

    inputs = layers.Input(shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS))
    x = rescale(inputs)

    if name == "ResNet50":
        base = keras.applications.ResNet50(
            include_top=False,
            pooling="avg",
            input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
            weights="imagenet"
        )
        base.trainable = False

        x = base(x)
        x = layers.Dense(512, activation="relu")(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs)

    elif name == "VGG16":
        base = keras.applications.VGG16(
            include_top=False,
            pooling="avg",
            input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, CHANNELS),
            weights="imagenet"
        )
        for layer in base.layers[:-10]:
            layer.trainable = False

        x = base(x)
        x = layers.Dense(64, activation="relu")(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs)

    else:  # CNN
        x = layers.Conv2D(32, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(64, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Conv2D(128, 3, activation="relu")(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation="relu")(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)

        model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=Adam(0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    return model

# ===============================
# MAIN
# ===============================
if uploaded_zip:
    with st.spinner("📂 Menyiapkan dataset..."):
        path = extract_zip(uploaded_zip)
        dataset_dir = os.path.join(path, os.listdir(path)[0])
        train_ds, val_ds, test_ds, class_names = load_dataset_custom(dataset_dir)

    # Dataset info
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Jumlah Kelas", len(class_names))
    c2.metric("Train Split", "70%")
    c3.metric("Val Split", "20%")
    c4.metric("Test Split", "10%")

    st.divider()

    if train_button:
        model = build_model(model_choice, len(class_names))

        with st.spinner("🧠 Training model berjalan..."):
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=EPOCHS,
                verbose=1
            )

        st.success("Training selesai 🎉")

        # ===============================
        # TABS
        # ===============================
        tab1, tab2, tab3 = st.tabs(
            ["📊 Ringkasan", "📈 Grafik", "🧪 Evaluasi"]
        )

        with tab1:
            col1, col2 = st.columns(2)
            col1.metric("Train Accuracy", f"{history.history['accuracy'][-1]*100:.2f}%")
            col2.metric("Val Accuracy", f"{history.history['val_accuracy'][-1]*100:.2f}%")

        with tab2:
            fig, ax = plt.subplots(1, 2, figsize=(16, 5))
            ax[0].plot(history.history['accuracy'], label="Train")
            ax[0].plot(history.history['val_accuracy'], label="Val")
            ax[0].set_title("Accuracy")
            ax[0].legend()

            ax[1].plot(history.history['loss'], label="Train")
            ax[1].plot(history.history['val_loss'], label="Val")
            ax[1].set_title("Loss")
            ax[1].legend()

            st.pyplot(fig)

        with tab3:
            loss, acc = model.evaluate(test_ds, verbose=0)
            st.metric("Test Accuracy", f"{acc*100:.2f}%")
            st.metric("Test Loss", f"{loss:.4f}")

else:
    st.info("⬅️ Upload dataset ZIP melalui sidebar untuk memulai")
