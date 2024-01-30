import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
import numpy as np

# Load the trained model
model_path = r"C:\Users\castr\OneDrive\Desktop\Image Classification\BC.h5"
model = load_model(model_path, compile=False)

# Define the labels
labels = {
    0: 'Crowned Pigeon',
    1: 'Green Imperial-Pigeon',
    2: 'Island Thrush',
    3: 'Philippine Coucal',
    4: 'Philippine Cuckoo-Dove',
    5: 'Philippine Eagle',
    6: 'Philippine Hawk-Eagle',
    7: 'Philippine Serpent Eagle',
    8: 'Philippine Turtle Dove',
    9: 'Philippine Wood Pigeon',
    10: 'Class_10',
    11: 'Class_11',
}

def processed_img(img_path, model, lab):
    img = load_img(img_path, target_size=(224, 224, 3))
    img = img_to_array(img)
    img = img / 255
    img = np.expand_dims(img, [0])
    answer = model.predict(img)
    y_class = answer.argmax(axis=-1)
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    y = int(y)
    res = lab[y]
    print(res)
    return res

def run():
    img1 = Image.open(r"C:\Users\castr\OneDrive\Desktop\Image Classification\logo1.png.png")
    img1 = img1.resize((350, 350))
    st.image(img1, use_column_width=False)
    st.title("Birds Species Classification")
    st.markdown('''<h4 style='text-align: left; color: #d73b5c;'>* Data is based "10 Bird Species also see 70 Sports Dataset"</h4>''',
                unsafe_allow_html=True)

    # File uploader
    img_file = st.file_uploader("Choose an Image of Bird", type=["jpg", "png"])

    if img_file is not None:
        st.image(img_file, use_column_width=False)
        save_image_path = r"C:\Users\castr\OneDrive\Desktop\Image Classification\upload_images" + img_file.name
        with open(save_image_path, "wb") as f:
            f.write(img_file.getbuffer())

        # Provide feedback to the user
        st.success(f"Image '{img_file.name}' uploaded and saved successfully at: {save_image_path}")

    if st.button("Predict"):
        result = processed_img(save_image_path, model=model, lab=labels)
        # Display the predicted bird species
        st.success("Predicted Bird is: " + result)

run()