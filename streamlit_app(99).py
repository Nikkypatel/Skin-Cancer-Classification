import streamlit as st
import PIL
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('SkinCancerModel_run_1.h5')

# Define the classes dictionary
classes = {
    0: ('akiec', 'actinic keratoses and intraepithelial carcinomae'),
    1: ('bcc', 'basal cell carcinoma'),
    2: ('bkl', 'benign keratosis-like lesions'),
    3: ('df', 'dermatofibroma'),
    4: ('nv', ' melanocytic nevi'),
    5: ('vasc', ' pyogenic granulomas and hemorrhage'),
    6: ('mel', 'melanoma'),
}

def predict_skin_cancer(image_path):
    # Load the image
    image = PIL.Image.open(image_path)
    image = image.resize((28, 28))

    # Convert the image to a NumPy array
    img = np.array(image)

    # Reshape the array if needed
    img = img.reshape(-1, 28, 28, 3)

    # Make a prediction using the loaded model
    result = model.predict(img)[0]
    result = result.tolist()

    # Get the predicted class
    max_prob = max(result)
    class_ind = result.index(max_prob)

    return classes[class_ind]

# Streamlit UI
st.title("Skin Cancer Classification")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    # Display the uploaded image
    image = PIL.Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image.", use_column_width=True)

    # Make prediction on the uploaded image
    result = predict_skin_cancer(uploaded_file)

    # Display the result
    st.subheader("Prediction:")
    st.write(result)