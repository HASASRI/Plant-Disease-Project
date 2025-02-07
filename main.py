import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return predictions  # return the raw predictions

# Sidebar
st.sidebar.title("Plant Disease Detection System for Sustainable Agriculture")
app_mode = st.sidebar.selectbox("Select Page", ["HOME", "DISEASE RECOGNITION", "INFORMATION"])

# Display image using streamlit
img = Image.open("Diseases.png")
st.image(img)

# Main Page
if app_mode == "HOME":
    st.markdown("<h2 style='text-align: center;'>Welcome To My Project,</h2><h1 style='text-align: center;'><i>Plant Disease Detection System for Sustainable Agriculture</i></h1>", unsafe_allow_html=True)

# Prediction Page
elif app_mode == "DISEASE RECOGNITION":
    st.header("Plant Disease Detection System for Sustainable Agriculture")
    test_image = st.file_uploader("Choose an Image:")
    if st.button("Show Image"):
        st.image(test_image, width=400)
    # Predict button
    if st.button("Predict"):
        st.snow()
        st.write("Our Prediction is:")
        predictions = model_prediction(test_image)
        result_index = np.argmax(predictions)
        confidence_score = np.max(predictions)
        # Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                      'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                      'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                      'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                      'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                      'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                      'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        plant_name, disease_name = class_name[result_index].split('___')
        st.write(f"Model is Predicting\n  It's a {plant_name} Plant\n\n\nDisease Name: {disease_name} , Confidence score: {confidence_score * 100:.2f}%")

        # Display detailed information about the predicted disease
        disease_info = {
            'Apple___Apple_scab': "Apple scab is a fungal disease that causes dark, scabby lesions on leaves, fruit, and twigs.",
            'Apple___Black_rot': "Black rot is a fungal disease that causes black, sunken lesions on fruit and cankers on branches.",
            'Apple___Cedar_apple_rust': "Cedar apple rust is a fungal disease that causes yellow-orange spots on leaves and fruit.",
            'Apple___healthy': "The apple plant is healthy.",
            'Blueberry___healthy': "The blueberry plant is healthy.",
            'Cherry_(including_sour)___Powdery_mildew': "Powdery mildew is a fungal disease that causes a white, powdery coating on leaves and stems.",
            'Cherry_(including_sour)___healthy': "The cherry plant is healthy.",
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': "Cercospora leaf spot is a fungal disease that causes gray to brown lesions on leaves.",
            'Corn_(maize)___Common_rust_': "Common rust is a fungal disease that causes reddish-brown pustules on leaves.",
            'Corn_(maize)___Northern_Leaf_Blight': "Northern leaf blight is a fungal disease that causes large, gray to tan lesions on leaves.",
            'Corn_(maize)___healthy': "The corn plant is healthy.",
            'Grape___Black_rot': "Black rot is a fungal disease that causes black, sunken lesions on fruit and cankers on branches.",
            'Grape___Esca_(Black_Measles)': "Esca (Black Measles) is a fungal disease that causes dark streaks on leaves and fruit.",
            'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)': "Leaf blight is a fungal disease that causes brown, necrotic lesions on leaves.",
            'Grape___healthy': "The grape plant is healthy.",
            'Orange___Haunglongbing_(Citrus_greening)': "Citrus greening is a bacterial disease that causes yellowing of leaves and misshapen fruit.",
            'Peach___Bacterial_spot': "Bacterial spot is a bacterial disease that causes dark, water-soaked lesions on leaves and fruit.",
            'Peach___healthy': "The peach plant is healthy.",
            'Pepper,_bell___Bacterial_spot': "Bacterial spot is a bacterial disease that causes dark, water-soaked lesions on leaves and fruit.",
            'Pepper,_bell___healthy': "The bell pepper plant is healthy.",
            'Potato___Early_blight': "Early blight is a fungal disease that causes dark, concentric lesions on leaves and stems.",
            'Potato___Late_blight': "Late blight is a fungal disease that causes dark, water-soaked lesions on leaves and stems.",
            'Potato___healthy': "The potato plant is healthy.",
            'Raspberry___healthy': "The raspberry plant is healthy.",
            'Soybean___healthy': "The soybean plant is healthy.",
            'Squash___Powdery_mildew': "Powdery mildew is a fungal disease that causes a white, powdery coating on leaves and stems.",
            'Strawberry___Leaf_scorch': "Leaf scorch is a fungal disease that causes dark, necrotic lesions on leaves.",
            'Strawberry___healthy': "The strawberry plant is healthy.",
            'Tomato___Bacterial_spot': "Bacterial spot is a bacterial disease that causes dark, water-soaked lesions on leaves and fruit.",
            'Tomato___Early_blight': "Early blight is a fungal disease that causes dark, concentric lesions on leaves and stems.",
            'Tomato___Late_blight': "Late blight is a fungal disease that causes dark, water-soaked lesions on leaves and stems.",
            'Tomato___Leaf_Mold': "Leaf mold is a fungal disease that causes yellow spots on leaves.",
            'Tomato___Septoria_leaf_spot': "Septoria leaf spot is a fungal disease that causes small, dark lesions on leaves.",
            'Tomato___Spider_mites Two-spotted_spider_mite': "Spider mites are tiny arachnids that cause stippling and webbing on leaves.",
            'Tomato___Target_Spot': "Target spot is a fungal disease that causes dark, concentric lesions on leaves and fruit.",
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus': "Tomato yellow leaf curl virus is a viral disease that causes yellowing and curling of leaves.",
            'Tomato___Tomato_mosaic_virus': "Tomato mosaic virus is a viral disease that causes mottling and distortion of leaves.",
            'Tomato___healthy': "The tomato plant is healthy."
        }
        st.write("**Disease Information:**")
        st.write(disease_info.get(class_name[result_index], "No information available for this disease."))

        # Determine plant condition based on confidence score
        condition = "Healthy" if confidence_score > 0.8 else "Moderate" if confidence_score > 0.4 else "Unhealthy"
        condition_color = "green" if condition == "Healthy" else "yellow" if condition == "Moderate" else "red"

        # Display histogram visualization
        st.write("**Plant Condition:**")
        plt.figure(figsize=(10, 5))
        plt.hist(predictions[0], bins=10, color=condition_color, edgecolor='black')
        plt.xlabel('Confidence Score')
        plt.ylabel('Frequency')
        plt.title('Plant Condition: {}'.format(condition))
        st.pyplot(plt)

# Information Page
elif app_mode == "INFORMATION":
    st.header("Information on Plant Diseases")
    st.write("""
    ### Apple Diseases
    - **Apple Scab**: A fungal disease that causes dark, scabby lesions on leaves, fruit, and twigs.
    - **Black Rot**: A fungal disease that causes black, sunken lesions on fruit and cankers on branches.
    - **Cedar Apple Rust**: A fungal disease that causes yellow-orange spots on leaves and fruit.

    ### Blueberry Diseases
    - **Powdery Mildew**: A fungal disease that causes a white, powdery coating on leaves and stems.

    ### Cherry Diseases
    - **Powdery Mildew**: A fungal disease that causes a white, powdery coating on leaves and stems.

    ### Corn Diseases
    - **Cercospora Leaf Spot**: A fungal disease that causes gray to brown lesions on leaves.
    - **Common Rust**: A fungal disease that causes reddish-brown pustules on leaves.
    - **Northern Leaf Blight**: A fungal disease that causes large, gray to tan lesions on leaves.

    ### Grape Diseases
    - **Black Rot**: A fungal disease that causes black, sunken lesions on fruit and cankers on branches.
    - **Esca (Black Measles)**: A fungal disease that causes dark streaks on leaves and fruit.
    - **Leaf Blight**: A fungal disease that causes brown, necrotic lesions on leaves.

    ### Orange Diseases
    - **Citrus Greening**: A bacterial disease that causes yellowing of leaves and misshapen fruit.

    ### Peach Diseases
    - **Bacterial Spot**: A bacterial disease that causes dark, water-soaked lesions on leaves and fruit.

    ### Pepper Diseases
    - **Bacterial Spot**: A bacterial disease that causes dark, water-soaked lesions on leaves and fruit.

    ### Potato Diseases
    - **Early Blight**: A fungal disease that causes dark, concentric lesions on leaves and stems.
    - **Late Blight**: A fungal disease that causes dark, water-soaked lesions on leaves and stems.

    ### Strawberry Diseases
    - **Leaf Scorch**: A fungal disease that causes dark, necrotic lesions on leaves.

    ### Tomato Diseases
    - **Bacterial Spot**: A bacterial disease that causes dark, water-soaked lesions on leaves and fruit.
    - **Early Blight**: A fungal disease that causes dark, concentric lesions on leaves and stems.
    - **Late Blight**: A fungal disease that causes dark, water-soaked lesions on leaves and stems.
    - **Leaf Mold**: A fungal disease that causes yellow spots on leaves.
    - **Septoria Leaf Spot**: A fungal disease that causes small, dark lesions on leaves.
    - **Spider Mites**: Tiny arachnids that cause stippling and webbing on leaves.
    - **Target Spot**: A fungal disease that causes dark, concentric lesions on leaves and fruit.
    - **Tomato Yellow Leaf Curl Virus**: A viral disease that causes yellowing and curling of leaves.
    - **Tomato Mosaic Virus**: A viral disease that causes mottling and distortion of leaves.

    ### General Prevention and Treatment
    - **Prevention**: Use disease-resistant varieties, practice crop rotation, and maintain proper spacing and sanitation.
    - **Treatment**: Apply appropriate fungicides or bactericides, remove and destroy infected plant parts, and ensure proper watering and fertilization.
    """)