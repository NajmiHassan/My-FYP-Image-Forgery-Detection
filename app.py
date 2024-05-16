import streamlit as st
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from PIL import Image, ImageChops, ImageEnhance
import numpy as np
from keras.models import load_model


# Set page title and other configurations
st.set_page_config(page_title="PixelProbe", page_icon=None, layout='centered', initial_sidebar_state='auto')

# New caching function
@st.cache_data
def load_model_cached(model_path):
    model = load_model(model_path)
    return model

# Load the trained model using the cached function
model_path = "C:/Users/Najmi/Machine Learning FYP/saved model/image_forgery_detection_model.keras"
model = load_model_cached(model_path)

# Display the model summary
# st.text('Model Summary:')
# model_summary = []
# model.summary(print_fn=lambda x: model_summary.append(x))
# st.text('\n'.join(model_summary))

# Define the pages
PAGES = {
    "Home": "home",
    "About": "about",
    "Contact": "contact"
}

def send_email(sender, recipient, subject, message):
    # SMTP server configuration
    smtp_server = 'smtp.gmail.com'
    smtp_port = 587
    smtp_username = 'your_email@gmail.com'  # Your email
    smtp_password = 'your_password'  # Your email password

    # Email content
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = recipient
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain'))

    # Send the email
    server = smtplib.SMTP(smtp_server, smtp_port)
    server.starttls()
    server.login(smtp_username, smtp_password)
    server.sendmail(sender, recipient, msg.as_string())
    server.quit()

def convert_to_ela_image(path, quality):
    temp_filename = 'temp_file_name.jpg'
    image = Image.open(path).convert('RGB')
    image.save(temp_filename, 'JPEG', quality = quality)
    temp_image = Image.open(temp_filename)
    ela_image = ImageChops.difference(image, temp_image)
    extrema = ela_image.getextrema()
    max_diff = max([ex[1] for ex in extrema])
    if max_diff == 0:
        max_diff = 1
    scale = 255.0 / max_diff
    ela_image = ImageEnhance.Brightness(ela_image).enhance(scale)
    return ela_image

image_size = (224, 224)

def prepare_image(image_path):
    # Convert the image to ELA and resize it
    ela_image = convert_to_ela_image(image_path, 90).resize(image_size)
    # Convert the image to a numpy array and normalize pixel values
    image_array = np.array(ela_image) / 255.0
    # Reshape the array to add the batch dimension and match the input shape of the model
    image_array = image_array.reshape(-1, 224, 224, 3)
    return image_array


def predict_image(image_path):
    # Preprocess the image
    processed_image = prepare_image(image_path)
    # Make predictions using the loaded model
    prediction = model.predict(np.expand_dims(processed_image, axis=0))
    # Extract the predicted class
    predicted_class = np.argmax(prediction)
    return "Authentic" if predicted_class == 1 else "Tampered"

def main():
    st.sidebar.image('images/logo.png', use_column_width=True)
    
    page = "home"  
    
    if st.sidebar.button("Home"):
        page = "home"
    if st.sidebar.button("About "):
        page = "about"
    if st.sidebar.button("Contact"):
        page = "contact"
    
    if page == "home":
        st.title("🔍Image Detector")
        st.markdown("Unveil the truth in every image with PixelProbe 🕵️‍♂️🔍, the advanced image forgery detection system. Our innovative technology 🔬 peels back the layers of digital alterations to reveal the original, unaltered state. Trust in the clarity of authenticity with PixelProbe—where every image tells the real story 🖼️✨.")
        
        st.title("🖼️ Upload Your Image")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
        if uploaded_image is not None:
            # To See details
            file_details = {"FileName":uploaded_image.name,"FileType":uploaded_image.type,"FileSize":uploaded_image.size}
            st.write(file_details)
            
            # Save the uploaded image to a temporary file
            with open("temp_image.jpg", "wb") as f:
                f.write(uploaded_image.getbuffer())
            
            # Make prediction on the uploaded image
            prediction = predict_image("temp_image.jpg")
            st.write(f"The image is classified as: **{prediction}**")

            # Display the uploaded image
            st.image(uploaded_image, caption='Uploaded Image.', use_column_width=True)
                
        else:
            st.write("No image uploaded yet.")
        
    elif page == "about":
        st.title("About Me And Project")
        st.write("🌟 Meet our dedicated team of professionals who are passionate about their work and committed to providing the best service to our clients. 💼👩‍💼👨‍💼🔧📈🌟")
       

    elif page == "contact":
        st.title("Contact Us")
        st.write("Send us your queries and we'll get back to you as soon as possible.")
        
        # Contact form
        with st.form(key='contact_form'):
            sender_email = st.text_input("Your Email")
            subject = st.text_input("Subject")
            message = st.text_area("Message")
            submit_button = st.form_submit_button(label='Send')
            
            if submit_button and sender_email and subject and message:
                # Assuming 'your_email@example.com' is the email you want to receive messages at
                send_email(sender_email, 'your_email@example.com', subject, message)
                st.success("Your query has been sent. We will contact you soon.")

        # Apply red border to input fields
        st.markdown("""
            <style>
                div[data-testid="stTextInput"] input {
                    border: 2px solid red !important;
                    border-radius: 5px;
                    padding: 5px;
                }
                div[data-testid="stTextArea"] textarea {
                    border: 2px solid red !important;
                    border-radius: 5px;
                    padding: 5px;
                }
            </style>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
