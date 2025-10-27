🩺✨ Skin Disease Classifier using Deep Learning

🌟 Overview

The Skin Disease Classifier is an intelligent deep learning web app 🧠 that classifies various skin diseases 🩹 using image-based analysis.
By uploading a photo of the affected area, 
users can instantly receive predictions powered by a Convolutional Neural Network (CNN) trained on real medical data.

This project supports early disease detection 🔍 and aims to aid dermatological diagnosis in a fast, accessible, and interactive way!




🎯 Objectives

🎨 Create a clean and user-friendly interface for image uploads.

⚡ Classify different skin diseases using CNN models.

📈 Display prediction results with confidence levels.

💬 Provide accurate, real-time analysis with intuitive visuals.

🧠 Model Architecture

The classifier is based on a Convolutional Neural Network (CNN) trained on preprocessed dermatology image datasets.


Architecture Highlights:


📥 Input: RGB skin image (224×224 px)

🔄 Layers: Convolutional + MaxPooling + Dropout

🧩 Output: Predicted disease label + confidence score

⚙️ Optimizer: Adam

📊 Loss: Categorical Crossentropy

⚙️ Tech Stack
💻 Layer	

🧩 Technologies Used


Frontend	Streamlit,
HTML, 
CSS
Backend	TensorFlow,
Keras,
NumPy,
PIL,
JSON

Model	CNN trained for skin disease classification

Environment	Python 3.x

Deployment	Streamlit Cloud / Heroku / Localhost


📦 Installation and Setup

🔹 1. Clone the Repository
git clone https://github.com/JeyapriyaG/skin-disease-classifier.git
cd skin-disease-classifier

🔹 2. Install Dependencies
pip install -r requirements.txt

🔹 3. Run the Streamlit App
streamlit run app/streamlit_app.py

🔹 4. Upload and Classify

Simply upload an image 📸 and let the AI predict the disease within seconds!

🗂️ Project Structure

skin-disease-classifier/

│
├── app/

│   ├── streamlit_app.py        # 🌈 Streamlit main UI

│   ├── static/                 # 🎨 Backgrounds, icons, etc.

│   └── models/
│       └── final_model.h5      # 🧠 Trained CNN model
│
├── dataset/
│   ├── train/                  # 🧾 Training images
│   └── test/                   # 🧪 Test images
│

├── requirements.txt

├── README.md
└── class_indices.json          # 🔢 Class label mappings


🖥️ Features

✨ Upload & preview skin images.

🤖 Predicts disease type and confidence level.

🌈 Light theme with animated glowing particles.

🧾 Displays previously tested images.

💬 Simple, minimal, and magical UI — like your story app! 🌟


🧾 Example Output

🖼️ Input Image	🧬 Predicted Disease	🎯 Confidence

acne.jpg	          Acne Vulgaris        	94.5%

eczema.png	         Eczema	               91.3%


🔮 Future Enhancements

🚀 Add disease descriptions and treatment guidelines.

🌍 Multi-language support.

📱 Mobile-friendly design.

📸 Integrate real-time camera capture.

💾 Include cloud storage for medical reports.

🧑‍💻 Author

👩‍💻 Jeyapriya G
💡 Passionate about AI, ML, and Deep Learning

📜 License

This project is licensed under the MIT License — free to use, modify, and share.

💬 Acknowledgements

🙏 Special thanks to:

🧠 TensorFlow & Keras open-source communities

💻 Streamlit for the interactive UI framework

📚 Researchers contributing to skin disease datasets

🌍 The global AI & medical imaging community

💖 "AI for health — empowering early detection, saving lives." 🩷
