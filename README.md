# 🚀 Face Detection with Real-Time Emotion Recognition  
Detect faces and recognize emotions in real-time using OpenCV and a deep learning model.  

## 📌 Features  
- **Real-time Face Detection** using OpenCV  
- **Emotion Recognition** using a pre-trained deep learning model  
- **Live Camera Feed Processing**  
- **Optimized Model Loading** with Git LFS  

---

## 🚀 Installation & Setup  

### 1️⃣ Clone the Repository  
  ```sh
  git clone https://github.com/deepakQE/Face-Detection.git
  cd Face-Detection
  ```

### 2️⃣ Install Dependencies  
Ensure you have Python installed, then run:  
  ```sh
  pip install -r requirement.txt
  ```

### 3️⃣ Download Large Model Files (Git LFS)  
Since the model file is stored using Git LFS, install and pull it:  
  ```sh
  git lfs install
  git lfs pull
  ```

### 4️⃣ Run the Application  
  ```sh
  python app.py
  ```

## 🛠 Tech Stack  
- **Python** 🐍  
- **OpenCV** 🎥  
- **TensorFlow/Keras** 🤖  
- **Git LFS** for large file handling
  

## 🔧 Git LFS Setup (For Uploading Large Files)  
- If you're making changes and need to upload a large model, ensure Git LFS is tracking `.h5` files:  
  ```sh
     git lfs track "*.h5"
     echo "*.h5 filter=lfs diff=lfs merge=lfs" >> .gitattributes
     git add .gitattributes
  ```


- Then commit and push as usual:
  ```sh
   git add best_emotion_recognition_model.h5
   git commit -m "Added updated emotion recognition model"
   git push origin main
  ```

## 📷 Sample Output:
- Uploading Soon


## 🤝 Contributing:
- If you’d like to contribute, fork the repo and submit a pull request. Suggestions and improvements are always welcome!


## 📜 License:
- This project is licensed under the MIT License.


## 📬 Contact:
- For any issues, feel free to open an issue in the repository or reach out via email.


 ***⭐ Don't forget to star the repo if you like this project! ⭐***



