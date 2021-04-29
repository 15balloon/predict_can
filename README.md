# Predict can
> 2020-2학기 '인공지능' 수업에서 진행한 프로젝트   

Predict 18 kinds of beverage cans and informs by voice.   
18가지 캔음료를 분류하여 음성으로 알려준다.   

---

## About the Project
![application](https://user-images.githubusercontent.com/81695614/116588165-2bad6e80-a956-11eb-96ab-6047707c8721.jpg)

When you load the photo and click the "Predict" button, the learned model predicts the type of drink through voice and text.   
사진 파일을 불러와서 'Predict' 버튼을 클릭하면 학습한 모델이 음료수의 종류를 음성과 텍스트로 알려준다.

---

## Built With
* [Python](https://www.python.org/)
* [OpenCV](https://opencv.org/)
* [Imgaug](https://github.com/aleju/imgaug)
* [Tensorflow](tensorflow.org/)
* gTTS
* Tkinter

---

## Image Preprocessing

### Image Augmentation
>img_aug.py

훈련에 사용할 파일 수가 모자라서 하나의 이미지를 15개의 증폭 방법을 사용해 증폭시켰다.

### Image Resize
>img_prep.py

증폭시킨 이미지는 훈련에 사용할 수 있도록 64x64 크기로 조정하였다.   
색깔 값은 255로 나누어 0~1사이의 값을 가지도록 했다.   
라벨을 붙여서 .npy파일로 저장하였다.

---

## Modeling
>learning.py

CNN 방식을 사용하였다.   
활성화 함수로는 Relu를 사용하였고 마지막에는 Softmax를 사용하였다.   
손실함수로는 CrossEntropy를 사용하였다.   
EarlyStopping과 Dropout을 적용하여 모델이 과적합되지 않도록 했다.
학습한 결과는 .h5파일에 저장하였다.

---

## Application
>speech.py   
>gui.py

Tkinter로 GUI를 구성했다.   
불러온 사진 파일은 예측을 진행할 수 있도록 전처리를 하였다.   
.h5파일을 불러와서 이미지 예측을 진행하였다.   
gTTS를 사용하여 예측 결과를 음성으로 알려줄 수 있게 했다.
