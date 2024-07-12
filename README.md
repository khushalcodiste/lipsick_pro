



## Clone the repository:
```bash
git clone https://github.com/khushalcodiste/lipsick_pro.git
cd LipSick
```
## Create and activate the Anaconda environment:
```bash
conda env create -f environment.yml
conda activate LipSick
```
```
wget -P ./asserts https://github.com/Inferencer/LipSick/releases/download/v1PretrainedModels/pretrained_lipsick.pth
```
```
wget -P ./asserts https://github.com/Inferencer/LipSick/releases/download/v1PretrainedModels/output_graph.pb
```

### For the folder ./models

Please download shape_predictor_68_face_landmarks.dat using this [link](https://github.com/Inferencer/LipSick/releases/download/v1PretrainedModels/shape_predictor_68_face_landmarks.dat) and place the file in the folder ./models

```
wget -P ./models https://github.com/Inferencer/LipSick/releases/download/v1PretrainedModels/shape_predictor_68_face_landmarks.dat
```

### The folder structure for manually downloaded models
```bash
.
├── ...
├── asserts                        
│   ├── examples                   # A place to store inputs if not using gradio UI
│   ├── inference_result           # Results will be saved to this folder
│   ├── output_graph.pb            # The DeepSpeech model you manually download and place here
│   └── pretrained_lipsick.pth     # Pre-trained model you manually download and place here
│                   
├── models
│   ├── Discriminator.py
│   ├── LipSick.py
│   ├── shape_predictor_68_face_landmarks.dat  # Dlib Landmark tracking model you manually download and place here
│   ├── Syncnet.py
│   └── VGG19.py   
└── ...
```


```
git clone https://github.com/TencentARC/GFPGAN.git

cd GFPGAN

pip install basicsr

pip install facexlib

pip install -r requirements.txt
python setup.py develop

pip install realesrgan resampy fastapi uvicorn python_speech_features facexlib basicsr

wget -P "/content/lipsick_pro/experiments/pretrained_models" https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

cd ..

pip install --force-reinstall charset-normalizer==3.1.0
pip install onnxruntime-gpu


```



## Run the application:

```bash
python app.py
```

