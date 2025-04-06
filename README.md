This project implements a Structure from Motion (SfM) pipeline that reconstructs a 3D point cloud and estimates camera poses from a sequence of 2D images. The system leverages classical computer vision techniques, primarily using OpenCV for feature extraction, matching, triangulation, and pose estimation.

Key features of this pipeline include:
	•	Feature Detection and Matching using SIFT (Scale-Invariant Feature Transform)
	•	Camera Pose Estimation using Essential Matrix Decomposition
	•	3D Point Triangulation based on matched features
	•	(NEW IDEA!) 3D Point Cloud Denoising using L2 distance-based filtering and DBSCAN clustering


The pipeline is modularized into the following components:

main.py
├── calls: sfm_origin.py
│   └── uses: utils.py
├── input: image directory + camera intrinsic matrix (K.txt)
├── output: camera poses, 3D point cloud, PLY files


Project Structure 

project_root/
├── main.py
├── sfm_origin.py
├── utils.py
├── K.txt
├── images/
│   ├── 0000.JPG
│   ├── 0001.JPG
│   └── ...

RUN METHODS

1. install library 
```bash 
pip install opencv-python numpy matplotlib scikit-learn
```

2. opperation command 
```bash
python main.py --img_dir ./images --K_path ./K.txt
```
3. To operate the SfM system, you must first calibrate your camera using a checkerboard to obtain the intrinsic matrix K. Then, save the K.txt file in the same directory specified by the --path argument in the command line.

Example 
```bash
python main.py  --path "/Users/yangdongjae/Desktop/2025/DGIST/Computer_Vision/Assignment/PA_ 202522027_양동재/step1~5/data" --image_num 32 --output_name test --mode raw_clean --devic cpu
```

- 

### ⚙️ Command Line Arguments (`main.py`)

The `main.py` script includes several command-line arguments to control the SfM pipeline behavior. Below is a detailed description of each option:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--path` | `str` | `"./data/"` | Path to the folder containing input images. |
| `--image_num` | `int` | `3` | Number of images to use in the reconstruction. Useful for testing or partial reconstruction. |
| `--output_name` | `str` | `"Moon_pst3D"` | Prefix name for the output PLY file(s). Files will be saved as `Moon_pst3D.ply`, or as a sequence if `--sequence` is enabled. |
| `--mode` | `str` | `'clean'` | Specifies which pipeline variant to run. Available modes:<br>• `'raw'`: baseline pipeline without filtering or bundle adjustment<br>• `'raw_clean'`: raw + noise filtering<br>• `'ba'`: includes bundle adjustment<br>• `'clean'`: noise filtering + bundle adjustment |
| `--device` | `str` | `'cpu'` | Device to use for bundle adjustment. Options:<br>• `'cpu'`: Run on CPU<br>• `'cuda'`: Use GPU (if available)<br>• `'mps'`: Use Apple M1/M2 GPU (macOS only) |
| `--sequence` | `flag` | `False` | If enabled, the script will generate and save PLY point clouds for **all views**, not just the final one. Use this to visualize per-step reconstruction results. |

---

## 📌 Structure from Motion (SfM) 파이프라인 소개

이 프로젝트는 **Structure from Motion (SfM)** 파이프라인을 구현한 것으로, 2D 이미지 시퀀스로부터 카메라 포즈를 추정하고 3D 포인트 클라우드를 재구성합니다. 본 시스템은 전통적인 컴퓨터 비전 기법에 기반하며, OpenCV 라이브러리를 활용하여 특징 추출, 매칭, 삼각측량, 포즈 추정을 수행합니다.

### ✅ 주요 기능

- **SIFT**(Scale-Invariant Feature Transform)를 이용한 특징점 검출 및 매칭  
- Essential Matrix 분해를 이용한 **카메라 포즈 추정**  
- 매칭된 특징점을 이용한 **3D 포인트 삼각측량 (Triangulation)**  
- **(New Idea!)** L2 거리 기반 필터링 및 **DBSCAN 클러스터링**을 활용한 **3D 포인트 클라우드 노이즈 제거**

---

### 🔁 시스템 구성도
main.py
├── calls: sfm_origin.py
│   └── uses: utils.py
├── input: image directory + camera intrinsic matrix (K.txt)
├── output: camera poses, 3D point cloud, PLY files

---

### 📂 프로젝트 구조
project_root/
├── main.py
├── sfm_origin.py
├── utils.py
├── K.txt
├── images/
│   ├── 0000.JPG
│   ├── 0001.JPG
│   └── …

---

### 🚀 실행 방법

1. 필수 라이브러리 설치:
```bash
pip install opencv-python numpy matplotlib scikit-learn
```

2. 실행 명령어: 
```bash
python main.py --path ./images --image_num 10 --output_name result --mode clean --device cpu
```
3. SfM 시스템을 실행하기 위해서는 체커보드로 카메라 보정을 수행하여 Intrinsic K 행렬을 얻은 후, 해당 값을 K.txt 파일로 저장하여 커맨드라인에서 --path로 지정한 디렉토리에 위치시켜야 합니다.

예시 
```bash
python main.py --path "/Users/yangdongjae/Desktop/2025/DGIST/Computer_Vision/Assignment/PA_202522027_양동재/step1~5/data" --image_num 32 --output_name test --mode raw_clean --device cpu
```
### ⚙️ 명령줄 인자 설명 (`main.py`)

| 인자 이름 | 타입 | 기본값 | 설명 |
|-----------|------|--------|------|
| `--path` | `str` | `"./data/"` | 입력 이미지들이 저장된 폴더 경로입니다. |
| `--image_num` | `int` | `3` | 사용할 이미지 개수입니다. 테스트용이나 부분 재구성 시 유용합니다. |
| `--output_name` | `str` | `"Moon_pst3D"` | 출력될 PLY 파일의 이름 접두사입니다. 예: `Moon_pst3D.ply` 또는 `--sequence` 사용 시 여러 파일 생성됩니다. |
| `--mode` | `str` | `'clean'` | 사용할 파이프라인 모드를 설정합니다:<br>• `'raw'`: 필터링/BA 없이 기본 파이프라인 실행<br>• `'raw_clean'`: raw + 노이즈 필터링<br>• `'ba'`: 번들 조정 포함<br>• `'clean'`: 노이즈 필터링 + 번들 조정 포함 |
| `--device` | `str` | `'cpu'` | 번들 조정을 수행할 디바이스 설정:<br>• `'cpu'`: CPU 사용<br>• `'cuda'`: NVIDIA GPU 사용<br>• `'mps'`: Apple Silicon GPU 사용(macOS) |
| `--sequence` | `flag` | `False` | 설정 시, 전체 뷰에 대한 PLY 결과를 순차적으로 저장합니다.|