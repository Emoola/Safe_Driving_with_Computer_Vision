# Safe_Driving
1. Git clone tensorflow folder: 
``` bash
git clone https://github.com/tensorflow/models.git
```
2. Download VS 2019 community and install c++ compiler
3. Download CUDA Toolkit 10.0 from https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork
4. Download cuDNN for CUDA 10.0 from https://developer.nvidia.com/rdp/cudnn-download
5. Copy files from cuda folder under cuDNN to CUDA:
    5.1 '<downloadpath>\cuda\bin\cudnn64_7.dll' to 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin\'
    5.2 '<downloadpath>\cuda\include\cudnn.h' to 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include\'
    5.3 '<downloadpath>\cuda\lib\x64\cudnn.lib' to 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64\'
6. Install the COCO API: 
    ``` bash
    pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
    ```
7. Install Protobuf Compilation:
    7.1 Download protoc-3.4.0-win32.zip from https://github.com/protocolbuffers/protobuf/releases/tag/v3.4.0 
    7.2 move bin/protoc.exe to models/research directory 
    7.3  
    ``` bash
    # From /models/research/
    protoc object_detection/protos/*.proto --python_out=.
    ```
8. Add Libraries to PYTHONPATH: 
    ``` bash
    # From /models/research/
    set PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
    ```
