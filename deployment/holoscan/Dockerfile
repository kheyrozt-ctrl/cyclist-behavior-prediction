FROM nvcr.io/nvidia/clara-holoscan/holoscan:v3.11.0-cuda12-dgpu

RUN python3 -m pip install \
      --disable-pip-version-check \
      --root-user-action=ignore \
      "numpy<2" \
      "mediapipe==0.10.21" \
      "onnxruntime>=1.27,<2" \
    && python3 -m pip uninstall -y opencv-python opencv-contrib-python \
    && python3 -m pip install \
      --disable-pip-version-check \
      --root-user-action=ignore \
      "opencv-contrib-python-headless<4.12"

WORKDIR /workspace
