
python -m venv venv
pip install chess numpy onnx onnxruntime pyinstaller
pyinstaller -F -n dinora .\dinora\__main__.py
