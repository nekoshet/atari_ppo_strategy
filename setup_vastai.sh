apt install python3.10-venv
python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements/requirements_tal.txt
apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
