#  How to Use
### Installation
Clone and change to repository
```bash
git clone https://github.com/LuisBarbosa02/Detection-Tracking-and-Speed-Estimation.git
cd Detection-Tracking-and-Speed-Estimation
```
[Optional] Create and activate virtual environment
```bash
python3.12 -m venv venv
source venv/bin/activate
```
Install dependencies
```bash
pip install -r requirements.txt
```
### Running Scripts
To use the main algorithm, run the following command inside the main directory:
```bash
python main.py \
	--source_video_path <path_to_video> \
	--target_video_path <path_to_video> \
	[--model <model_name>]		   
``` 
To use the evaluation algorithm, run: 
```bash
python evaluation.py
```
