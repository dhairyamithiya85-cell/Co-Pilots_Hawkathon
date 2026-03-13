@echo off
REM setup_env.bat – Windows environment setup
REM Run from Anaconda Prompt

SET ENV_NAME=EDU

echo ============================================
echo  Offroad Segmentation - Environment Setup
echo ============================================

call conda create -y -n %ENV_NAME% python=3.10
call conda activate %ENV_NAME%

REM PyTorch with CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt

echo.
echo Environment %ENV_NAME% is ready.
echo Activate with: conda activate %ENV_NAME%
echo Train with:    python scripts/train.py
pause
