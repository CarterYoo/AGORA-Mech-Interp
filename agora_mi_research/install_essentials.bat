@echo off
echo ================================================
echo AGORA MI Research - Essential Packages Install
echo ================================================
echo.

echo [1/7] Installing core utilities...
pip install loguru pyyaml python-dotenv tqdm

echo.
echo [2/7] Installing data processing...
pip install pandas numpy

echo.
echo [3/7] Installing PyTorch (CUDA 11.8)...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

echo.
echo [4/7] Installing transformers...
pip install transformers accelerate bitsandbytes

echo.
echo [5/7] Installing RAG components...
pip install sentence-transformers chromadb

echo.
echo [6/7] Installing analysis tools...
pip install scipy scikit-learn matplotlib seaborn

echo.
echo [7/7] Installing annotation tools...
pip install label-studio label-studio-sdk

echo.
echo ================================================
echo Installation Complete!
echo ================================================
echo.
echo Optional: Install RAGatouille for AGORA ColBERT
echo   pip install ragatouille
echo.
echo Optional: Install advanced packages
echo   pip install h5py statsmodels datasets
echo.
echo Test installation:
echo   python -c "import loguru; print('OK')"
echo.
pause

