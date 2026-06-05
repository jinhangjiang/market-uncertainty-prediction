FROM mcr.microsoft.com/azureml/curated/acpt-pytorch-2.2-cuda12.1:latest

WORKDIR /workspace

RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/environment.yml

RUN pip install --no-cache-dir \
    pandas>=2.0 \
    numpy>=1.24 \
    scikit-learn>=1.3 \
    neuralforecast>=1.7 \
    shap>=0.44 \
    matplotlib>=3.7 \
    seaborn>=0.12 \
    gdown>=4.7 \
    pyyaml>=6.0 \
    jinja2>=3.1 \
    beautifulsoup4>=4.12 \
    lxml>=4.9 \
    pyarrow>=14.0 \
    fastparquet \
    nltk>=3.8 \
    tqdm>=4.66 \
    scipy>=1.11 \
    azure-ai-ml>=1.12 \
    openpyxl

RUN git clone https://github.com/cygit/gdcm.git /workspace/gdcm && \
    pip install --no-cache-dir -e /workspace/gdcm/src

RUN python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords')"
