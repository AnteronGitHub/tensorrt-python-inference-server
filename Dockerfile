FROM nvcr.io/nvidia/l4t-tensorrt:r8.5.2-runtime

RUN pip3 install cuda-python

WORKDIR /app

COPY main.py main.py

CMD ["python3", "main.py"]
