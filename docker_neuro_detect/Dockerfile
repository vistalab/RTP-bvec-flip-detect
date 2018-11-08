
FROM python:3.6


WORKDIR /flywheel/v0

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY detector ./detector
COPY detectWithinDir.py ./run.py
COPY manifest.json ./

CMD ["python", "run.py"]
