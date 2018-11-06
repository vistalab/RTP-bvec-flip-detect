FROM python:3.6


WORKDIR /usr/src/

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY detector ./detector
COPY detectWithinDir.py ./
# COPY dtiInit_02-Nov-2018_20-39-23.zip ./

CMD ["python", "detectWithinDir.py"]
