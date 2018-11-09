
FROM python:3.6


WORKDIR /flywheel/v0

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY detector/* ./
COPY detectWithinDir.py ./run
COPY manifest.json ./

RUN chmod +x *

CMD ["python", "run"]
