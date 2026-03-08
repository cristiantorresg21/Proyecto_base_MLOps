FROM python:3.13-slim

RUN useradd -m repl

WORKDIR /app

COPY . .

RUN pip install --upgrade pip \
&& pip install -r requirements.txt

RUN chown -R repl:repl /app

USER repl

CMD ["python3", "-m", "scripts.train_pipeline", "--alpha", "0.1"]