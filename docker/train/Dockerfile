FROM ubuntu

RUN apt-get update \
&& apt-get install -y python3 python3-pip \
&& useradd -m repl \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY . .

RUN pip3 install -r requirements.txt

USER repl

CMD ["python3", "-m", "scripts.train_pipeline", "--alpha", "0.1"]