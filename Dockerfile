FROM ivanleung/tfimage-gpu
WORKDIR /app
RUN mkdir /app/data
RUN mkdir /app/src
ENTRYPOINT ["python", "src/main.py"]
