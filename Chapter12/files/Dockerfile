FROM public.ecr.aws/lambda/python:3.6

RUN pip3 install mxnet

COPY additional-params.json   ./additional-params.json
COPY app.py   ./app.py
COPY manifest.json ./manifest.json
COPY model_algo-1 ./model_algo-1
COPY mx-mod-0000.params ./mx-mod-0000.params
COPY mx-mod-symbol.json ./mx-mod-symbol.json

CMD ["app.handler"]
