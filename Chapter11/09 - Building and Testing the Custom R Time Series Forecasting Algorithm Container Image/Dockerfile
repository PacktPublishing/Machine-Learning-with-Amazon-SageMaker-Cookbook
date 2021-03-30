FROM arvslat/amazon-sagemaker-cookbook-r-base:1

RUN R -e "install.packages('forecast',repos='https://cloud.r-project.org')"

COPY train /usr/local/bin/train
COPY serve /usr/local/bin/serve
COPY api.r /usr/local/bin/api.r