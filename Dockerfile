FROM python:3.6
MAINTAINER Varnit Mittal <varnitmittal77@gmail.com>
ADD . /my_app
WORKDIR /my_app
RUN pip install torch==1.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install -r requirements.txt
EXPOSE 8000
#CMD ["gunicorn", "-b", "0.0.0.0:8000", "app"]
CMD gunicorn -b 0.0.0.0:8000 --timeout=180 --worker-class=gevent --worker-connections=3 --workers=1 api:app