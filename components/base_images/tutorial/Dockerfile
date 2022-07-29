FROM python:3.9.10

COPY ./components/base_images/utils components/base_images/utils
COPY ./components/base_images/tutorial/ components/base_images/tutorial/
COPY ./components/base_images/tutorial/setup.py setup.py

RUN pip3 install --upgrade setuptools
RUN pip3 install pip-tools
RUN pip-compile components/base_images/tutorial/requirements.in
RUN pip3 install -r components/base_images/tutorial/requirements.txt

RUN pip3 install -e .