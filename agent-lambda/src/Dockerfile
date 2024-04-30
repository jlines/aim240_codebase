FROM public.ecr.aws/lambda/python:3.11
RUN yum -y install git
COPY requirements.txt ./

RUN python3.11 -m pip install -r requirements.txt -t .

COPY app.py ./
COPY runpod_lm.py ./
COPY ssnragtotal.py ./

# Command can be overwritten by providing a different command in the template directly.
CMD ["app.lambda_handler"]
