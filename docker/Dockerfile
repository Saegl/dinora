FROM tensorflow/tensorflow:latest-gpu

# Copy workdir
COPY . /proj
WORKDIR /proj

# Install python dependencies with poetry
RUN pip install -r docker/requirements.txt

# Binary
ENV PATH "$PATH:/proj/bin"
ENV PYTHONPATH "${PYTHONPATH}:/proj"
