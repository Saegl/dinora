FROM tensorflow/tensorflow:latest-gpu

# Copy workdir
COPY . /proj
WORKDIR /proj

# Install python dependencies with poetry
RUN pip install -r docker/requirements.txt

# Binary
RUN useradd -m -d /proj -s /bin/bash app
RUN chown -R app:app /proj/*
RUN chmod +x /proj/bin/*
ENV PATH "$PATH:/proj/bin"
ENV PYTHONPATH "${PYTHONPATH}:/proj"
USER app
