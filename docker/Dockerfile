FROM tensorflow/tensorflow:latest-gpu

# Copy workdir
COPY . /proj
WORKDIR /proj

# Install python dependencies with poetry
RUN pip install -U pip
RUN pip install poetry

RUN poetry config virtualenvs.create false \
  && poetry install --no-interaction --no-ansi

# Binary
RUN useradd -m -d /proj -s /bin/bash app
RUN chown -R app:app /proj/*
RUN chmod +x /proj/bin/*
ENV PATH "$PATH:/proj/bin"
USER app
