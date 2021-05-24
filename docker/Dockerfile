FROM tensorflow/tensorflow:2.2.0-gpu

ARG USER_ID

RUN apt-get update && apt-get install -y git

WORKDIR /workspace

# Fix permissions
RUN chown --recursive ${USER_ID} /workspace
RUN mkdir /.local && chown ${USER_ID} /.local
RUN mkdir /.jupyter && chown ${USER_ID} /.jupyter
RUN mkdir /.cache && chown ${USER_ID} /.cache
RUN mkdir /.config && chown ${USER_ID} /.config
RUN mkdir /tensorflow_datasets && chown ${USER_ID} /tensorflow_datasets
RUN mkdir /nltk_data && chown ${USER_ID} /nltk_data

# Install python packages
ADD explainability_transfer/requirements.txt .
RUN pip install -U pip
RUN pip install -r requirements.txt
RUN python -m spacy download en_core_web_sm

# Make sure we can import from /workspace and the explainability_transfer folder
ENV PYTHONPATH "/workspace:${PYTHONPATH}"
ENV PYTHONPATH "/workspace/explainability_transfer:${PYTHONPATH}"
ENV PATH="/.local/bin:${PATH}"
