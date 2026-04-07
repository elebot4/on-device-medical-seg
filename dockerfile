FROM pytorch/pytorch:latest
WORKDIR /app

# copy requirements file & install python libraries
COPY requirements.txt requirements.txt
RUN python -m pip install -r requirements.txt

# copy remaining source code
COPY . .

# add default nonroot user w/ uid & pid 1000
RUN groupadd --gid 1000 nonroot \
    && useradd --uid 1000 --gid 1000 -m nonroot
RUN apt-get update \
    && apt-get install -y sudo \
    && echo nonroot ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/nonroot \
    && chmod 0440 /etc/sudoers.d/nonroot

# switch to created nonroot user
USER nonroot
EXPOSE 8080

# run shell by default
CMD ["/bin/bash"]
