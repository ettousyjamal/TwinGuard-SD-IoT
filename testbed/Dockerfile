# Data plane of the TwinGuard-SD-IoT testbed: Mininet 2.3 with Open vSwitch 2.13 (kernel datapath) on Ubuntu 20.04.
# Build:  docker build -t twinguard-testbed:latest testbed/
# The container runs privileged; the host needs the openvswitch kernel module (modprobe openvswitch).
FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      mininet openvswitch-switch openvswitch-common python3-ryu python3-numpy python3-networkx \
      iproute2 iputils-ping iperf3 net-tools procps python3-pip ethtool kmod \
    && rm -rf /var/lib/apt/lists/*
RUN pip3 install --no-cache-dir mininet==2.3.0.dev6 && python3 -c "import mininet.net; print('mininet py3 ok')"
WORKDIR /work
CMD ["/bin/bash"]
