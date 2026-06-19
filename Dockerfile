# Run the cg simulator locally. libcg.so is Linux x86-64 only, so on Apple
# Silicon / Windows this runs under emulation (slow but works).
#   docker build --platform=linux/amd64 -t ptcg-sim .
#   docker run --platform=linux/amd64 --rm -v "$PWD":/work -w /work ptcg-sim \
#       python scripts/sim_smoke.py
FROM --platform=linux/amd64 python:3.12-slim

RUN pip install --no-cache-dir pandas numpy

WORKDIR /work
CMD ["python", "scripts/sim_smoke.py"]
