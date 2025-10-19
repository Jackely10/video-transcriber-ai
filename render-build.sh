#!/usr/bin/env bash
set -euxo pipefail

apt-get update
apt-get install -y --no-install-recommends ffmpeg
apt-get clean
rm -rf /var/lib/apt/lists/*
