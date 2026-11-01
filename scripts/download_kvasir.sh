#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data/raw .local/source_pages
url=https://datasets.simula.no/downloads/kvasir-seg.zip
bundle=()
if [[ -f .local/simula-ca-bundle.pem ]]; then bundle=(--cacert .local/simula-ca-bundle.pem); fi
if [[ -f data/raw/kvasir-seg.zip ]]; then
  echo 'Archive already exists; verify docs/downloads.json before replacing.'
  exit 0
fi
curl --fail --location "${bundle[@]}" --connect-timeout 20 --max-time 900 --retry 2 --continue-at - --output data/raw/kvasir-seg.zip.part "$url"
mv data/raw/kvasir-seg.zip.part data/raw/kvasir-seg.zip
