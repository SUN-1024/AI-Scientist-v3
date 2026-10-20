#!/usr/bin/env bash
# Fetch the archived MIL benchmark datasets used by the OR-MIL audit case study.
#
# Usage: CASE_DATA=/path/to/data [HTTPS_PROXY=http://host:port] bash fetch_archive.sh
# The archive is verified against its SHA-256 before extraction; only the files the case uses are
# copied out (Corel svm-light, TREC9 svm-light, and the generation scripts quoted in the papers).
set -euo pipefail

URL="http://www.cs.columbia.edu/~andrews/mil/data/MIL-Data-2002-Musk-Corel-Trec9.tgz"
SHA256="590d7e6269cae7ff3ce013a7d11125e75b80721e3ae3bd62826d0158a49667a9"
DATA_DIR="${CASE_DATA:?set CASE_DATA to the directory that should receive the data}"
mkdir -p "$DATA_DIR"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

echo "downloading $URL"
if command -v curl >/dev/null 2>&1; then
  curl -fsSL ${HTTPS_PROXY:+--proxy "$HTTPS_PROXY"} "$URL" -o "$WORK/archive.tgz"
else
  wget -q ${HTTPS_PROXY:+--execute=https_proxy="$HTTPS_PROXY"} -O "$WORK/archive.tgz" "$URL"
fi

actual="$(sha256sum "$WORK/archive.tgz" | cut -d' ' -f1)"
if [ "$actual" != "$SHA256" ]; then
  echo "SHA-256 mismatch: expected $SHA256, got $actual" >&2
  exit 1
fi

tar -xzf "$WORK/archive.tgz" -C "$WORK"
mkdir -p "$DATA_DIR/mil2002"
cp "$WORK"/MilData/elephant/elephant.svm "$DATA_DIR/mil2002/"
cp "$WORK"/MilData/fox/fox.svm "$DATA_DIR/mil2002/"
cp "$WORK"/MilData/tiger/tiger.svm "$DATA_DIR/mil2002/"
for category in 1 2 3 4 7 9 10; do
  cp "$WORK/MilData/$category/data_200x200.svm" "$DATA_DIR/mil2002/trec9_$category.svm"
done
cp -R "$WORK/MilData/GenerateBlobworldMilData" "$DATA_DIR/"
echo "data extracted to $DATA_DIR/mil2002 (archive sha256 verified)"
