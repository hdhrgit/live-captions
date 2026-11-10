#!/bin/sh
# The build version from git:  deploy/version.sh [run-number]
#   on a v* tag:     0.1.0
#   anywhere else:   <latest tag>+r<run>.g<commit>, e.g. 0.1.0+r3.g6b33b4a
#                    (without a run number: 0.1.0+g6b33b4a; no tag yet: 0.0.0+...)
cd "$(dirname "$0")/.."
sha=$(git rev-parse --short HEAD)
exact=$(git describe --tags --exact-match --match 'v[0-9]*' 2>/dev/null) && { echo "${exact#v}"; exit; }
last=$(git describe --tags --abbrev=0 --match 'v[0-9]*' 2>/dev/null || echo v0.0.0)
echo "${last#v}+${1:+r$1.}g$sha"
