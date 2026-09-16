#!/usr/bin/env bash

# Prints the version from Cargo.toml's [package] section.
#
# `cargo metadata` would need a full dependency resolve to hand back one string
# this file spells out plainly, and it is the one thing tag.sh, release-notes.sh
# and the release workflow must all agree on.

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

VERSION="$(
    awk '
        /^\[package\]/ { in_pkg = 1; next }
        /^\[/          { in_pkg = 0 }
        in_pkg && /^version[[:space:]]*=/ { gsub(/[",]/, ""); print $3; exit }
    ' Cargo.toml
)"

if [ -z "$VERSION" ]; then
    echo "!> could not read version from Cargo.toml" >&2
    exit 1
fi

printf '%s\n' "$VERSION"
