#!/usr/bin/env bash

# Tags the version in Cargo.toml and pushes it. It never moves an existing tag.
# If the version is already tagged, bump Cargo.toml instead.
#
# Tags are the bare version with no `v` prefix (`0.18.0`), matching every tag
# this repo has shipped since 0.0.1 and the release URLs users already have.
#
# Every remote is checked BEFORE the tag is created, so a multi-remote run
# either tags and pushes to all of them or touches none. A half-pushed release
# is the one state worth ruling out, since each remote builds and publishes its
# own release from the tag.
#
# Usage: tools/tag.sh [remote...]  (default: origin)

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [ "$#" -gt 0 ]; then
    REMOTES=("$@")
else
    REMOTES=("${REMOTE:-origin}")
fi

VERSION="$(tools/version.sh)"
TAG="${VERSION}"

# The release body comes from CHANGELOG.md, so a missing section means CI would
# publish an empty release. Fail here, where it costs a commit rather than a tag.
if ! tools/release-notes.sh "$VERSION" >/dev/null; then
    echo "!> add a '## ${VERSION} - <theme>' section to CHANGELOG.md before tagging" >&2
    exit 1
fi

# the tag has to point at a commit, not at whatever is lying around
if [ -n "$(git status --porcelain)" ]; then
    echo "!> working tree is dirty, commit before tagging ${TAG}" >&2
    exit 1
fi

if git rev-parse -q --verify "refs/tags/${TAG}" >/dev/null; then
    echo "!> ${TAG} already exists locally" >&2
    exit 1
fi

for REMOTE in "${REMOTES[@]}"; do
    # 0 found, 2 absent, anything else means the remote could not be read. Do
    # not tag on an unknown remote state, the push would just fail afterwards
    set +e
    git ls-remote --exit-code --tags "$REMOTE" "refs/tags/${TAG}" >/dev/null 2>&1
    REMOTE_CHECK=$?
    set -e
    case "${REMOTE_CHECK}" in
        0)
            echo "!> ${TAG} already exists on ${REMOTE}" >&2
            exit 1
            ;;
        2) ;;
        *)
            echo "!> could not reach ${REMOTE} to check for ${TAG}" >&2
            exit 1
            ;;
    esac

    # A tag whose commit the remote does not have builds nothing: the workflow
    # would check out the tag on a repo that has never seen the commit. The
    # fetch is what makes this honest: a remote-tracking ref that has not been
    # updated since the last push reports the remote as behind when it is not.
    git fetch --quiet "$REMOTE" || {
        echo "!> could not fetch ${REMOTE}" >&2
        exit 1
    }
    if [ -z "$(git branch -r --contains HEAD --list "${REMOTE}/*")" ]; then
        echo "!> ${REMOTE} does not have $(git rev-parse --short HEAD), push the branch first" >&2
        exit 1
    fi
done

TITLE="$(tools/release-notes.sh "$VERSION" --title)"

git tag -a "${TAG}" -m "${TITLE}"
for REMOTE in "${REMOTES[@]}"; do
    git push "$REMOTE" "${TAG}"
    echo ">> pushed ${TAG} to ${REMOTE} ($(git rev-parse --short HEAD))"
done
