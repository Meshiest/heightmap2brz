# Releasing

## Cutting a release

Bump `version` in `Cargo.toml` and add a `## <version> - <theme>` section at the
top of `CHANGELOG.md`, in the same commit. `just notes` prints what the release
will say. Push the branch to every remote that should publish, then:

```
just tag             # the default remote
just tag origin      # a named remote
just tag-all         # one tag, pushed to every remote in REMOTES
```

Tags are the bare version with no `v` prefix (`0.18.0`).

`just tag` refuses a dirty working tree, a tag that already exists locally or on
a remote, a remote it cannot reach, a commit the remote does not have, and a
version `CHANGELOG.md` does not mention. It checks every remote before creating
the tag, so `tag-all` either tags and pushes to all of them or touches none.

## The workflow

`.github/workflows/release.yml` runs on any `N.N.N` tag push:

| Job | What it does |
| --- | --- |
| `prepare` | Fails if the tag disagrees with `Cargo.toml` or `CHANGELOG.md` has no section for it. The `##` heading becomes the release title, the rest the body. |
| `build` | `windows-latest` builds `heightmap.exe` + `heightmap_gui.exe`; `ubuntu-latest` builds the CLI only, as `heightmap`. |
| `release` | Writes `SHA256SUMS.txt`, attaches a build-provenance attestation, publishes. Re-runs update the release in place. |

The CLI builds `--no-default-features`, dropping `gui` and with it
eframe/rfd/rodio, so Linux needs no GTK, ALSA or X11 packages on the runner.
`just dist` builds the same way.

Each remote is a separate GitHub repo running its own copy of the workflow,
publishing its own release from the same tag.

## Verifying a download

```
sha256sum -c SHA256SUMS.txt --ignore-missing
gh attestation verify heightmap.exe --repo Meshiest/heightmap2brs
```

The attestation ties the exact bytes to the workflow run and the commit that
built them.

