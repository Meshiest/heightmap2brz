//! WebVTT (`.vtt` / `.webvtt`) parser.
//!
//! Different enough from SubRip that routing `.vtt` through [`super::srt`]
//! cannot parse a real file: the signature block (`WEBVTT`, no `-->`) reads
//! as a cue missing its timing line, the fraction separator is `.` not `,`,
//! and the hours field is optional (`MM:SS.mmm` is legal).
//!
//! A timing line may also carry cue settings after the end timestamp
//! (`... --> ... align:start position:10%`), a cue may be preceded by an
//! identifier line, `NOTE`/`STYLE`/`REGION` blocks may appear between cues,
//! and the payload may carry inline markup (`<v Roger>`, `<c.loud>`, `<i>`,
//! `<00:00:01.500>`) plus HTML entities.
//!
//! Deliberately out of scope: positioning, styling and regions. Cue
//! settings are parsed only far enough to be ignored; `STYLE`/`REGION`
//! blocks are skipped whole.

use super::{Cue, Subtitles};

/// Whether `input` opens with WebVTT's own signature, ignoring a BOM.
///
/// Used by [`super::parse_auto`] to sniff a `.vtt` that arrived under some
/// other name (or none). The signature is the format's only unambiguous
/// marker: WebVTT also contains `-->`, so sniffing on the arrow alone would
/// hand every WebVTT file to the SubRip parser, which cannot read one.
pub fn is_webvtt(input: &str) -> bool {
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);
    let Some(rest) = input.strip_prefix("WEBVTT") else {
        return false;
    };
    // The spec allows `WEBVTT`, `WEBVTT<space>anything`, `WEBVTT<tab>anything`
    // and nothing else -- `WEBVTTX` is not a signature.
    matches!(rest.chars().next(), None | Some(' ') | Some('\t') | Some('\n') | Some('\r'))
}

/// Parses WebVTT text into a [`Subtitles`] track.
///
/// The `WEBVTT` signature is required, not guessed: a file without it is
/// most likely SubRip under a `.vtt` name, and reading that as WebVTT would
/// silently misparse its `,`-separated timestamps rather than erroring.
///
/// An empty cue list is valid, matching [`super::srt::parse`]: a `.vtt`
/// holding only a header and a `NOTE` is a real, if useless, file.
pub fn parse(input: &str) -> Result<Subtitles, String> {
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);
    let normalized = input.replace("\r\n", "\n").replace('\r', "\n");

    if !is_webvtt(&normalized) {
        let first = normalized.lines().next().unwrap_or("");
        return Err(format!(
            "not a WebVTT file: it must begin with a \"WEBVTT\" signature line, but it \
             begins with {first:?}. If this is really a SubRip file, name it .srt -- its \
             timestamps use \",\" for the fraction and would be misread here."
        ));
    }

    // Blocks are separated by a blank line, tested AFTER trimming -- see
    // `super::blocks`'s doc for why that trim matters.
    let mut cues = Vec::new();
    let mut blocks = super::blocks(&normalized).into_iter();
    // The signature block is everything up to the first blank line; it
    // carries no cue, only the `WEBVTT` line and any header metadata.
    // `is_webvtt` above guarantees the input starts with that line, so there
    // is always at least one block to discard here.
    blocks.next();
    for block in blocks {
        if let Some(cue) = parse_block(&block)? {
            cues.push(cue);
        }
    }

    Ok(Subtitles::new(cues))
}

/// Parses one non-header block into a cue, or `None` for a block that
/// carries no cue (a comment, a style sheet, a region definition).
///
/// A block that is none of those and still has no `-->` is an `Err`,
/// matching [`super::srt::parse`]: silently skipping it would shorten the
/// track with no diagnostic.
fn parse_block(block: &[&str]) -> Result<Option<Cue>, String> {
    let first = block[0].trim();
    if is_keyword_block(first, "NOTE") || is_keyword_block(first, "STYLE")
        || is_keyword_block(first, "REGION")
    {
        return Ok(None);
    }

    let timing_idx = block.iter().position(|l| l.contains("-->")).ok_or_else(|| {
        format!("WebVTT cue is missing a timing line (\"-->\"): {:?}", block.join("\n"))
    })?;
    let (start_s, end_s) = parse_timing_line(block[timing_idx])?;

    // Everything before the timing line is the optional cue identifier
    // (WebVTT allows exactly one such line); everything after it is payload.
    let text = block[timing_idx + 1..]
        .iter()
        .map(|l| clean_text(l))
        .collect::<Vec<_>>()
        .join("\n");
    let text = text.trim().to_string();
    if text.is_empty() {
        // A cue with no text would still occupy its time window and, being
        // the most recently started cue, would blank whatever overlapping
        // cue is actually speaking. Dropping it is the safer of the two.
        return Ok(None);
    }
    Ok(Some(Cue { start_s, end_s, text }))
}

/// Whether `line` is a block-opening keyword: the bare word, or the word
/// followed by whitespace. `NOTEBOOK` is not a `NOTE`.
fn is_keyword_block(line: &str, keyword: &str) -> bool {
    match line.strip_prefix(keyword) {
        Some(rest) => rest.is_empty() || rest.starts_with(char::is_whitespace),
        None => false,
    }
}

/// Parses a `start --> end [cue settings]` line into `(start_s, end_s)`.
///
/// Cue settings (`align:`, `line:`, `position:`, `size:`, `vertical:`,
/// `region:`) trail the end timestamp separated by whitespace and are
/// ignored -- see the module doc.
fn parse_timing_line(line: &str) -> Result<(f64, f64), String> {
    let (left, right) = line
        .split_once("-->")
        .ok_or_else(|| format!("WebVTT timing line is missing \"-->\": {line:?}"))?;
    let start_s = parse_timestamp(left)
        .ok_or_else(|| format!("invalid WebVTT timestamp in timing line {line:?}"))?;
    // Only the first whitespace-separated token on the right is the end
    // timestamp; the rest is cue settings.
    let end_token = right.split_whitespace().next().unwrap_or("");
    let end_s = parse_timestamp(end_token)
        .ok_or_else(|| format!("invalid WebVTT timestamp in timing line {line:?}"))?;
    super::reject_reversed("WebVTT", start_s, end_s, &format!("{line:?}"))?;
    Ok((start_s, end_s))
}

/// Parses a `HH:MM:SS.mmm` or `MM:SS.mmm` timestamp into seconds:
/// dot-separated milliseconds, hours optional. See [`super::parse_timestamp`]
/// for the shared digit handling (rejects `inf`/`NaN`/signs/exponents rather
/// than parsing them into a cue no frame time can ever fall inside).
fn parse_timestamp(s: &str) -> Option<f64> {
    super::parse_timestamp(s, '.', 1000.0, true)
}

/// Strips WebVTT inline markup and resolves the entities the format defines.
///
/// Markup is every `<...>` span: voice spans (`<v Roger>`), class spans
/// (`<c.loud>`), the plain `<i>`/`<b>`/`<u>`/`<ruby>` tags, and karaoke
/// timestamps (`<00:00:01.500>`) -- all carry no text of their own.
///
/// An unterminated `<` is kept literally, not swallowed.
fn clean_text(raw: &str) -> String {
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    while let Some(open) = rest.find('<') {
        out.push_str(&rest[..open]);
        match rest[open..].find('>') {
            Some(close) => rest = &rest[open + close + 1..],
            None => {
                // No closing `>` anywhere: this is text, not markup.
                out.push_str(&rest[open..]);
                rest = "";
                break;
            }
        }
    }
    out.push_str(rest);
    decode_entities(&out)
}

/// Resolves the five character references WebVTT names, plus `&nbsp;`.
///
/// Each reference is consumed whole in one left-to-right pass, so
/// `&amp;lt;` yields `&lt;` as text rather than being re-scanned into `<`.
fn decode_entities(raw: &str) -> String {
    const ENTITIES: &[(&str, &str)] = &[
        ("&amp;", "&"),
        ("&lt;", "<"),
        ("&gt;", ">"),
        ("&nbsp;", " "),
        ("&lrm;", ""),
        ("&rlm;", ""),
    ];
    if !raw.contains('&') {
        return raw.to_string();
    }
    let mut out = String::with_capacity(raw.len());
    let mut rest = raw;
    'outer: while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        for (entity, replacement) in ENTITIES {
            if rest[amp..].starts_with(entity) {
                out.push_str(replacement);
                rest = &rest[amp + entity.len()..];
                continue 'outer;
            }
        }
        // Not a reference this parser knows: keep the `&` as written.
        out.push('&');
        rest = &rest[amp + 1..];
    }
    out.push_str(rest);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_basic_cue() {
        let s = parse("WEBVTT\n\n00:00:01.000 --> 00:00:02.500\nHello\n").unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s.at(1.5), "Hello");
        assert_eq!(s.at(2.6), "");
    }

    #[test]
    fn the_fraction_separator_is_a_dot_not_a_comma() {
        // The SubRip parser requires `,` and returns None on anything else,
        // so this is the difference that makes routing .vtt at it useless.
        let s = parse("WEBVTT\n\n00:00:01.500 --> 00:00:02.000\nx\n").unwrap();
        assert_eq!(s.at(1.6), "x", "1.500 must be one and a half seconds");
        assert_eq!(s.at(1.4), "", "and not 1.0");
    }

    #[test]
    fn the_hours_field_is_optional() {
        let s = parse("WEBVTT\n\n01:30.000 --> 01:31.000\nx\n").unwrap();
        assert_eq!(s.at(90.5), "x", "MM:SS.mmm is 90.5 s in, not 1.5 s");
    }

    #[test]
    fn a_header_with_trailing_text_and_metadata_lines_is_skipped() {
        let s = parse(
            "WEBVTT - This file has a title\nKind: captions\nLanguage: en\n\n\
             00:00:01.000 --> 00:00:02.000\nHello\n",
        )
        .unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s.at(1.5), "Hello");
    }

    #[test]
    fn a_cue_identifier_line_is_not_part_of_the_text() {
        let s = parse("WEBVTT\n\nintro\n00:00:01.000 --> 00:00:02.000\nHello\n").unwrap();
        assert_eq!(s.at(1.5), "Hello", "the identifier must not be spoken");
    }

    #[test]
    fn cue_settings_after_the_end_timestamp_are_ignored_not_fatal() {
        let s = parse(
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000 align:start position:10%\nHello\n",
        )
        .unwrap();
        assert_eq!(s.at(1.5), "Hello");
    }

    #[test]
    fn inline_tags_are_stripped_and_their_text_kept() {
        let s = parse(
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n\
             <v Roger>Hi <c.loud>there</c></v>\n",
        )
        .unwrap();
        assert_eq!(s.at(1.5), "Hi there");
    }

    #[test]
    fn karaoke_timestamp_tags_are_stripped() {
        let s = parse(
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nGo<00:00:01.500> now\n",
        )
        .unwrap();
        assert_eq!(s.at(1.2), "Go now");
    }

    #[test]
    fn an_unterminated_angle_bracket_is_text_not_a_swallowed_line() {
        let s = parse("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\n5 < 6 always\n").unwrap();
        assert_eq!(s.at(1.5), "5 < 6 always", "text must not vanish");
    }

    #[test]
    fn entities_are_resolved() {
        let s = parse(
            "WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nTom &amp; Jerry &lt;3\n",
        )
        .unwrap();
        assert_eq!(s.at(1.5), "Tom & Jerry <3");
    }

    #[test]
    fn note_style_and_region_blocks_are_skipped_without_erroring() {
        let s = parse(
            "WEBVTT\n\n\
             NOTE this is a comment\nspanning two lines\n\n\
             STYLE\n::cue { color: papayawhip; }\n\n\
             REGION\nid:fred width:40%\n\n\
             00:00:01.000 --> 00:00:02.000\nHello\n",
        )
        .unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s.at(1.5), "Hello");
    }

    #[test]
    fn a_notebook_is_not_a_note() {
        // `NOTE` is a keyword only as a whole word; a cue identifier that
        // merely starts with those letters must still yield its cue.
        let s = parse("WEBVTT\n\nNOTEBOOK\n00:00:01.000 --> 00:00:02.000\nHello\n").unwrap();
        assert_eq!(s.at(1.5), "Hello");
    }

    #[test]
    fn multi_line_payloads_join_with_a_newline() {
        let s = parse("WEBVTT\n\n00:00:01.000 --> 00:00:02.000\nline one\nline two\n").unwrap();
        assert_eq!(s.at(1.5), "line one\nline two");
    }

    #[test]
    fn crlf_and_a_bom_are_both_tolerated() {
        let s = parse("\u{feff}WEBVTT\r\n\r\n00:00:01.000 --> 00:00:02.000\r\nHello\r\n").unwrap();
        assert_eq!(s.at(1.5), "Hello", "no stray CR, and the BOM is not a signature break");
    }

    #[test]
    fn a_whitespace_only_separator_still_separates_cues() {
        // Same trap `srt::parse` documents: the separator below holds a
        // space, so splitting on a literal "\n\n" would merge both cues and
        // lose the second one silently.
        let s = parse(
            "WEBVTT\n \n\
             00:00:01.000 --> 00:00:02.000\nFirst\n   \n\
             00:00:03.000 --> 00:00:04.000\nSecond\n",
        )
        .unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s.at(1.5), "First");
        assert_eq!(s.at(3.5), "Second");
    }

    #[test]
    fn a_file_without_the_signature_is_rejected_naming_the_format() {
        let err = parse("1\n00:00:01,000 --> 00:00:02,000\nHello\n").expect_err("must reject");
        assert!(err.contains("WEBVTT"), "names the signature: {err}");
        assert!(err.contains("SubRip") || err.contains(".srt"), "points at the likely truth: {err}");
    }

    #[test]
    fn webvttx_is_not_a_signature() {
        assert!(!is_webvtt("WEBVTTX\n\n"), "the signature is a whole token");
        assert!(is_webvtt("WEBVTT\n"));
        assert!(is_webvtt("WEBVTT"));
        assert!(is_webvtt("WEBVTT - title\n"));
        assert!(is_webvtt("\u{feff}WEBVTT\n"));
    }

    #[test]
    fn a_cue_block_with_no_timing_line_errors_rather_than_shortening_the_track() {
        let err = parse("WEBVTT\n\nid only\nno arrow here\n").expect_err("must reject");
        assert!(err.contains("-->"), "names what is missing: {err}");
    }

    #[test]
    fn a_reversed_cue_errors_rather_than_never_displaying() {
        let err = parse("WEBVTT\n\n00:00:05.000 --> 00:00:02.000\nx\n").expect_err("must reject");
        assert!(err.to_lowercase().contains("before it starts"), "{err}");
    }

    #[test]
    fn nonsense_timestamps_are_refused_rather_than_becoming_undisplayable_cues() {
        for bad in ["inf:00:00.000 --> 00:00:02.000", "NaN:00:00.000 --> 00:00:02.000",
                    "-00:00:01.000 --> 00:00:02.000", "1e3:00:00.000 --> 00:00:02.000"] {
            let err = parse(&format!("WEBVTT\n\n{bad}\nx\n"))
                .expect_err("{bad} must be rejected");
            assert!(err.contains("timestamp"), "{err}");
        }
    }

    #[test]
    fn a_header_only_file_is_an_empty_track_not_an_error() {
        assert!(parse("WEBVTT\n").unwrap().is_empty());
        assert!(parse("WEBVTT\n\nNOTE nothing to say\n").unwrap().is_empty());
    }
}
