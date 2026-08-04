//! SubRip (`.srt`) parser.
//!
//! Blocks are separated by blank lines. Each block is an optional numeric
//! index line, a `HH:MM:SS,mmm --> HH:MM:SS,mmm` timing line, and one or
//! more text lines joined with `\n`. A UTF-8 BOM is stripped and CRLF line
//! endings are normalised before parsing.

use super::{Cue, Subtitles};

/// Parses SubRip text into a [`Subtitles`] track.
///
/// An empty (or all-blank) file parses to an empty track rather than
/// erroring -- no cues is a valid, if unusual, subtitle file.
pub fn parse(input: &str) -> Result<Subtitles, String> {
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);
    let normalized = input.replace("\r\n", "\n").replace('\r', "\n");

    // Cues are separated by a blank-AFTER-trim line -- see `super::blocks`'s
    // doc for why that trim matters.
    let mut cues = Vec::new();
    for block in super::blocks(&normalized) {
        let timing_idx = block.iter().position(|l| l.contains("-->")).ok_or_else(|| {
            format!("SubRip cue is missing a timing line (\"-->\"): {:?}", block.join("\n"))
        })?;
        let (start_s, end_s) = parse_timing_line(block[timing_idx])?;
        let text = block[timing_idx + 1..].join("\n");
        cues.push(Cue { start_s, end_s, text });
    }

    Ok(Subtitles::new(cues))
}

/// Parses a `HH:MM:SS,mmm --> HH:MM:SS,mmm` line into `(start_s, end_s)`.
///
/// A cue whose end precedes its start is an `Err`, not a cue. It can never
/// satisfy `start_s <= t < end_s` in [`super::Subtitles::at`], so accepting it
/// would drop that one line from the render with no diagnostic at all -- the
/// per-cue form of the empty-track failure this module's own docs forbid.
fn parse_timing_line(line: &str) -> Result<(f64, f64), String> {
    let (left, right) = line
        .split_once("-->")
        .ok_or_else(|| format!("SubRip timing line is missing \"-->\": {line:?}"))?;
    let start_s = parse_timestamp(left)
        .ok_or_else(|| format!("invalid SubRip timestamp in timing line {line:?}"))?;
    let end_s = parse_timestamp(right)
        .ok_or_else(|| format!("invalid SubRip timestamp in timing line {line:?}"))?;
    super::reject_reversed("SubRip", start_s, end_s, &format!("{line:?}"))?;
    Ok((start_s, end_s))
}

/// Parses a single `HH:MM:SS,mmm` timestamp into seconds: comma-separated
/// milliseconds, hours mandatory. See [`super::parse_timestamp`] for the
/// shared digit handling -- a leading `-` is rejected rather than parsed: as
/// a float, `"-00".parse::<f64>()` is `-0.0`, and `-0.0 * 3600.0 + 1.0` is
/// `+1.0` -- the sign would be lost, not the value rejected.
fn parse_timestamp(s: &str) -> Option<f64> {
    super::parse_timestamp(s, ',', 1000.0, false)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_basic_cue() {
        let s = parse("1\n00:00:01,000 --> 00:00:02,500\nHello\n").unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s.at(1.5), "Hello");
        assert_eq!(s.at(2.4), "Hello");
        assert_eq!(s.at(2.6), "");
    }

    #[test]
    fn joins_multi_line_cues_with_a_newline() {
        let s = parse("1\n00:00:01,000 --> 00:00:02,000\nline one\nline two\n").unwrap();
        assert_eq!(s.at(1.5), "line one\nline two");
    }

    #[test]
    fn parses_hours_minutes_seconds_and_milliseconds() {
        let s = parse("1\n01:02:03,004 --> 01:02:04,000\nx\n").unwrap();
        // 1*3600 + 2*60 + 3 + 0.004
        assert!((s.at(3723.5).len() as i32) > 0);
        assert_eq!(s.at(3723.005), "x");
    }

    #[test]
    fn accepts_crlf_line_endings() {
        let s = parse("1\r\n00:00:01,000 --> 00:00:02,000\r\nHello\r\n").unwrap();
        assert_eq!(s.at(1.5), "Hello", "a stray CR must not end up in the text");
    }

    #[test]
    fn skips_a_utf8_bom() {
        let s = parse("\u{feff}1\n00:00:01,000 --> 00:00:02,000\nHello\n").unwrap();
        assert_eq!(s.len(), 1, "a BOM must not break the first cue");
    }

    #[test]
    fn a_malformed_timestamp_errors_naming_the_line() {
        let err = parse("1\n00:00:01 --> nonsense\nHello\n").expect_err("must reject");
        assert!(err.contains("nonsense") || err.contains("timestamp"), "{err}");
    }

    #[test]
    fn an_empty_file_parses_to_an_empty_track_rather_than_erroring() {
        assert!(parse("").unwrap().is_empty(), "no cues is valid, not an error");
    }

    #[test]
    fn a_separator_line_holding_whitespace_still_separates_cues() {
        // The separator here is "   ", not "". Real files are full of these
        // and the difference is invisible in an editor. Splitting on a literal
        // "\n\n" does not break here, and filtering blank lines afterwards
        // deletes the separator -- so both cues merge, cue 1's text swallows
        // cue 2's index and timing line, and cue 2 disappears silently.
        let s = parse(
            "1\n00:00:01,000 --> 00:00:02,000\nFirst\n   \n\
             2\n00:00:03,000 --> 00:00:04,000\nSecond\n",
        )
        .unwrap();
        assert_eq!(s.len(), 2, "two cues, not one merged one");
        assert_eq!(s.at(1.5), "First", "cue 1 must not swallow what follows it");
        assert_eq!(s.at(3.5), "Second", "cue 2 must survive");
    }

    #[test]
    fn a_tab_separator_also_separates() {
        let s = parse(
            "1\n00:00:01,000 --> 00:00:02,000\nFirst\n\t\n\
             2\n00:00:03,000 --> 00:00:04,000\nSecond\n",
        )
        .unwrap();
        assert_eq!(s.len(), 2);
        assert_eq!(s.at(3.5), "Second");
    }

    #[test]
    fn several_blank_separator_lines_do_not_emit_empty_cues() {
        let s = parse(
            "1\n00:00:01,000 --> 00:00:02,000\nFirst\n\n \n\n\
             2\n00:00:03,000 --> 00:00:04,000\nSecond\n",
        )
        .unwrap();
        assert_eq!(s.len(), 2, "runs of blank lines are one separator");
    }

    #[test]
    fn nonsense_timestamps_are_refused_rather_than_becoming_undisplayable_cues() {
        // `f64::from_str` accepts every one of these. None can ever satisfy
        // `start_s <= t < end_s`, so each used to parse to Ok and then vanish
        // from the render with no diagnostic at all.
        for bad in [
            "inf:00:00,000 --> 00:00:02,000",
            "NaN:00:00,000 --> 00:00:02,000",
            "1e3:00:00,000 --> 00:00:02,000",
            "00:00:01,000 --> inf:00:00,000",
        ] {
            let err = parse(&format!("1\n{bad}\nx\n"))
                .expect_err(&format!("{bad:?} must be rejected"));
            assert!(err.contains("timestamp"), "{err}");
        }
    }

    #[test]
    fn a_negative_timestamp_is_refused_rather_than_silently_turning_positive() {
        // `"-00".parse::<f64>()` is -0.0, and -0.0 * 3600 + 0 * 60 + 1 == 1.0,
        // so `-00:00:01,000` used to become a POSITIVE one-second timestamp.
        let err = parse("1\n-00:00:01,000 --> 00:00:02,000\nx\n").expect_err("must reject");
        assert!(err.contains("timestamp"), "{err}");
    }

    #[test]
    fn a_reversed_cue_errors_rather_than_never_displaying() {
        let err = parse("1\n00:00:05,000 --> 00:00:02,000\nx\n").expect_err("must reject");
        assert!(err.to_lowercase().contains("before it starts"), "{err}");
    }

    #[test]
    fn a_file_not_ending_in_a_blank_line_still_yields_its_last_cue() {
        let s = parse("1\n00:00:01,000 --> 00:00:02,000\nOnly").unwrap();
        assert_eq!(s.len(), 1);
        assert_eq!(s.at(1.5), "Only");
    }
}
