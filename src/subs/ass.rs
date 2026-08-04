//! Advanced SubStation Alpha (`.ass` / `.ssa`) parser.
//!
//! Reads the `[Events]` section only. `Dialogue:` lines are taken;
//! `Comment:` lines are ignored. The section's own `Format:` line declares
//! field order, so `Start`/`End`/`Text` are located by name, not by a fixed
//! index -- field order varies between files.
//!
//! Deliberately out of scope: styles, positioning, karaoke and drawing
//! commands. `Text` gets three transforms: `\N`/`\n` -> newline, `\h` ->
//! space, and `{...}` override blocks stripped (a literal `\{` survives).

use super::{Cue, Subtitles};

/// Parses Advanced SubStation Alpha text into a [`Subtitles`] track.
///
/// A missing `Format:` line, or no `[Events]` section at all, is an error
/// rather than a guess at field positions or a silently empty track (see the
/// module doc for why that distinction matters). An `[Events]` section that
/// is genuinely empty (a header and a `Format:` line, no `Dialogue:` lines)
/// is still a valid empty track: the file said so.
pub fn parse(input: &str) -> Result<Subtitles, String> {
    // U+FEFF is `Cf`, not `White_Space`, so `str::trim` does NOT remove it --
    // a BOM'd file whose first line is `[Events]` would fail the section test
    // below, never enter the section, and skip every `Dialogue:` line.
    // `srt::parse` has always stripped it; this parser did not.
    let input = input.strip_prefix('\u{feff}').unwrap_or(input);
    let normalized = input.replace("\r\n", "\n").replace('\r', "\n");

    let mut fields: Option<Vec<String>> = None;
    let mut cues = Vec::new();

    let mut in_events = false;
    let mut seen_events = false;
    for line in normalized.lines() {
        let trimmed = line.trim();

        if let Some(name) = section_name(trimmed) {
            in_events = name.trim().eq_ignore_ascii_case("Events");
            seen_events |= in_events;
            continue;
        }
        if !in_events {
            continue;
        }

        if let Some(rest) = strip_prefix_ci(trimmed, "Format:") {
            fields = Some(rest.split(',').map(|f| f.trim().to_string()).collect());
            continue;
        }

        let Some(rest) = strip_prefix_ci(trimmed, "Dialogue:") else {
            continue;
        };
        let fields = fields.as_ref().ok_or_else(|| {
            "ASS [Events] section has a Dialogue: line before its Format: line \
             (or is missing one) -- field order cannot be determined"
                .to_string()
        })?;

        let start_idx = field_index(fields, "Start")?;
        let end_idx = field_index(fields, "End")?;
        let text_idx = field_index(fields, "Text")?;

        // `Text` is the last field declared in Format and may itself
        // contain commas, so split into exactly `fields.len()` parts: a
        // plain `split(',')` would truncate the text at its first comma.
        let parts: Vec<&str> = rest.splitn(fields.len(), ',').map(|p| p.trim()).collect();
        if parts.len() < fields.len() {
            return Err(format!(
                "ASS Dialogue line has {} field(s), expected {} per Format: {:?}",
                parts.len(),
                fields.len(),
                rest
            ));
        }

        let start_s = parse_timestamp(parts[start_idx])
            .ok_or_else(|| format!("invalid ASS timestamp in Dialogue line: {:?}", parts[start_idx]))?;
        let end_s = parse_timestamp(parts[end_idx])
            .ok_or_else(|| format!("invalid ASS timestamp in Dialogue line: {:?}", parts[end_idx]))?;
        // Can never satisfy `start_s <= t < end_s` in `Subtitles::at`, so
        // accepting a reversed cue would drop this one line from the render
        // with no diagnostic -- the per-cue form of the empty-track failure.
        super::reject_reversed("ASS", start_s, end_s, &format!("{rest:?}"))?;
        // `clean_text` removes any drawing-mode SPAN (see its doc) along with
        // the override blocks; what is left is whatever real dialogue the
        // line also carried.
        let text = clean_text(parts[text_idx]);
        if text.trim().is_empty() {
            // Nothing to say. Emitting the cue anyway would give it a time
            // window where it wins `Subtitles::at`'s most-recently-begun
            // overlap rule, blanking whatever overlapping line is actually
            // speaking. A pure drawing lands here, as does a `Dialogue:`
            // whose Text field was empty to begin with.
            continue;
        }

        cues.push(Cue { start_s, end_s, text });
    }

    if !seen_events {
        return Err(
            "ASS/SSA file has no [Events] section, so it carries no dialogue this parser \
             can read -- an empty subtitle track would render as a video where nobody \
             speaks, which is indistinguishable from a correct render of a quiet scene"
                .to_string(),
        );
    }

    Ok(Subtitles::new(cues))
}

/// The name inside a `[Section]` header line, or `None` for any other line.
///
/// Matched as "starts with `[`, up to the first `]`" rather than
/// `starts_with('[') && ends_with(']')` so that a trailing comment
/// (`[Events] ; dialogue below`) still enters the section. Under the stricter
/// test such a line was not a header at all, `in_events` never became true,
/// and the whole file parsed to `Ok` with zero cues.
fn section_name(line: &str) -> Option<&str> {
    let rest = line.strip_prefix('[')?;
    let end = rest.find(']')?;
    Some(&rest[..end])
}

/// Case-insensitive prefix strip, returning the trimmed remainder.
///
/// Checks `is_char_boundary` before slicing: `prefix` is always ASCII, but
/// `line` is arbitrary UTF-8 subtitle text and a byte-length slice on a
/// non-boundary would panic.
fn strip_prefix_ci<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    if line.len() >= prefix.len()
        && line.is_char_boundary(prefix.len())
        && line[..prefix.len()].eq_ignore_ascii_case(prefix)
    {
        Some(line[prefix.len()..].trim_start())
    } else {
        None
    }
}

/// Finds `name`'s position among the `Format:` line's declared fields
/// (case-insensitive), erroring rather than assuming a fixed layout.
fn field_index(fields: &[String], name: &str) -> Result<usize, String> {
    fields
        .iter()
        .position(|f| f.eq_ignore_ascii_case(name))
        .ok_or_else(|| format!("ASS Format: line is missing the {name:?} field: {fields:?}"))
}

/// Parses a `H:MM:SS.cc` ASS timestamp into seconds.
///
/// The fractional part is centiseconds, not milliseconds -- `.50` is half a
/// second, not 50 ms. See [`super::parse_timestamp`] for the shared digit
/// handling (rejects `inf`/`NaN`/signs/exponents rather than parsing them).
fn parse_timestamp(s: &str) -> Option<f64> {
    super::parse_timestamp(s, '.', 100.0, false)
}

/// The drawing-mode argument an override block sets, if it sets one.
///
/// `\p<nonzero>` opens ASS drawing mode: the path coords that follow sit
/// BETWEEN override blocks, so brace-stripping alone leaves them as text.
/// Last `\p` in a block wins (libass order). `\pos` is not drawing (`o` is
/// not a digit).
fn drawing_arg(block: &str) -> Option<u32> {
    let bytes = block.as_bytes();
    let mut found = None;
    for i in 0..bytes.len().saturating_sub(2) {
        if bytes[i] != b'\\' || (bytes[i + 1] != b'p' && bytes[i + 1] != b'P') {
            continue;
        }
        let mut end = i + 2;
        while end < bytes.len() && bytes[end].is_ascii_digit() {
            end += 1;
        }
        if end == i + 2 {
            continue; // `\pos`, `\pbo`, a bare `\p` -- no argument.
        }
        // Saturating: a pathological `\p999999999999` is still just "on".
        found = Some(block[i + 2..end].parse::<u32>().unwrap_or(u32::MAX));
    }
    found
}

/// Applies ASS text cleanup: strips `{...}` override blocks (a literal
/// `\{` survives) and any drawing-mode span they open (see [`drawing_arg`]),
/// then converts `\N`/`\n` to a newline and `\h` to a space.
///
/// Only the drawing span itself (from `\p<nonzero>` to `\p0` or line end) is
/// dropped -- text before and after it is kept.
fn clean_text(raw: &str) -> String {
    let mut stripped = String::with_capacity(raw.len());
    let mut chars = raw.chars().peekable();
    let mut drawing = false;
    while let Some(c) = chars.next() {
        if c == '\\' && chars.peek() == Some(&'{') {
            // Literal `\{` is not an override block; keep the brace.
            chars.next();
            if !drawing {
                stripped.push('{');
            }
            continue;
        }
        if c == '{' {
            // Collect to the matching `}`, if any, so the block's own tags
            // can be inspected before it is discarded.
            let mut block = String::new();
            for skipped in chars.by_ref() {
                if skipped == '}' {
                    break;
                }
                block.push(skipped);
            }
            if let Some(arg) = drawing_arg(&block) {
                drawing = arg != 0;
            }
            continue;
        }
        if drawing {
            continue; // path coordinates, not dialogue
        }
        stripped.push(c);
    }

    let mut out = String::with_capacity(stripped.len());
    let mut chars = stripped.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.peek() {
                Some('N') | Some('n') => {
                    chars.next();
                    out.push('\n');
                    continue;
                }
                Some('h') => {
                    chars.next();
                    out.push(' ');
                    continue;
                }
                _ => {}
            }
        }
        out.push(c);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    const HEAD: &str = "[Events]\nFormat: Layer, Start, End, Style, Name, \
MarginL, MarginR, MarginV, Effect, Text\n";

    #[test]
    fn parses_a_dialogue_line() {
        let s = parse(&format!(
            "{HEAD}Dialogue: 0,0:00:01.00,0:00:02.50,Default,,0,0,0,,Hello\n"
        ))
        .unwrap();
        assert_eq!(s.at(1.5), "Hello");
        assert_eq!(s.at(2.6), "");
    }

    #[test]
    fn centiseconds_not_milliseconds() {
        // ASS timestamps are H:MM:SS.cc -- ".50" is half a second, not 50 ms.
        let s = parse(&format!(
            "{HEAD}Dialogue: 0,0:00:01.00,0:00:01.50,Default,,0,0,0,,x\n"
        ))
        .unwrap();
        assert_eq!(s.at(1.4), "x", "cue must still be live at 1.4 s");
        assert_eq!(s.at(1.6), "", "and over by 1.6 s");
    }

    #[test]
    fn strips_override_blocks() {
        let s = parse(&format!(
            "{HEAD}Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{{\\an8\\i1}}Hi{{\\i0}}\n"
        ))
        .unwrap();
        assert_eq!(s.at(1.5), "Hi");
    }

    #[test]
    fn converts_hard_line_breaks_and_hard_spaces() {
        let s = parse(&format!(
            "{HEAD}Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,a\\Nb\\hc\n"
        ))
        .unwrap();
        assert_eq!(s.at(1.5), "a\nb c");
    }

    #[test]
    fn text_containing_commas_is_not_truncated() {
        // Text is the LAST field and may contain commas; splitting naively
        // would cut it at the first one.
        let s = parse(&format!(
            "{HEAD}Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,Well, hello, there\n"
        ))
        .unwrap();
        assert_eq!(s.at(1.5), "Well, hello, there");
    }

    #[test]
    fn field_order_is_read_from_the_format_line_not_assumed() {
        let s = parse(
            "[Events]\nFormat: Start, End, Text\n\
             Dialogue: 0:00:01.00,0:00:02.00,Hello\n",
        )
        .unwrap();
        assert_eq!(s.at(1.5), "Hello");
    }

    #[test]
    fn comment_lines_are_ignored() {
        let s = parse(&format!(
            "{HEAD}Comment: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,nope\n"
        ))
        .unwrap();
        assert!(s.is_empty(), "Comment: is not a subtitle");
    }

    #[test]
    fn a_missing_format_line_errors_rather_than_guessing_field_positions() {
        let err = parse("[Events]\nDialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,x\n")
            .expect_err("must reject");
        assert!(err.contains("Format"), "{err}");
    }

    #[test]
    fn a_utf8_bom_before_the_events_header_does_not_empty_the_track() {
        // `str::trim` does not remove U+FEFF (it is Cf, not White_Space), so
        // `\u{feff}[Events]` used to fail the section test and every
        // Dialogue line was skipped, yielding Ok with zero cues.
        let s = parse(
            "\u{feff}[Events]\nFormat: Start, End, Text\n\
             Dialogue: 0:00:01.00,0:00:02.00,Hello\n",
        )
        .unwrap();
        assert_eq!(s.len(), 1, "the BOM must not swallow the whole file");
        assert_eq!(s.at(1.5), "Hello");
    }

    #[test]
    fn a_section_header_with_a_trailing_comment_still_enters_the_section() {
        let s = parse(
            "[Events] ; the dialogue follows\nFormat: Start, End, Text\n\
             Dialogue: 0:00:01.00,0:00:02.00,Hello\n",
        )
        .unwrap();
        assert_eq!(s.at(1.5), "Hello");
    }

    #[test]
    fn a_file_with_no_events_section_errors_rather_than_parsing_to_nothing() {
        let err = parse("[Script Info]\nTitle: nothing\n").expect_err("must reject");
        assert!(err.contains("[Events]"), "names what is missing: {err}");
    }

    #[test]
    fn an_events_section_with_no_dialogue_is_a_valid_empty_track() {
        // The file explicitly said there is nothing here, which is different
        // from the parser having failed to find the section at all.
        assert!(parse("[Events]\nFormat: Start, End, Text\n").unwrap().is_empty());
    }

    #[test]
    fn dialogue_mixed_with_a_drawing_keeps_the_dialogue() {
        // `\p1` opens a drawing span; only that SPAN is path data. Dropping
        // the whole cue lost "Real line" with no error at all.
        let s = parse(&format!(
            "{HEAD}Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,\
             Real line{{\\p1}}m 0 0 l 9 9\n"
        ))
        .unwrap();
        assert_eq!(s.at(1.5), "Real line", "the spoken half must survive");
    }

    #[test]
    fn a_drawing_span_ends_at_p0_and_text_after_it_is_kept() {
        let s = parse(&format!(
            "{HEAD}Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,\
             before{{\\p1}}m 0 0 l 9 9{{\\p0}}after\n"
        ))
        .unwrap();
        assert_eq!(s.at(1.5), "beforeafter", "only the span between is path data");
    }

    #[test]
    fn a_backslash_p_in_ordinary_text_is_not_a_drawing_tag() {
        // Drawing mode is an override tag -- it can only be set inside `{}`.
        // Scanning the raw field dropped these cues entirely.
        for text in ["it costs \\p1 per unit", "C:\\pics\\p1", "a\\p2b"] {
            let s = parse(&format!(
                "{HEAD}Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{text}\n"
            ))
            .unwrap();
            assert_eq!(s.len(), 1, "{text:?} is dialogue, not a drawing");
            assert!(!s.at(1.5).is_empty(), "{text:?} must keep its text");
        }
    }

    #[test]
    fn the_last_p_tag_in_a_block_decides_drawing_mode() {
        let s = parse(&format!(
            "{HEAD}Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{{\\p1\\p0}}Hello\n"
        ))
        .unwrap();
        assert_eq!(s.at(1.5), "Hello", "the trailing \\p0 turned drawing back off");
    }

    #[test]
    fn an_empty_cue_is_dropped_rather_than_blanking_an_overlapping_line() {
        // An empty cue would be the most recently begun cue in its window and
        // would therefore win `Subtitles::at`'s overlap rule, hiding the line
        // that is actually speaking underneath it.
        let s = parse(&format!(
            "{HEAD}Dialogue: 0,0:00:01.00,0:00:04.00,Default,,0,0,0,,Speaking\n\
             Dialogue: 0,0:00:02.00,0:00:03.00,Default,,0,0,0,,{{\\p1}}m 0 0 l 9 9{{\\p0}}\n"
        ))
        .unwrap();
        assert_eq!(s.len(), 1, "the pure drawing must not become a cue");
        assert_eq!(s.at(2.5), "Speaking", "and must not blank the dialogue under it");
    }

    #[test]
    fn vector_drawings_are_dropped_rather_than_emitted_as_coordinate_soup() {
        // `\p1` switches to drawing mode; the path data that follows sits
        // between the override blocks, so brace-stripping alone leaves it
        // behind as text.
        let s = parse(&format!(
            "{HEAD}Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,\
             {{\\p1}}m 0 0 l 100 0 l 100 100{{\\p0}}\n"
        ))
        .unwrap();
        assert!(s.is_empty(), "a drawing is not dialogue, got {:?}", s.at(1.5));
    }

    #[test]
    fn nonsense_timestamps_are_refused_rather_than_becoming_undisplayable_cues() {
        for bad in ["inf:00:00.00", "NaN:00:00.00", "-0:00:01.00", "1e3:00:00.00"] {
            let err = parse(&format!(
                "{HEAD}Dialogue: 0,{bad},0:00:09.00,Default,,0,0,0,,x\n"
            ))
            .expect_err(&format!("{bad:?} must be rejected"));
            assert!(err.contains("timestamp"), "{err}");
        }
    }

    #[test]
    fn a_reversed_cue_errors_rather_than_never_displaying() {
        let err = parse(&format!(
            "{HEAD}Dialogue: 0,0:00:05.00,0:00:02.00,Default,,0,0,0,,x\n"
        ))
        .expect_err("must reject");
        assert!(err.to_lowercase().contains("before it starts"), "{err}");
    }

    #[test]
    fn pos_is_not_mistaken_for_a_drawing() {
        // `\pos(...)` is positioning, not drawing. The non-zero-digit check is
        // what separates them; treating every `\p*` as a drawing would
        // silently discard ordinary positioned dialogue.
        let s = parse(&format!(
            "{HEAD}Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{{\\pos(10,20)}}Hello\n"
        ))
        .unwrap();
        assert_eq!(s.at(1.5), "Hello");
    }

    #[test]
    fn an_explicit_p0_means_drawing_mode_off_and_is_kept() {
        let s = parse(&format!(
            "{HEAD}Dialogue: 0,0:00:01.00,0:00:02.00,Default,,0,0,0,,{{\\p0}}Hello\n"
        ))
        .unwrap();
        assert_eq!(s.at(1.5), "Hello");
    }
}
