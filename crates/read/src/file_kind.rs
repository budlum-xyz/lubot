#![forbid(unsafe_code)]
//! # file_kind - know what you are reading before you read it
//!
//! Know the kind before the content, encoded as routing: magic
//! bytes decide the kind, and the kind decides the route. A file whose kind
//! has no reading path is *refused* before any extraction - reading a
//! container as text, or an executable as a document, is how a reader
//! hallucinates. Magic bytes only: no extension trust, no content sniffing
//! beyond the first few bytes.

/// What the first bytes say a file is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FileKind {
    Pdf,
    Gzip,
    Zip,
    Png,
    Jpeg,
    Elf,
    Text,
    OtherBinary,
}

/// The reading path a kind admits.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Route {
    /// `lubot doc`: extract and chunk the text inside.
    Doc,
    /// `lubot corpus`: load as a (`jsonl.gz`) corpus file.
    Corpus,
    /// Plain text: read directly.
    Text,
    /// No reading path; the reason is named.
    Refuse { reason: &'static str },
}

/// Recognise a file from its leading magic bytes.
#[must_use]
pub fn file_kind(bytes: &[u8]) -> FileKind {
    if bytes.starts_with(b"%PDF-") {
        return FileKind::Pdf;
    }
    if bytes.starts_with(&[0x1f, 0x8b]) {
        return FileKind::Gzip;
    }
    if bytes.starts_with(b"PK\x03\x04") {
        return FileKind::Zip;
    }
    if bytes.starts_with(&[0x89, 0x50, 0x4e, 0x47]) {
        return FileKind::Png;
    }
    if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
        return FileKind::Jpeg;
    }
    if bytes.starts_with(b"\x7fELF") {
        return FileKind::Elf;
    }
    if std::str::from_utf8(bytes).is_ok() {
        return FileKind::Text;
    }
    FileKind::OtherBinary
}

/// The route a file takes, given its kind and name. The name matters only
/// for gzip: only a `*.jsonl.gz` corpus name is a corpus; any other gzip is
/// refused, because decompressing into something unread is not reading.
#[must_use]
pub fn route(kind: FileKind, name: &str) -> Route {
    match kind {
        FileKind::Pdf => Route::Doc,
        FileKind::Gzip if name.ends_with(".jsonl.gz") => Route::Corpus,
        FileKind::Gzip => Route::Refuse {
            reason: "gzip container: not a corpus name",
        },
        FileKind::Zip => Route::Refuse {
            reason: "zip archive: extract first",
        },
        FileKind::Png | FileKind::Jpeg => Route::Refuse {
            reason: "image: no decoder on this reading path",
        },
        FileKind::Elf => Route::Refuse {
            reason: "executable: not a reading target",
        },
        FileKind::Text => Route::Text,
        FileKind::OtherBinary => Route::Refuse {
            reason: "binary content",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_magic_bytes_are_recognised() {
        assert_eq!(file_kind(b"%PDF-1.7 ..."), FileKind::Pdf);
        assert_eq!(file_kind(&[0x1f, 0x8b, 0x08, 0x00]), FileKind::Gzip);
        assert_eq!(file_kind(b"PK\x03\x04rest"), FileKind::Zip);
        assert_eq!(
            file_kind(&[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a]),
            FileKind::Png
        );
        assert_eq!(file_kind(&[0xff, 0xd8, 0xff, 0xe0]), FileKind::Jpeg);
        assert_eq!(file_kind(b"\x7fELF\x02\x01"), FileKind::Elf);
        assert_eq!(file_kind("merhaba".as_bytes()), FileKind::Text);
        assert_eq!(file_kind(&[0xff, 0xfe, 0x00, 0x01]), FileKind::OtherBinary);
    }

    #[test]
    fn the_routes_refuse_what_has_no_reading_path() {
        assert_eq!(route(FileKind::Pdf, "a.pdf"), Route::Doc);
        assert_eq!(route(FileKind::Gzip, "korpus.jsonl.gz"), Route::Corpus);
        assert!(matches!(
            route(FileKind::Gzip, "veri.tar.gz"),
            Route::Refuse { .. }
        ));
        assert!(matches!(route(FileKind::Elf, "bud"), Route::Refuse { .. }));
        assert!(matches!(
            route(FileKind::Zip, "a.zip"),
            Route::Refuse { .. }
        ));
        assert!(matches!(
            route(FileKind::Png, "a.png"),
            Route::Refuse { .. }
        ));
        assert_eq!(route(FileKind::Text, "not.md"), Route::Text);
    }

    #[test]
    fn a_text_file_is_read_but_binary_is_not() {
        assert_eq!(file_kind(b"{\"kind\": \"markdown\"}"), FileKind::Text);
        assert!(matches!(
            route(file_kind(&[0xde, 0xad, 0xbe, 0xef]), "f"),
            Route::Refuse { .. }
        ));
    }
}
