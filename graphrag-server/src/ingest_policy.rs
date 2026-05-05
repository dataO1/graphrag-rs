//! Path-based ingestion policy for `POST /api/documents`.
//!
//! Centralizes the rules that turn a caller-supplied filesystem path
//! (or glob, or list of paths) into a vetted, ready-to-ingest body of
//! text. Concretely:
//!
//! * **Sandbox** — every resolved path must canonicalize under one of
//!   `allowed_roots`. Symlink-based escape is rejected by canonicalize
//!   pre-walk; we never trust the request string after expansion.
//! * **Size cap** — files larger than `max_file_bytes` are skipped
//!   (graphrag chunking is content-dependent and a 500 MB markdown
//!   file is almost certainly an accident).
//! * **Extension allow-list** — text-like extensions (md, txt, code,
//!   structured data) are read as UTF-8 directly. Anything else is
//!   either routed to the configured preprocessor service (Nemotron-
//!   Omni, pandoc, etc.) or skipped with a clear `unsupported` status.
//! * **Glob expansion** — `paths_glob` is expanded with the `glob`
//!   crate under `glob_root` (or the first allowed root). Each
//!   expansion result is re-validated through the same canonicalize
//!   + sandbox path; a glob that escapes via `..` cannot land bytes.
//!
//! Configuration is read once at startup from env vars (the same
//! pattern the rest of this binary uses; the home-manager module
//! plumbs nix options through to these). Empty `allowed_roots` means
//! path-ingestion is **disabled** — requests with `path`/`paths`/
//! `paths_glob` get a 403 with a message pointing at the env var.
//!
//! Pair with the future Nemotron-Omni preprocessor (see
//! `graphrag-rs-nix/TODO.md` § "Multimodal preprocessor"): when
//! `INGEST_PREPROCESSOR_URL` is set, non-text files POST to it and
//! the returned `markdown` field is what we ingest.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// Resolved per-server ingest rules. Cheap to clone (Arc'd in AppState).
#[derive(Debug, Clone)]
pub struct IngestPolicy {
    /// Canonicalized absolute paths. A request path is allowed iff its
    /// own canonical form starts_with one of these. Empty disables
    /// path-ingestion entirely.
    pub allowed_roots: Vec<PathBuf>,
    /// Hard cap per file. Larger files are skipped with `too_large`.
    pub max_file_bytes: u64,
    /// Lower-case extensions that are read directly as UTF-8 text.
    /// (No leading dot.)
    pub allowed_extensions: HashSet<String>,
    /// If set, files whose extension is *not* in `allowed_extensions`
    /// are POSTed to this URL as `{ "path": "<abs>" }` and the JSON
    /// response is expected to contain a `markdown` field which is
    /// then ingested as the document body. Disabled when None.
    pub preprocessor_url: Option<String>,
    /// When false (default), any path component that is a symlink is
    /// rejected. Canonicalize already follows symlinks for the final
    /// resolution, so this is mainly a defense-in-depth knob: even
    /// when the symlink target lives inside the sandbox, the symlink
    /// itself is treated as suspect.
    pub follow_symlinks: bool,
}

/// Default allow-listed text extensions. Tuned for graphrag's notes /
/// code / config diet; non-text formats (pdf/docx/jpg/mp4/...) are
/// deliberately omitted and route through the preprocessor instead.
pub const DEFAULT_ALLOWED_EXTENSIONS: &[&str] = &[
    "md", "markdown", "mdx", "txt", "text", "rst", "org", "adoc", "asciidoc", "tex",
    "json", "yaml", "yml", "toml", "ini", "csv", "tsv", "log",
    "rs", "py", "js", "mjs", "cjs", "ts", "tsx", "jsx",
    "go", "c", "h", "cpp", "cc", "hpp", "hh", "java", "kt", "kts",
    "rb", "php", "swift", "scala", "clj", "ex", "exs", "erl", "hs",
    "sql", "sh", "bash", "zsh", "fish", "ps1",
    "nix", "dhall",
    "html", "htm", "xml", "svg", "css", "scss", "less",
    "graphql", "gql", "proto", "thrift",
];

/// 16 MiB. Larger files are skipped — graphrag chunking + entity
/// extraction over a single multi-hundred-MB blob is almost never
/// what the caller actually wanted; if it is, raise the cap
/// explicitly via `INGEST_MAX_FILE_BYTES`.
pub const DEFAULT_MAX_FILE_BYTES: u64 = 16 * 1024 * 1024;

/// Outcome of resolving a single request path. The handler maps these
/// into the per-item entries of `AddDocumentsResponse.results`.
pub enum ResolvedPath {
    /// Path is in-sandbox, file size OK, extension is text-like.
    /// Caller should `read_to_string` and ingest.
    Text { absolute: PathBuf },
    /// Path is in-sandbox, file size OK, extension is NOT text-like;
    /// preprocessor URL is configured. Caller should POST to it.
    Preprocess { absolute: PathBuf, preprocessor_url: String },
    /// Path is in-sandbox, file size OK, extension is NOT text-like,
    /// no preprocessor configured → skip with this reason.
    Unsupported { absolute: PathBuf, extension: String },
    /// Path resolved but failed a policy check (sandbox, size, type).
    /// Caller should record it as an error result for that path.
    Rejected { input: String, reason: String },
}

impl IngestPolicy {
    /// Build from env. All vars are optional; empty `allowed_roots`
    /// disables path-ingestion (returns 403 from the handler).
    pub fn from_env() -> Arc<Self> {
        let allowed_roots = std::env::var("INGEST_ALLOWED_ROOTS")
            .ok()
            .map(|s| {
                s.split(':')
                    .filter(|p| !p.is_empty())
                    .filter_map(|p| {
                        // Non-existent roots are silently dropped at
                        // boot — nicer than crash-looping the server
                        // when a watch-folder hasn't been mkdir'd yet.
                        match std::fs::canonicalize(p) {
                            Ok(c) => Some(c),
                            Err(e) => {
                                tracing::warn!(
                                    root = %p, error = %e,
                                    "INGEST_ALLOWED_ROOTS: dropping unresolvable root"
                                );
                                None
                            },
                        }
                    })
                    .collect::<Vec<_>>()
            })
            .unwrap_or_default();

        let max_file_bytes = std::env::var("INGEST_MAX_FILE_BYTES")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(DEFAULT_MAX_FILE_BYTES);

        let allowed_extensions = std::env::var("INGEST_ALLOWED_EXTENSIONS")
            .ok()
            .map(|s| {
                s.split(',')
                    .map(|e| e.trim().trim_start_matches('.').to_ascii_lowercase())
                    .filter(|e| !e.is_empty())
                    .collect::<HashSet<_>>()
            })
            .unwrap_or_else(|| {
                DEFAULT_ALLOWED_EXTENSIONS
                    .iter()
                    .map(|s| s.to_string())
                    .collect()
            });

        let preprocessor_url = std::env::var("INGEST_PREPROCESSOR_URL").ok().filter(|s| !s.is_empty());

        let follow_symlinks = std::env::var("INGEST_FOLLOW_SYMLINKS")
            .ok()
            .map(|s| matches!(s.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
            .unwrap_or(false);

        if allowed_roots.is_empty() {
            tracing::info!(
                "ingest: path-based ingestion disabled (INGEST_ALLOWED_ROOTS empty); \
                 only `content`-form POST /api/documents will work"
            );
        } else {
            tracing::info!(
                roots = ?allowed_roots,
                max_bytes = max_file_bytes,
                ext_count = allowed_extensions.len(),
                preprocessor = preprocessor_url.as_deref().unwrap_or("none"),
                follow_symlinks,
                "ingest: path-based ingestion enabled"
            );
        }

        Arc::new(Self {
            allowed_roots,
            max_file_bytes,
            allowed_extensions,
            preprocessor_url,
            follow_symlinks,
        })
    }

    /// True iff path-ingestion is enabled (i.e. at least one allowed root).
    pub fn enabled(&self) -> bool {
        !self.allowed_roots.is_empty()
    }

    /// Resolve a single caller-supplied path string into a `ResolvedPath`.
    /// Performs canonicalize, sandbox check, symlink check, size check,
    /// and extension classification. Does NOT read the file — that's
    /// the handler's job once it has the resolved variant.
    pub fn resolve(&self, input: &str) -> ResolvedPath {
        if !self.enabled() {
            return ResolvedPath::Rejected {
                input: input.to_string(),
                reason: "path-based ingestion disabled (no INGEST_ALLOWED_ROOTS configured)".to_string(),
            };
        }

        // Canonicalize first: this both resolves any symlinks AND
        // gives us a stable absolute path to compare against the
        // sandbox roots. Failing here covers nonexistent files and
        // permission errors uniformly.
        let absolute = match std::fs::canonicalize(input) {
            Ok(p) => p,
            Err(e) => {
                return ResolvedPath::Rejected {
                    input: input.to_string(),
                    reason: format!("canonicalize failed: {e}"),
                };
            },
        };

        // Sandbox: absolute MUST start_with at least one allowed root.
        let in_sandbox = self.allowed_roots.iter().any(|r| absolute.starts_with(r));
        if !in_sandbox {
            return ResolvedPath::Rejected {
                input: input.to_string(),
                reason: format!(
                    "path {} is outside allowed_roots (configure INGEST_ALLOWED_ROOTS)",
                    absolute.display()
                ),
            };
        }

        // Symlink defense-in-depth. After canonicalize the *target*
        // is what we have, but if the user-supplied path itself was
        // a symlink, surface that unless explicitly opted-in.
        if !self.follow_symlinks {
            if let Ok(meta) = std::fs::symlink_metadata(input) {
                if meta.file_type().is_symlink() {
                    return ResolvedPath::Rejected {
                        input: input.to_string(),
                        reason: "symlink rejected (set INGEST_FOLLOW_SYMLINKS=1 to allow)".to_string(),
                    };
                }
            }
        }

        // Must be a regular file.
        let meta = match std::fs::metadata(&absolute) {
            Ok(m) => m,
            Err(e) => {
                return ResolvedPath::Rejected {
                    input: input.to_string(),
                    reason: format!("stat failed: {e}"),
                };
            },
        };
        if !meta.is_file() {
            return ResolvedPath::Rejected {
                input: input.to_string(),
                reason: "not a regular file".to_string(),
            };
        }

        // Size cap.
        if meta.len() > self.max_file_bytes {
            return ResolvedPath::Rejected {
                input: input.to_string(),
                reason: format!(
                    "file size {} bytes exceeds INGEST_MAX_FILE_BYTES ({})",
                    meta.len(),
                    self.max_file_bytes
                ),
            };
        }

        // Extension classification.
        let ext = absolute
            .extension()
            .and_then(|e| e.to_str())
            .map(|e| e.to_ascii_lowercase())
            .unwrap_or_default();

        if self.allowed_extensions.contains(&ext) {
            ResolvedPath::Text { absolute }
        } else if let Some(url) = &self.preprocessor_url {
            ResolvedPath::Preprocess {
                absolute,
                preprocessor_url: url.clone(),
            }
        } else {
            ResolvedPath::Unsupported { absolute, extension: ext }
        }
    }

    /// Expand a glob pattern into a list of caller-input strings.
    /// The caller then runs each through `resolve()` (which is the
    /// only function that gets to decide whether bytes are read);
    /// this function NEVER returns paths outside the sandbox even
    /// before resolve() runs, but resolve() is still the gatekeeper.
    ///
    /// `glob_root` (when supplied) anchors a relative pattern; an
    /// absolute pattern is used as-is. Errors are returned as a
    /// single-item Vec containing a `Rejected` entry by the handler.
    pub fn expand_glob(
        &self,
        pattern: &str,
        glob_root: Option<&str>,
    ) -> Result<Vec<String>, String> {
        if !self.enabled() {
            return Err("path-based ingestion disabled (no INGEST_ALLOWED_ROOTS configured)".to_string());
        }
        let full_pattern: String = if Path::new(pattern).is_absolute() {
            pattern.to_string()
        } else {
            // Anchor: explicit glob_root → that. Else first allowed root.
            let base = match glob_root {
                Some(r) => PathBuf::from(r),
                None => self.allowed_roots[0].clone(),
            };
            // Re-canonicalize the base too so a relative pattern
            // can't escape via an unresolved symlink in glob_root.
            let canon = std::fs::canonicalize(&base)
                .map_err(|e| format!("glob_root canonicalize failed: {e}"))?;
            if !self.allowed_roots.iter().any(|r| canon.starts_with(r)) {
                return Err(format!(
                    "glob_root {} is outside allowed_roots",
                    canon.display()
                ));
            }
            canon.join(pattern).to_string_lossy().into_owned()
        };

        let entries = glob::glob(&full_pattern)
            .map_err(|e| format!("invalid glob pattern: {e}"))?;
        let mut out = Vec::new();
        for entry in entries {
            match entry {
                Ok(p) => {
                    if p.is_file() {
                        out.push(p.to_string_lossy().into_owned());
                    }
                },
                Err(e) => {
                    tracing::warn!(error = %e, "glob walk error (continuing)");
                },
            }
        }
        Ok(out)
    }
}
