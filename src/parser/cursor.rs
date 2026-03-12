use crate::session::{Message, Role, Session, SessionSource};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use super::{extract_text_content, join_consecutive_messages, SessionParser};

/// Cursor Agent CLI transcript line format
/// Each line: {"role": "user"|"assistant", "message": {"content": [{"type": "text", "text": "..."}]}}
#[derive(Debug, Deserialize)]
struct CursorLine {
    role: String,
    message: Option<CursorMessage>,
}

#[derive(Debug, Deserialize)]
struct CursorMessage {
    content: serde_json::Value,
}

pub struct CursorParser;

impl SessionParser for CursorParser {
    fn can_parse(path: &Path) -> bool {
        let s = path.to_str().unwrap_or("");
        (s.contains(".cursor/projects") || s.contains(".cursor\\projects"))
            && s.contains("agent-transcripts")
            && path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e == "jsonl")
                .unwrap_or(false)
    }

    fn parse_file(path: &Path) -> Result<Session> {
        let file = File::open(path).context("Failed to open file")?;
        let reader = BufReader::with_capacity(64 * 1024, file);

        // Session ID = filename without extension (UUID)
        let session_id = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Resolve cwd from the project directory name
        let cwd = path
            .ancestors()
            .find(|p| {
                p.file_name()
                    .and_then(|n| n.to_str())
                    .map(|n| n == "agent-transcripts")
                    .unwrap_or(false)
            })
            .and_then(|transcripts_dir| transcripts_dir.parent())
            .and_then(|project_dir| project_dir.file_name())
            .and_then(|name| name.to_str())
            .and_then(decode_workspace_path)
            .unwrap_or_else(|| ".".to_string());

        let mut messages: Vec<Message> = Vec::new();

        // Use file mtime as timestamp (transcripts don't have per-message timestamps)
        let file_ts = std::fs::metadata(path)
            .ok()
            .and_then(|m| m.modified().ok())
            .map(DateTime::<Utc>::from)
            .unwrap_or_else(Utc::now);

        for line in reader.lines() {
            let line = line.context("Failed to read line")?;
            if line.trim().is_empty() {
                continue;
            }

            let entry: CursorLine = match serde_json::from_str(&line) {
                Ok(e) => e,
                Err(_) => continue,
            };

            let role = match entry.role.as_str() {
                "user" => Role::User,
                "assistant" => Role::Assistant,
                _ => continue,
            };

            let content = entry
                .message
                .as_ref()
                .map(|m| extract_text_content(&m.content))
                .unwrap_or_default();

            if content.is_empty() {
                continue;
            }

            messages.push(Message {
                role,
                content,
                timestamp: file_ts,
            });
        }

        Ok(Session {
            id: session_id,
            source: SessionSource::CursorCli,
            file_path: path.to_path_buf(),
            cwd,
            git_branch: None,
            timestamp: file_ts,
            messages: join_consecutive_messages(messages),
        })
    }
}


/// Encode a filesystem path the same way Cursor does:
/// replace all non-alphanumeric chars with `-`, then collapse consecutive `-`.
fn cursor_encode_path(path: &str) -> String {
    let replaced = path
        .trim_start_matches('/')
        .replace(|c: char| !c.is_alphanumeric(), "-");
    let mut prev_dash = false;
    replaced
        .chars()
        .filter(|&c| {
            if c == '-' {
                if prev_dash {
                    return false;
                }
                prev_dash = true;
            } else {
                prev_dash = false;
            }
            true
        })
        .collect()
}

/// Decode a Cursor workspace directory name to a filesystem path.
/// Cursor encodes all non-alphanumeric chars as `-`, collapses runs of `-`.
/// Strategy: walk the filesystem, encoding each candidate path and comparing
/// against the target name. This avoids exponential DFS over separator guesses.
pub fn decode_workspace_path(name: &str) -> Option<String> {
    let parts: Vec<&str> = name.split('-').collect();
    if parts.is_empty() {
        return None;
    }

    // Start from the first component (e.g., "Users", "var", "home")
    let start = PathBuf::from("/").join(parts[0]);
    if !start.exists() {
        return None;
    }

    let mut best: Option<PathBuf> = None;
    walk_decode(name, &start, &mut best);
    best.map(|p| p.to_string_lossy().into_owned())
}

/// Recursively walk directories, checking if encoding each path matches the target name.
fn walk_decode(target: &str, current: &Path, best: &mut Option<PathBuf>) {
    let encoded = cursor_encode_path(&current.to_string_lossy());

    // If current path's encoding matches the target exactly, we found it
    if encoded == target {
        *best = Some(current.to_path_buf());
        return;
    }

    // If the target doesn't start with our encoding, prune this branch
    if !target.starts_with(&encoded) || !target[encoded.len()..].starts_with('-') {
        return;
    }

    // List children and recurse
    let entries = match std::fs::read_dir(current) {
        Ok(e) => e,
        Err(_) => return,
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            walk_decode(target, &path, best);
            if best.is_some() {
                return; // Exact match found, stop early
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_parse_cursor_path() {
        assert!(CursorParser::can_parse(Path::new(
            "/home/user/.cursor/projects/myproject/agent-transcripts/abc-123.jsonl"
        )));
    }

    #[test]
    fn test_can_parse_rejects_other() {
        assert!(!CursorParser::can_parse(Path::new(
            "/home/user/.claude/projects/test/session.jsonl"
        )));
        assert!(!CursorParser::can_parse(Path::new(
            "/home/user/.cursor/chats/abc-123/store.db"
        )));
    }

    #[test]
    fn test_parse_cursor_transcript() {
        let dir = tempfile::TempDir::new().unwrap();
        let transcripts_dir = dir
            .path()
            .join(".cursor/projects/test-project/agent-transcripts");
        std::fs::create_dir_all(&transcripts_dir).unwrap();
        let jsonl_path = transcripts_dir.join("abc-123-def.jsonl");

        let content = r#"{"role":"user","message":{"content":[{"type":"text","text":"Hello cursor"}]}}
{"role":"assistant","message":{"content":[{"type":"text","text":"Hi there!"}]}}
{"role":"system","message":{"content":"system prompt"}}
"#;
        std::fs::write(&jsonl_path, content).unwrap();

        let session = CursorParser::parse_file(&jsonl_path).unwrap();
        assert_eq!(session.id, "abc-123-def");
        assert_eq!(session.source, SessionSource::CursorCli);
        assert_eq!(session.messages.len(), 2); // system skipped
        assert_eq!(session.messages[0].role, Role::User);
        assert_eq!(session.messages[0].content, "Hello cursor");
        assert_eq!(session.messages[1].role, Role::Assistant);
        assert_eq!(session.messages[1].content, "Hi there!");
    }

    #[test]
    fn test_parse_cursor_consecutive_messages() {
        let dir = tempfile::TempDir::new().unwrap();
        let transcripts_dir = dir
            .path()
            .join(".cursor/projects/test-project/agent-transcripts");
        std::fs::create_dir_all(&transcripts_dir).unwrap();
        let jsonl_path = transcripts_dir.join("test-join.jsonl");

        let content = r#"{"role":"user","message":{"content":[{"type":"text","text":"Part 1"}]}}
{"role":"assistant","message":{"content":[{"type":"text","text":"Response A"}]}}
{"role":"assistant","message":{"content":[{"type":"text","text":"Response B"}]}}
"#;
        std::fs::write(&jsonl_path, content).unwrap();

        let session = CursorParser::parse_file(&jsonl_path).unwrap();
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[1].content, "Response A\n\nResponse B");
    }

    #[test]
    fn test_decode_workspace_path() {
        let dir = tempfile::TempDir::new().unwrap();
        let nested = dir.path().join("a/b.c/d");
        std::fs::create_dir_all(&nested).unwrap();

        let encoded = cursor_encode_path(&nested.to_string_lossy());
        let result = decode_workspace_path(&encoded);
        assert_eq!(result, Some(nested.to_string_lossy().into_owned()));
    }

    #[test]
    fn test_decode_workspace_path_simple() {
        let dir = tempfile::TempDir::new().unwrap();
        let project = dir.path().join("myproject");
        std::fs::create_dir_all(&project).unwrap();

        let encoded = cursor_encode_path(&project.to_string_lossy());
        let result = decode_workspace_path(&encoded);
        assert_eq!(result, Some(project.to_string_lossy().into_owned()));
    }

    #[test]
    fn test_decode_workspace_path_with_hyphen() {
        let dir = tempfile::TempDir::new().unwrap();
        let project = dir.path().join("my-project/sub-dir");
        std::fs::create_dir_all(&project).unwrap();

        let encoded = cursor_encode_path(&project.to_string_lossy());
        let result = decode_workspace_path(&encoded);
        assert_eq!(result, Some(project.to_string_lossy().into_owned()));
    }
}
