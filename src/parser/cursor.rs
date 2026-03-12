use crate::session::{Message, Role, Session, SessionSource};
use anyhow::{Context, Result};
use chrono::{TimeZone, Utc};
use rusqlite::Connection;
use serde::Deserialize;
use std::path::{Path, PathBuf};

use super::{join_consecutive_messages, SessionParser};

/// Legacy bubble format (older Cursor versions)
#[derive(Debug, Deserialize)]
struct CursorBubble {
    #[serde(rename = "type")]
    bubble_type: Option<i32>,
    text: Option<String>,
    #[serde(rename = "richText")]
    rich_text: Option<String>,
    timestamp: Option<i64>,
}

/// OpenAI-compatible message format (current Cursor Agent CLI)
#[derive(Debug, Deserialize)]
struct OpenAIMessage {
    role: Option<String>,
    content: Option<serde_json::Value>,
}

pub struct CursorParser;

impl SessionParser for CursorParser {
    fn can_parse(path: &Path) -> bool {
        let s = path.to_str().unwrap_or("");
        (s.contains(".cursor/chats") || s.contains(".cursor\\chats")
            || s.contains(".config/cursor/chats") || s.contains(".config\\cursor\\chats"))
            && (s.ends_with("/store.db") || s.ends_with("\\store.db") || s.ends_with("store.db"))
    }

    fn parse_file(path: &Path) -> Result<Session> {
        let conn = Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .context("Failed to open Cursor SQLite database")?;

        let session_id = path
            .parent()
            .and_then(|p| p.file_name())
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
            .to_string();

        // Read metadata from meta table (try key/value then id/data schema)
        let meta_value: Option<serde_json::Value> = conn
            .query_row("SELECT value FROM meta WHERE key = '0'", [], |row| {
                row.get::<_, String>(0)
            })
            .or_else(|_| {
                conn.query_row("SELECT data FROM meta WHERE id = '0'", [], |row| {
                    row.get::<_, String>(0)
                })
            })
            .ok()
            .and_then(|s| decode_meta_value(&s));

        let created_at = meta_value
            .as_ref()
            .and_then(|v| v["createdAt"].as_i64())
            .and_then(|ms| Utc.timestamp_millis_opt(ms).single());

        let cwd = meta_value
            .as_ref()
            .and_then(|v| v["cwd"].as_str().map(|s| s.to_string()));

        // Read messages from blobs table (try OpenAI format first, then legacy bubble format)
        let fallback_ts = created_at.unwrap_or_else(Utc::now);
        let messages = match read_openai_messages(&conn, fallback_ts) {
            Ok(msgs) if !msgs.is_empty() => msgs,
            _ => read_blobs(&conn)
                .map(|bubbles| bubbles_to_messages(&bubbles))
                .unwrap_or_default(),
        };

        // Prefer the latest message timestamp for consistency with other parsers.
        // OpenAI-format messages only have createdAt as fallback (no per-message timestamps).
        let latest_message_ts = messages.iter().map(|m| m.timestamp).max();
        let timestamp = latest_message_ts
            .or(created_at)
            .unwrap_or_else(Utc::now);

        Ok(Session {
            id: session_id,
            source: SessionSource::CursorCli,
            file_path: path.to_path_buf(),
            cwd: cwd.unwrap_or_else(|| ".".to_string()),
            git_branch: None,
            timestamp,
            messages: join_consecutive_messages(messages),
        })
    }
}

/// Decode meta value: try direct JSON, then hex-encoded JSON
fn decode_meta_value(s: &str) -> Option<serde_json::Value> {
    // Try direct JSON parse
    if let Ok(v) = serde_json::from_str(s) {
        return Some(v);
    }
    // Try hex decode then JSON parse
    hex::decode(s)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
}

/// Convert legacy bubbles to Messages
fn bubbles_to_messages(bubbles: &[CursorBubble]) -> Vec<Message> {
    let mut messages = Vec::new();
    for bubble in bubbles {
        let role = match bubble.bubble_type {
            Some(1) => Role::User,
            Some(2) => Role::Assistant,
            _ => continue,
        };
        let content = bubble
            .text
            .clone()
            .or_else(|| bubble.rich_text.clone())
            .unwrap_or_default();
        if content.is_empty() {
            continue;
        }
        let timestamp = bubble
            .timestamp
            .and_then(|ms| Utc.timestamp_millis_opt(ms).single())
            .unwrap_or_else(Utc::now);
        messages.push(Message {
            role,
            content,
            timestamp,
        });
    }
    messages
}

/// Extract text content from OpenAI content field (string or array of content parts)
fn extract_openai_content(content: &serde_json::Value) -> String {
    match content {
        serde_json::Value::String(s) => s.clone(),
        serde_json::Value::Array(arr) => arr
            .iter()
            .filter_map(|part| {
                let t = part.get("type")?.as_str()?;
                if t == "text" {
                    part.get("text")?.as_str().map(|s| s.to_string())
                } else {
                    None
                }
            })
            .collect::<Vec<_>>()
            .join("\n"),
        _ => String::new(),
    }
}

/// Read blobs as OpenAI-compatible messages (current Cursor Agent CLI format)
fn read_openai_messages(conn: &Connection, fallback_ts: chrono::DateTime<Utc>) -> Result<Vec<Message>> {
    // Try id/data schema first (more common in current Cursor), then key/value
    let mut stmt = conn
        .prepare("SELECT data FROM blobs WHERE data IS NOT NULL ORDER BY rowid")
        .or_else(|_| conn.prepare("SELECT value FROM blobs WHERE value IS NOT NULL ORDER BY rowid"))?;

    let rows = stmt.query_map([], |row| row.get::<_, Vec<u8>>(0))?;

    let mut messages = Vec::new();
    for raw in rows.flatten() {
        let msg: OpenAIMessage = match serde_json::from_slice(&raw) {
            Ok(m) => m,
            Err(_) => continue,
        };
        let role = match msg.role.as_deref() {
            Some("user") => Role::User,
            Some("assistant") => Role::Assistant,
            _ => continue,
        };
        let content = msg.content.as_ref().map(extract_openai_content).unwrap_or_default();
        if content.is_empty() {
            continue;
        }
        messages.push(Message {
            role,
            content,
            timestamp: fallback_ts,
        });
    }
    Ok(messages)
}

fn decode_blob_value(raw: &[u8]) -> Vec<CursorBubble> {
    // Try direct JSON parse
    if let Ok(val) = serde_json::from_slice::<serde_json::Value>(raw) {
        return extract_bubbles(&val);
    }

    // Try base64 decode then JSON parse
    if let Ok(decoded) = base64::Engine::decode(
        &base64::engine::general_purpose::STANDARD,
        raw,
    ) {
        if let Ok(val) = serde_json::from_slice::<serde_json::Value>(&decoded) {
            return extract_bubbles(&val);
        }
    }

    Vec::new()
}

fn extract_bubbles(val: &serde_json::Value) -> Vec<CursorBubble> {
    // Could be a single bubble or an array/object containing bubbles
    if let Ok(bubble) = serde_json::from_value::<CursorBubble>(val.clone()) {
        if bubble.bubble_type.is_some() {
            return vec![bubble];
        }
    }

    if let Some(arr) = val.as_array() {
        return arr
            .iter()
            .filter_map(|v| serde_json::from_value::<CursorBubble>(v.clone()).ok())
            .filter(|b| b.bubble_type.is_some())
            .collect();
    }

    // Try "bubbles" key
    if let Some(arr) = val.get("bubbles").and_then(|v| v.as_array()) {
        return arr
            .iter()
            .filter_map(|v| serde_json::from_value::<CursorBubble>(v.clone()).ok())
            .filter(|b| b.bubble_type.is_some())
            .collect();
    }

    Vec::new()
}

fn read_blobs(conn: &Connection) -> Result<Vec<CursorBubble>> {
    let mut stmt = conn
        .prepare("SELECT value FROM blobs WHERE value IS NOT NULL ORDER BY rowid")
        .or_else(|_| conn.prepare("SELECT data FROM blobs WHERE data IS NOT NULL ORDER BY rowid"))?;
    let rows = stmt.query_map([], |row| row.get::<_, Vec<u8>>(0))?;

    let mut bubbles = Vec::new();
    for raw in rows.flatten() {
        bubbles.extend(decode_blob_value(&raw));
    }
    Ok(bubbles)
}

/// Resolve cwd for a Cursor session by looking up its UUID in ~/.cursor/projects/
pub fn resolve_cursor_cwd(uuid: &str) -> Option<String> {
    let home = dirs::home_dir()?;
    let projects_dir = home.join(".cursor/projects");

    for entry in std::fs::read_dir(&projects_dir).ok()?.flatten() {
        let transcript = entry
            .path()
            .join("agent-transcripts")
            .join(format!("{}.jsonl", uuid));
        if transcript.exists() {
            return decode_workspace_path(entry.file_name().to_str()?);
        }
    }
    None
}

/// Decode a Cursor workspace directory name to a filesystem path.
/// Cursor encodes `/`, `.` as `-` in workspace names.
/// Uses greedy DFS with Path::exists() validation to resolve ambiguity.
fn decode_workspace_path(name: &str) -> Option<String> {
    let parts: Vec<&str> = name.split('-').collect();
    if parts.is_empty() {
        return None;
    }
    let start = PathBuf::from("/").join(parts[0]);
    let mut best: Option<PathBuf> = None;
    dfs_decode(&parts, 1, &start, &mut best);
    best.map(|p| p.to_string_lossy().into_owned())
}

fn dfs_decode(parts: &[&str], idx: usize, current: &Path, best: &mut Option<PathBuf>) {
    if current.exists() {
        *best = Some(current.to_path_buf());
    }
    if idx >= parts.len() {
        return;
    }

    // Try '/' separator (new directory component)
    let with_slash = current.join(parts[idx]);
    dfs_decode(parts, idx + 1, &with_slash, best);

    // Try '.' separator (dot-join with current last component)
    if let (Some(parent), Some(last)) = (current.parent(), current.file_name()) {
        let dotted = parent.join(format!("{}.{}", last.to_string_lossy(), parts[idx]));
        dfs_decode(parts, idx + 1, &dotted, best);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::params;

    #[test]
    fn test_can_parse_cursor_path() {
        // macOS
        assert!(CursorParser::can_parse(Path::new(
            "/home/user/.cursor/chats/abc-123/store.db"
        )));
        // Linux
        assert!(CursorParser::can_parse(Path::new(
            "/home/user/.config/cursor/chats/abc-123/store.db"
        )));
        // Windows backslash paths
        assert!(CursorParser::can_parse(Path::new(
            "C:\\Users\\user\\.cursor\\chats\\abc-123\\store.db"
        )));
        assert!(CursorParser::can_parse(Path::new(
            "C:\\Users\\user\\.config\\cursor\\chats\\abc-123\\store.db"
        )));
    }

    #[test]
    fn test_can_parse_rejects_other() {
        assert!(!CursorParser::can_parse(Path::new(
            "/home/user/.claude/projects/test/session.jsonl"
        )));
        assert!(!CursorParser::can_parse(Path::new(
            "/home/user/.cursor/chats/abc-123/other.db"
        )));
        assert!(!CursorParser::can_parse(Path::new(
            "/home/user/.codex/sessions/test.jsonl"
        )));
    }

    #[test]
    fn test_parse_cursor_session() {
        let dir = tempfile::TempDir::new().unwrap();
        let chat_dir = dir.path().join(".cursor/chats/test-chat-001");
        std::fs::create_dir_all(&chat_dir).unwrap();
        let db_path = chat_dir.join("store.db");

        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
             CREATE TABLE blobs (key TEXT PRIMARY KEY, value BLOB);",
        )
        .unwrap();

        let meta_json = serde_json::json!({
            "createdAt": 1700000000000_i64,
            "cwd": "/test/project"
        });
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('0', ?1)",
            params![meta_json.to_string()],
        )
        .unwrap();

        let bubble1 = serde_json::json!({
            "bubbleId": "b1",
            "type": 1,
            "text": "Hello cursor",
            "timestamp": 1700000001000_i64
        });
        let bubble2 = serde_json::json!({
            "bubbleId": "b2",
            "type": 2,
            "text": "Hi there!",
            "timestamp": 1700000002000_i64
        });

        conn.execute(
            "INSERT INTO blobs (key, value) VALUES ('b1', ?1)",
            params![bubble1.to_string().as_bytes()],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO blobs (key, value) VALUES ('b2', ?1)",
            params![bubble2.to_string().as_bytes()],
        )
        .unwrap();
        drop(conn);

        let session = CursorParser::parse_file(&db_path).unwrap();
        assert_eq!(session.id, "test-chat-001");
        assert_eq!(session.source, SessionSource::CursorCli);
        assert_eq!(session.cwd, "/test/project");
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[0].role, Role::User);
        assert_eq!(session.messages[0].content, "Hello cursor");
        assert_eq!(session.messages[1].role, Role::Assistant);
        assert_eq!(session.messages[1].content, "Hi there!");
    }

    #[test]
    fn test_parse_cursor_session_base64() {
        use base64::Engine;

        let dir = tempfile::TempDir::new().unwrap();
        let chat_dir = dir.path().join(".cursor/chats/test-b64");
        std::fs::create_dir_all(&chat_dir).unwrap();
        let db_path = chat_dir.join("store.db");

        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
             CREATE TABLE blobs (key TEXT PRIMARY KEY, value BLOB);",
        )
        .unwrap();

        let bubble = serde_json::json!({
            "bubbleId": "b1",
            "type": 1,
            "text": "base64 message",
            "timestamp": 1700000001000_i64
        });
        let encoded = base64::engine::general_purpose::STANDARD.encode(bubble.to_string());

        conn.execute(
            "INSERT INTO blobs (key, value) VALUES ('b1', ?1)",
            params![encoded.as_bytes()],
        )
        .unwrap();
        drop(conn);

        let session = CursorParser::parse_file(&db_path).unwrap();
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.messages[0].role, Role::User);
        assert_eq!(session.messages[0].content, "base64 message");
    }

    #[test]
    fn test_decode_workspace_path() {
        let dir = tempfile::TempDir::new().unwrap();
        // Create /a/b.c/d structure
        let nested = dir.path().join("a/b.c/d");
        std::fs::create_dir_all(&nested).unwrap();

        // Encoded as "a-b-c-d" (both / and . become -)
        let dir_str = dir.path().to_str().unwrap();
        // We need to test with real filesystem paths, so construct the encoded name
        // from the tempdir path components
        let parts: Vec<&str> = dir_str.trim_start_matches('/').split('/').collect();
        // Build encoded name: join all path components + "a-b-c-d" with '-'
        let mut encoded_parts = parts.clone();
        encoded_parts.extend_from_slice(&["a", "b", "c", "d"]);
        let encoded = encoded_parts.join("-");

        let result = decode_workspace_path(&encoded);
        assert_eq!(result, Some(nested.to_string_lossy().into_owned()));
    }

    #[test]
    fn test_decode_workspace_path_simple() {
        // Test with paths that actually exist on the system
        // /tmp always exists
        let dir = tempfile::TempDir::new().unwrap();
        let project = dir.path().join("myproject");
        std::fs::create_dir_all(&project).unwrap();

        let dir_str = dir.path().to_str().unwrap();
        let parts: Vec<&str> = dir_str.trim_start_matches('/').split('/').collect();
        let mut encoded_parts = parts;
        encoded_parts.push("myproject");
        let encoded = encoded_parts.join("-");

        let result = decode_workspace_path(&encoded);
        assert_eq!(result, Some(project.to_string_lossy().into_owned()));
    }

    #[test]
    fn test_resolve_cursor_cwd_not_found() {
        // UUID that doesn't exist should return None
        let result = resolve_cursor_cwd("nonexistent-uuid-12345");
        assert!(result.is_none());
    }

    #[test]
    fn test_parse_openai_format() {
        let dir = tempfile::TempDir::new().unwrap();
        let chat_dir = dir.path().join(".cursor/chats/test-openai");
        std::fs::create_dir_all(&chat_dir).unwrap();
        let db_path = chat_dir.join("store.db");

        let conn = Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
             CREATE TABLE blobs (id TEXT PRIMARY KEY, data BLOB);",
        )
        .unwrap();

        let meta_json = serde_json::json!({
            "agentId": "test-openai",
            "createdAt": 1700000000000_i64,
        });
        // Store meta as hex-encoded JSON (like real Cursor Agent CLI)
        let hex_meta = hex::encode(meta_json.to_string());
        conn.execute(
            "INSERT INTO meta (key, value) VALUES ('0', ?1)",
            params![hex_meta],
        )
        .unwrap();

        let msg1 = serde_json::json!({"role": "user", "content": [{"type": "text", "text": "hello openai format"}]});
        let msg2 = serde_json::json!({"role": "assistant", "content": [{"type": "text", "text": "response here"}]});
        let msg3 = serde_json::json!({"role": "system", "content": "system prompt"});

        conn.execute("INSERT INTO blobs (id, data) VALUES ('s1', ?1)", params![msg3.to_string().as_bytes()]).unwrap();
        conn.execute("INSERT INTO blobs (id, data) VALUES ('u1', ?1)", params![msg1.to_string().as_bytes()]).unwrap();
        conn.execute("INSERT INTO blobs (id, data) VALUES ('a1', ?1)", params![msg2.to_string().as_bytes()]).unwrap();
        drop(conn);

        let session = CursorParser::parse_file(&db_path).unwrap();
        assert_eq!(session.id, "test-openai");
        assert_eq!(session.messages.len(), 2); // system skipped
        assert_eq!(session.messages[0].role, Role::User);
        assert_eq!(session.messages[0].content, "hello openai format");
        assert_eq!(session.messages[1].role, Role::Assistant);
        assert_eq!(session.messages[1].content, "response here");
        // Timestamp should use createdAt from meta, not Utc::now()
        assert_eq!(session.timestamp, Utc.timestamp_millis_opt(1700000000000).unwrap());
        assert_eq!(session.messages[0].timestamp, session.timestamp);
    }
}
