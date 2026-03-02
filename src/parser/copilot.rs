use crate::session::{Message, Role, Session, SessionSource};
use anyhow::{Context, Result};
use chrono::{DateTime, Utc};
use serde::Deserialize;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

use super::{join_consecutive_messages, SessionParser};

#[derive(Debug, Deserialize)]
struct CopilotLine {
    #[serde(rename = "type")]
    entry_type: String,
    data: serde_json::Value,
    timestamp: Option<String>,
}

pub struct CopilotParser;

impl SessionParser for CopilotParser {
    fn can_parse(path: &Path) -> bool {
        let s = path.to_str().unwrap_or("");
        (s.contains(".copilot/session-state") || s.contains(".copilot\\session-state"))
            && path
                .file_name()
                .and_then(|n| n.to_str())
                .map(|n| n == "events.jsonl")
                .unwrap_or(false)
    }

    fn parse_file(path: &Path) -> Result<Session> {
        let file = File::open(path).context("Failed to open file")?;
        let reader = BufReader::with_capacity(64 * 1024, file);

        let mut session_id: Option<String> = None;
        let mut cwd: Option<String> = None;
        let mut git_branch: Option<String> = None;
        let mut latest_timestamp: Option<DateTime<Utc>> = None;
        let mut messages: Vec<Message> = Vec::new();

        for line in reader.lines() {
            let line = line.context("Failed to read line")?;
            if line.trim().is_empty() {
                continue;
            }

            let entry: CopilotLine = match serde_json::from_str(&line) {
                Ok(e) => e,
                Err(_) => continue,
            };

            // Parse timestamp
            let timestamp = entry
                .timestamp
                .as_ref()
                .and_then(|ts| DateTime::parse_from_rfc3339(ts).ok())
                .map(|dt| dt.with_timezone(&Utc));

            if let Some(ts) = timestamp {
                if latest_timestamp.is_none() || ts > latest_timestamp.unwrap() {
                    latest_timestamp = Some(ts);
                }
            }

            match entry.entry_type.as_str() {
                "session.start" => {
                    if session_id.is_none() {
                        session_id = entry.data["sessionId"].as_str().map(|s| s.to_string());
                    }
                    if cwd.is_none() {
                        cwd = entry.data["context"]["cwd"]
                            .as_str()
                            .map(|s| s.to_string());
                    }
                    if git_branch.is_none() {
                        git_branch = entry.data["context"]["branch"]
                            .as_str()
                            .map(|s| s.to_string());
                    }
                }
                "user.message" => {
                    if let Some(content) = entry.data["content"].as_str() {
                        if !content.is_empty() {
                            messages.push(Message {
                                role: Role::User,
                                content: content.to_string(),
                                timestamp: timestamp.unwrap_or_else(Utc::now),
                            });
                        }
                    }
                }
                "assistant.message" => {
                    if let Some(content) = entry.data["content"].as_str() {
                        if !content.is_empty() {
                            messages.push(Message {
                                role: Role::Assistant,
                                content: content.to_string(),
                                timestamp: timestamp.unwrap_or_else(Utc::now),
                            });
                        }
                    }
                }
                _ => {}
            }
        }

        // Fall back to directory name for session ID
        let session_id = session_id.unwrap_or_else(|| {
            path.parent()
                .and_then(|p| p.file_name())
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string()
        });

        Ok(Session {
            id: session_id,
            source: SessionSource::CopilotCli,
            file_path: path.to_path_buf(),
            cwd: cwd.unwrap_or_else(|| ".".to_string()),
            git_branch,
            timestamp: latest_timestamp.unwrap_or_else(Utc::now),
            messages: join_consecutive_messages(messages),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_can_parse_copilot_path() {
        assert!(CopilotParser::can_parse(Path::new(
            "/home/user/.copilot/session-state/abc-123/events.jsonl"
        )));
    }

    #[test]
    fn test_can_parse_rejects_other_paths() {
        assert!(!CopilotParser::can_parse(Path::new(
            "/home/user/.claude/projects/test/session.jsonl"
        )));
        assert!(!CopilotParser::can_parse(Path::new(
            "/home/user/.copilot/session-state/abc-123/workspace.yaml"
        )));
    }

    #[test]
    fn test_parse_events_basic() {
        let dir = tempfile::TempDir::new().unwrap();
        let session_dir = dir.path().join(".copilot/session-state/test-id-001");
        std::fs::create_dir_all(&session_dir).unwrap();
        let events_path = session_dir.join("events.jsonl");

        let content = r#"{"type":"session.start","data":{"sessionId":"test-id-001","context":{"cwd":"/test/project","branch":"main"}},"timestamp":"2026-01-15T10:00:00.000Z"}
{"type":"user.message","data":{"content":"hello"},"timestamp":"2026-01-15T10:00:05.000Z"}
{"type":"assistant.turn_start","data":{"turnId":"0"},"timestamp":"2026-01-15T10:00:05.100Z"}
{"type":"assistant.message","data":{"content":"Hi there!"},"timestamp":"2026-01-15T10:00:10.000Z"}
{"type":"assistant.turn_end","data":{},"timestamp":"2026-01-15T10:00:10.100Z"}"#;

        std::fs::write(&events_path, content).unwrap();

        let session = CopilotParser::parse_file(&events_path).unwrap();
        assert_eq!(session.id, "test-id-001");
        assert_eq!(session.source, SessionSource::CopilotCli);
        assert_eq!(session.cwd, "/test/project");
        assert_eq!(session.git_branch, Some("main".to_string()));
        assert_eq!(session.messages.len(), 2);
        assert_eq!(session.messages[0].role, Role::User);
        assert_eq!(session.messages[0].content, "hello");
        assert_eq!(session.messages[1].role, Role::Assistant);
        assert_eq!(session.messages[1].content, "Hi there!");
    }

    #[test]
    fn test_parse_events_skips_non_messages() {
        let dir = tempfile::TempDir::new().unwrap();
        let session_dir = dir.path().join(".copilot/session-state/test-skip");
        std::fs::create_dir_all(&session_dir).unwrap();
        let events_path = session_dir.join("events.jsonl");

        let content = r#"{"type":"session.start","data":{"sessionId":"test-skip","context":{"cwd":"/"}},"timestamp":"2026-01-15T10:00:00.000Z"}
{"type":"session.model_change","data":{"newModel":"gpt-4.1"},"timestamp":"2026-01-15T10:00:01.000Z"}
{"type":"session.info","data":{"infoType":"model","message":"Model changed"},"timestamp":"2026-01-15T10:00:01.000Z"}
{"type":"session.error","data":{"message":"something failed"},"timestamp":"2026-01-15T10:00:02.000Z"}
{"type":"user.message","data":{"content":"test"},"timestamp":"2026-01-15T10:00:05.000Z"}"#;

        std::fs::write(&events_path, content).unwrap();

        let session = CopilotParser::parse_file(&events_path).unwrap();
        assert_eq!(session.messages.len(), 1);
        assert_eq!(session.messages[0].content, "test");
    }
}
