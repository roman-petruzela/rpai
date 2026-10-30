use rusqlite::{Connection, Result as SqlResult};
use std::sync::{Arc, Mutex};
use crate::ollama::OllamaMessage;

pub fn init_db() -> SqlResult<Connection> {
    let conn = Connection::open("rpai.db")?;
    conn.execute(
        "CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )",
        [],
    )?;
    Ok(conn)
}

pub fn save_message(conn: &Arc<Mutex<Connection>>, role: &str, content: &str) {
    if let Ok(c) = conn.lock() {
        let _ = c.execute(
            "INSERT INTO history (role, content) VALUES (?1, ?2)",
            [role, content],
        );
    }
}

pub fn load_recent_history(conn: &Arc<Mutex<Connection>>, limit: usize) -> Vec<OllamaMessage> {
    let mut history = Vec::new();
    if let Ok(c) = conn.lock() {
        if let Ok(mut stmt) = c.prepare("SELECT role, content FROM history ORDER BY id DESC LIMIT ?1") {
            if let Ok(rows) = stmt.query_map([limit], |row| {
                Ok(OllamaMessage {
                    role: row.get(0)?,
                    content: row.get(1)?,
                })
            }) {
                for r in rows.flatten() {
                    history.push(r);
                }
            }
        }
    }
    history.reverse();
    history
}
