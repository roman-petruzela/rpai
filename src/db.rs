use rusqlite::{params, Connection, Result as SqlResult};
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

use crate::ollama::OllamaMessage;

const DATABASE_PATH: &str = "rpai.db";
const MAX_PROJECT_TERMS: usize = 8;
const MAX_PROJECT_RESULTS: usize = 5;

pub fn init_db() -> SqlResult<Connection> {
    let conn = Connection::open(DATABASE_PATH)?;
    conn.busy_timeout(std::time::Duration::from_secs(5))?;

    conn.execute_batch(
        "
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA foreign_keys = ON;

        CREATE TABLE IF NOT EXISTS sector_profile (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sector_projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            content TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS sector_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_history_session_id
        ON sector_history(session_id, id);

        CREATE INDEX IF NOT EXISTS idx_projects_category
        ON sector_projects(category);
        ",
    )?;

    // Keep the original defaults. Existing values are not overwritten.
    conn.execute(
        "INSERT OR IGNORE INTO sector_profile (key, value)
         VALUES ('user_name', 'Admin'), ('preferred_os', 'Arch Linux')",
        [],
    )?;

    println!("[DATABASE] SQLite database initialized.");
    Ok(conn)
}

pub fn save_message(
    conn: &Arc<Mutex<Connection>>,
    session_id: &str,
    role: &str,
    content: &str,
) {
    if session_id.trim().is_empty() || content.trim().is_empty() {
        eprintln!("[DATABASE] Refusing to save an empty session or message.");
        return;
    }

    if !matches!(role, "system" | "user" | "assistant" | "tool") {
        eprintln!("[DATABASE] Refusing unsupported message role: {}", role);
        return;
    }

    let connection = match conn.lock() {
        Ok(connection) => connection,
        Err(_) => {
            eprintln!("[DATABASE] Database mutex is poisoned.");
            return;
        }
    };

    if let Err(error) = connection.execute(
        "INSERT INTO sector_history (session_id, role, content)
         VALUES (?1, ?2, ?3)",
        params![session_id, role, content],
    ) {
        eprintln!("[DATABASE] Failed to save message: {}", error);
    }
}

pub fn load_history(
    conn: &Arc<Mutex<Connection>>,
    session_id: &str,
    limit: usize,
) -> Vec<OllamaMessage> {
    if limit == 0 {
        return Vec::new();
    }

    let connection = match conn.lock() {
        Ok(connection) => connection,
        Err(_) => {
            eprintln!("[DATABASE] Database mutex is poisoned.");
            return Vec::new();
        }
    };

    let mut statement = match connection.prepare(
        "SELECT role, content
         FROM (
             SELECT id, role, content
             FROM sector_history
             WHERE session_id = ?1
             ORDER BY id DESC
             LIMIT ?2
         )
         ORDER BY id ASC",
    ) {
        Ok(statement) => statement,
        Err(error) => {
            eprintln!("[DATABASE] Failed to prepare history query: {}", error);
            return Vec::new();
        }
    };

    let rows = match statement.query_map(params![session_id, limit.min(100)], |row| {
        Ok(OllamaMessage {
            role: row.get(0)?,
            content: row.get(1)?,
        })
    }) {
        Ok(rows) => rows,
        Err(error) => {
            eprintln!("[DATABASE] Failed to query history: {}", error);
            return Vec::new();
        }
    };

    rows.filter_map(|row| match row {
        Ok(message) => Some(message),
        Err(error) => {
            eprintln!("[DATABASE] Failed to read history row: {}", error);
            None
        }
    })
    .collect()
}

pub fn load_profile(conn: &Arc<Mutex<Connection>>) -> String {
    let connection = match conn.lock() {
        Ok(connection) => connection,
        Err(_) => {
            eprintln!("[DATABASE] Database mutex is poisoned.");
            return String::new();
        }
    };

    let mut statement = match connection.prepare(
        "SELECT key, value FROM sector_profile ORDER BY key ASC",
    ) {
        Ok(statement) => statement,
        Err(error) => {
            eprintln!("[DATABASE] Failed to prepare profile query: {}", error);
            return String::new();
        }
    };

    let rows = match statement.query_map([], |row| {
        let key: String = row.get(0)?;
        let value: String = row.get(1)?;
        Ok(format!("{}: {}", key, value))
    }) {
        Ok(rows) => rows,
        Err(error) => {
            eprintln!("[DATABASE] Failed to query profile: {}", error);
            return String::new();
        }
    };

    rows.filter_map(Result::ok).collect::<Vec<_>>().join("\n")
}

pub fn set_profile_value(
    conn: &Arc<Mutex<Connection>>,
    key: &str,
    value: &str,
) -> Result<(), String> {
    if key.trim().is_empty() || value.trim().is_empty() {
        return Err("Profile key and value cannot be empty.".to_string());
    }

    let connection = conn
        .lock()
        .map_err(|_| "Database mutex is poisoned.".to_string())?;

    connection
        .execute(
            "INSERT INTO sector_profile (key, value) VALUES (?1, ?2)
             ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            params![key.trim(), value.trim()],
        )
        .map_err(|error| format!("Failed to update profile: {}", error))?;

    Ok(())
}

pub fn delete_profile_value(
    conn: &Arc<Mutex<Connection>>,
    key: &str,
) -> Result<(), String> {
    let connection = conn
        .lock()
        .map_err(|_| "Database mutex is poisoned.".to_string())?;

    connection
        .execute("DELETE FROM sector_profile WHERE key = ?1", params![key])
        .map_err(|error| format!("Failed to delete profile value: {}", error))?;

    Ok(())
}

pub fn save_project(
    conn: &Arc<Mutex<Connection>>,
    category: &str,
    content: &str,
) -> Result<(), String> {
    if category.trim().is_empty() || content.trim().is_empty() {
        return Err("Project category and content cannot be empty.".to_string());
    }

    let connection = conn
        .lock()
        .map_err(|_| "Database mutex is poisoned.".to_string())?;

    connection
        .execute(
            "INSERT INTO sector_projects (category, content) VALUES (?1, ?2)",
            params![category.trim(), content.trim()],
        )
        .map_err(|error| format!("Failed to save project knowledge: {}", error))?;

    Ok(())
}

fn escape_like(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

fn extract_terms(query: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut seen = HashSet::new();

    for word in query.split(|character: char| !character.is_alphanumeric()) {
        if word.chars().count() < 3 {
            continue;
        }

        let normalized = word.to_lowercase();
        if seen.insert(normalized) {
            terms.push(word.to_string());
        }

        if terms.len() >= MAX_PROJECT_TERMS {
            break;
        }
    }

    terms
}

pub fn search_project_sector(
    conn: &Arc<Mutex<Connection>>,
    query: &str,
) -> String {
    let terms = extract_terms(query);
    if terms.is_empty() {
        return String::new();
    }

    let connection = match conn.lock() {
        Ok(connection) => connection,
        Err(_) => {
            eprintln!("[DATABASE] Database mutex is poisoned.");
            return String::new();
        }
    };

    let mut results = Vec::new();
    let mut seen = HashSet::new();

    for term in terms {
        if results.len() >= MAX_PROJECT_RESULTS {
            break;
        }

        let pattern = format!("%{}%", escape_like(&term));
        let mut statement = match connection.prepare(
            "SELECT category, content
             FROM sector_projects
             WHERE content LIKE ?1 ESCAPE '\\'
                OR category LIKE ?1 ESCAPE '\\'
             ORDER BY id DESC
             LIMIT 3",
        ) {
            Ok(statement) => statement,
            Err(error) => {
                eprintln!("[DATABASE] Failed to prepare project query: {}", error);
                return results.join("\n");
            }
        };

        let rows = match statement.query_map(params![pattern], |row| {
            let category: String = row.get(0)?;
            let content: String = row.get(1)?;
            Ok((category, content))
        }) {
            Ok(rows) => rows,
            Err(error) => {
                eprintln!("[DATABASE] Failed to search projects: {}", error);
                continue;
            }
        };

        for row in rows {
            match row {
                Ok((category, content)) => {
                    if seen.insert((category.clone(), content.clone())) {
                        results.push(format!("[{}]: {}", category, content));
                    }
                }
                Err(error) => eprintln!("[DATABASE] Failed to read project row: {}", error),
            }

            if results.len() >= MAX_PROJECT_RESULTS {
                break;
            }
        }
    }

    results.join("\n")
}
