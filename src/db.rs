use rusqlite::{
    params,
    Connection,
    Result as SqlResult,
};
use std::{
    collections::HashSet,
    sync::{Arc, Mutex, MutexGuard},
    time::Duration,
};

use crate::ollama::OllamaMessage;

// ============================================================
// CONFIGURATION
// ============================================================

const DATABASE_PATH: &str = "rpai.db";

const MAX_HISTORY_LIMIT: usize = 100;
const MAX_PROJECT_RESULTS: usize = 5;
const MAX_SEARCH_TERMS: usize = 8;
const MAX_RESULTS_PER_TERM: i64 = 3;

// ============================================================
// DATABASE CONNECTION
// ============================================================

pub fn init_db() -> SqlResult<Connection> {
    println!("[DATABASE] Opening SQLite database...");

    let conn = Connection::open(DATABASE_PATH)?;

    // Configure SQLite.
    conn.busy_timeout(Duration::from_secs(5))?;

    conn.execute_batch(
        "
        PRAGMA journal_mode = WAL;
        PRAGMA synchronous = NORMAL;
        PRAGMA foreign_keys = ON;
        "
    )?;

    // --------------------------------------------------------
    // PROFILE SECTOR
    // --------------------------------------------------------

    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS sector_profile (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        "
    )?;

    // --------------------------------------------------------
    // PROJECTS SECTOR
    // --------------------------------------------------------

    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS sector_projects (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            category TEXT NOT NULL,
            content TEXT NOT NULL
        );
        "
    )?;

    // --------------------------------------------------------
    // CONVERSATION HISTORY
    // --------------------------------------------------------

    conn.execute_batch(
        "
        CREATE TABLE IF NOT EXISTS sector_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        "
    )?;

    // --------------------------------------------------------
    // INDEXES
    // --------------------------------------------------------

    conn.execute_batch(
        "
        CREATE INDEX IF NOT EXISTS idx_history_session_id
        ON sector_history(session_id, id);

        CREATE INDEX IF NOT EXISTS idx_projects_category
        ON sector_projects(category);
        "
    )?;

    // --------------------------------------------------------
    // DEFAULT PROFILE
    // --------------------------------------------------------

    // These are original default values.
    // Existing user-defined values will never be overwritten.

    conn.execute(
        "
        INSERT OR IGNORE INTO sector_profile (key, value)
        VALUES
            ('user_name', 'Admin'),
            ('preferred_os', 'Arch Linux')
        ",
        [],
    )?;

    println!("[DATABASE] Database initialized successfully.");

    Ok(conn)
}

// ============================================================
// CONNECTION MANAGEMENT
// ============================================================

fn lock_connection<'a>(
    conn: &'a Arc<Mutex<Connection>>,
) -> Result<MutexGuard<'a, Connection>, String> {

    conn.lock().map_err(|_| {
        "Database connection mutex is poisoned.".to_string()
    })
}

// ============================================================
// ERROR HANDLING
// ============================================================

fn database_error(error: rusqlite::Error) -> String {
    format!("SQLite database error: {}", error)
}

// ============================================================
// PROFILE SECTOR
// ============================================================

pub fn try_load_profile(
    conn: &Arc<Mutex<Connection>>,
) -> Result<String, String> {

    let connection = lock_connection(conn)?;

    let mut stmt = connection
        .prepare(
            "
            SELECT key, value
            FROM sector_profile
            ORDER BY key ASC
            "
        )
        .map_err(database_error)?;

    let rows = stmt
        .query_map([], |row| {
            let key: String = row.get(0)?;
            let value: String = row.get(1)?;

            Ok(format!("{}: {}", key, value))
        })
        .map_err(database_error)?;

    let mut profile = Vec::new();

    for row in rows {
        profile.push(row.map_err(database_error)?);
    }

    Ok(profile.join("\n"))
}

// Compatibility wrapper for existing ollama.rs.
pub fn load_profile(
    conn: &Arc<Mutex<Connection>>,
) -> String {

    match try_load_profile(conn) {
        Ok(profile) => profile,

        Err(error) => {
            eprintln!("[DATABASE] {}", error);
            String::new()
        }
    }
}

// ------------------------------------------------------------
// UPDATE OR INSERT PROFILE VALUE
// ------------------------------------------------------------

pub fn set_profile_value(
    conn: &Arc<Mutex<Connection>>,
    key: &str,
    value: &str,
) -> Result<(), String> {

    if key.trim().is_empty() || value.trim().is_empty() {
        return Err(
            "Profile key and value cannot be empty.".to_string()
        );
    }

    let connection = lock_connection(conn)?;

    connection.execute(
        "
        INSERT INTO sector_profile (key, value)
        VALUES (?1, ?2)
        ON CONFLICT(key) DO UPDATE SET
            value = excluded.value
        ",
        params![key.trim(), value.trim()],
    )
    .map_err(database_error)?;

    println!("[DATABASE] Profile value updated: {}", key);

    Ok(())
}

// ------------------------------------------------------------
// DELETE PROFILE VALUE
// ------------------------------------------------------------

pub fn delete_profile_value(
    conn: &Arc<Mutex<Connection>>,
    key: &str,
) -> Result<(), String> {

    let connection = lock_connection(conn)?;

    connection.execute(
        "DELETE FROM sector_profile WHERE key = ?1",
        params![key],
    )
    .map_err(database_error)?;

    println!("[DATABASE] Profile value deleted: {}", key);

    Ok(())
}

// ============================================================
// PROJECTS SECTOR
// ============================================================

pub fn save_project(
    conn: &Arc<Mutex<Connection>>,
    category: &str,
    content: &str,
) -> Result<(), String> {

    if category.trim().is_empty() || content.trim().is_empty() {
        return Err(
            "Project category and content cannot be empty.".to_string()
        );
    }

    let connection = lock_connection(conn)?;

    connection.execute(
        "
        INSERT INTO sector_projects (category, content)
        VALUES (?1, ?2)
        ",
        params![category.trim(), content.trim()],
    )
    .map_err(database_error)?;

    println!(
        "[DATABASE] Project knowledge saved: {}",
        category
    );

    Ok(())
}

// ------------------------------------------------------------
// PROJECT SEARCH HELPERS
// ------------------------------------------------------------

fn extract_search_terms(query: &str) -> Vec<String> {
    let mut terms = Vec::new();
    let mut seen = HashSet::new();

    for word in query.split(|c: char| !c.is_alphanumeric()) {

        let word = word.trim();

        if word.chars().count() < 3 {
            continue;
        }

        let normalized = word.to_lowercase();

        if seen.insert(normalized) {
            terms.push(word.to_string());
        }

        if terms.len() >= MAX_SEARCH_TERMS {
            break;
        }
    }

    terms
}

// Escape SQLite LIKE wildcard characters.
fn escape_like(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('%', "\\%")
        .replace('_', "\\_")
}

// ------------------------------------------------------------
// SEARCH PROJECT KNOWLEDGE
// ------------------------------------------------------------

pub fn try_search_project_sector(
    conn: &Arc<Mutex<Connection>>,
    query: &str,
) -> Result<String, String> {

    let terms = extract_search_terms(query);

    if terms.is_empty() {
        return Ok(String::new());
    }

    let connection = lock_connection(conn)?;

    let mut results = Vec::new();

    let mut seen_results: HashSet<(String, String)> =
        HashSet::new();

    for term in terms {

        if results.len() >= MAX_PROJECT_RESULTS {
            break;
        }

        let pattern = format!(
            "%{}%",
            escape_like(&term)
        );

        let mut stmt = connection
            .prepare(
                r#"
                SELECT category, content
                FROM sector_projects
                WHERE
                    content LIKE ?1 ESCAPE '\'
                    OR category LIKE ?1 ESCAPE '\'
                ORDER BY id DESC
                LIMIT ?2
                "#
            )
            .map_err(database_error)?;

        let rows = stmt
            .query_map(
                params![pattern, MAX_RESULTS_PER_TERM],
                |row| {
                    let category: String = row.get(0)?;
                    let content: String = row.get(1)?;

                    Ok((category, content))
                },
            )
            .map_err(database_error)?;

        for row in rows {

            let (category, content) =
                row.map_err(database_error)?;

            let identity = (
                category.clone(),
                content.clone(),
            );

            if !seen_results.insert(identity) {
                continue;
            }

            results.push(
                format!(
                    "[{}]: {}",
                    category,
                    content
                )
            );

            if results.len() >= MAX_PROJECT_RESULTS {
                break;
            }
        }
    }

    Ok(results.join("\n"))
}

// Compatibility wrapper for existing ollama.rs.
pub fn search_project_sector(
    conn: &Arc<Mutex<Connection>>,
    query: &str,
) -> String {

    match try_search_project_sector(conn, query) {
        Ok(results) => results,

        Err(error) => {
            eprintln!("[DATABASE] {}", error);
            String::new()
        }
    }
}

// ============================================================
// CONVERSATION HISTORY
// ============================================================

pub fn try_save_message(
    conn: &Arc<Mutex<Connection>>,
    session_id: &str,
    role: &str,
    content: &str,
) -> Result<(), String> {

    if session_id.trim().is_empty() {
        return Err("Session ID cannot be empty.".to_string());
    }

    if !matches!(role, "system" | "user" | "assistant" | "tool") {
        return Err(
            format!("Unsupported message role: {}", role)
        );
    }

    if content.trim().is_empty() {
        return Err("Message content cannot be empty.".to_string());
    }

    let connection = lock_connection(conn)?;

    connection.execute(
        "
        INSERT INTO sector_history (
            session_id,
            role,
            content
        )
        VALUES (?1, ?2, ?3)
        ",
        params![session_id, role, content],
    )
    .map_err(database_error)?;

    Ok(())
}

// Compatibility wrapper for existing ollama.rs.
pub fn save_message(
    conn: &Arc<Mutex<Connection>>,
    session_id: &str,
    role: &str,
    content: &str,
) {

    if let Err(error) = try_save_message(
        conn,
        session_id,
        role,
        content,
    ) {
        eprintln!("[DATABASE] Failed to save message: {}", error);
    }
}

// ------------------------------------------------------------
// LOAD CONVERSATION HISTORY
// ------------------------------------------------------------

pub fn try_load_history(
    conn: &Arc<Mutex<Connection>>,
    session_id: &str,
    limit: usize,
) -> Result<Vec<OllamaMessage>, String> {

    if limit == 0 {
        return Ok(Vec::new());
    }

    let limit = limit.min(MAX_HISTORY_LIMIT) as i64;

    let connection = lock_connection(conn)?;

    let mut stmt = connection
        .prepare(
            "
            SELECT role, content
            FROM (
                SELECT id, role, content
                FROM sector_history
                WHERE session_id = ?1
                ORDER BY id DESC
                LIMIT ?2
            )
            ORDER BY id ASC
            "
        )
        .map_err(database_error)?;

    let rows = stmt
        .query_map(
            params![session_id, limit],
            |row| {
                Ok(OllamaMessage {
                    role: row.get(0)?,
                    content: row.get(1)?,
                })
            },
        )
        .map_err(database_error)?;

    let mut history = Vec::new();

    for row in rows {
        history.push(row.map_err(database_error)?);
    }

    Ok(history)
}

// Compatibility wrapper for existing ollama.rs.
pub fn load_history(
    conn: &Arc<Mutex<Connection>>,
    session_id: &str,
    limit: usize,
) -> Vec<OllamaMessage> {

    match try_load_history(conn, session_id, limit) {
        Ok(history) => history,

        Err(error) => {
            eprintln!("[DATABASE] Failed to load history: {}", error);
            Vec::new()
        }
    }
}
