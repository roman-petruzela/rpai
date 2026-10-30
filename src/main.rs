use axum::{
    extract::Extension,
    response::sse::{Event, KeepAlive, Sse},
    routing::post,
    Json, Router,
};
use futures_util::stream::Stream;
use futures_util::StreamExt;
use rusqlite::{Connection, Result as SqlResult};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;

#[derive(Deserialize)]
struct ChatRequest {
    message: String,
    #[serde(default)]
    use_search: bool,
}

#[derive(Serialize, Deserialize, Clone)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Serialize)]
struct OllamaChatRequest {
    model: String,
    messages: Vec<OllamaMessage>,
    stream: bool,
}

#[derive(Deserialize)]
struct OllamaStreamChunk {
    message: Option<OllamaMessage>,
}

fn init_db() -> SqlResult<Connection> {
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

fn save_message(conn: &Arc<Mutex<Connection>>, role: &str, content: &str) {
    if let Ok(c) = conn.lock() {
        let _ = c.execute(
            "INSERT INTO history (role, content) VALUES (?1, ?2)",
            [role, content],
        );
    }
}

fn load_recent_history(conn: &Arc<Mutex<Connection>>, limit: usize) -> Vec<OllamaMessage> {
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

async fn search_duckduckgo(query: &str) -> String {
    let client = reqwest::Client::new();
    let url = format!("https://html.duckduckgo.com/html/?q={}", urlencoding::encode(query));
    if let Ok(res) = client.get(&url).header("User-Agent", "Mozilla/5.0").send().await {
        if let Ok(text) = res.text().await {
            let mut results = Vec::new();
            for cap in text.match_indices("class=\"result__snippet\"") {
                let start = cap.0;
                if let Some(end) = text[start..].find("</a>") {
                    let snippet = &text[start..start + end];
                    let clean = snippet
                        .replace("class=\"result__snippet\">", "")
                        .replace("<b>", "")
                        .replace("</b>", "");
                    results.push(clean);
                    if results.len() >= 3 {
                        break;
                    }
                }
            }
            if !results.is_empty() {
                return format!("Výsledky z webu:\n{}", results.join("\n"));
            }
        }
    }
    "Nenalezeny žádné výsledky.".to_string()
}

async fn chat_stream_handler(
    Extension(db): Extension<Arc<Mutex<Connection>>>,
    Json(payload): Json<ChatRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    save_message(&db, "user", &payload.message);

    let history = load_recent_history(&db, 10);

    let mut sys_prompt = "Jsi RPAI, pokročilý osobní AI asistent na lokálním serveru. Odpovídáš věcně, stručně a v češtině.".to_string();

    if payload.use_search {
        let search_results = search_duckduckgo(&payload.message).await;
        sys_prompt.push_str(&format!("\n\n{}", search_results));
    }

    let mut messages = vec![OllamaMessage {
        role: "system".to_string(),
        content: sys_prompt,
    }];
    messages.extend(history);

    let ollama_req = OllamaChatRequest {
        model: "llama3.1:8b".to_string(),
        messages,
        stream: true,
    };

    let client = reqwest::Client::new();
    let req_builder = client.post("http://localhost:11434/api/chat").json(&ollama_req);

    let db_clone = db.clone();

    let stream = async_stream::stream! {
        if let Ok(res) = req_builder.send().await {
            let mut byte_stream = res.bytes_stream();
            let mut full_reply = String::new();

            while let Some(chunk) = byte_stream.next().await {
                if let Ok(bytes) = chunk {
                    if let Ok(chunk_str) = std::str::from_utf8(&bytes) {
                        for line in chunk_str.lines() {
                            if let Ok(parsed) = serde_json::from_str::<OllamaStreamChunk>(line) {
                                if let Some(msg) = parsed.message {
                                    full_reply.push_str(&msg.content);
                                    yield Ok(Event::default().data(msg.content));
                                }
                            }
                        }
                    }
                }
            }
            save_message(&db_clone, "assistant", &full_reply);
        } else {
            yield Ok(Event::default().data("Chyba při komunikaci s Ollamou."));
        }
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

#[tokio::main]
async fn main() {
    let conn = init_db().expect("Chyba při inicializaci SQLite databáze");
    let db = Arc::new(Mutex::new(conn));

    let app = Router::new()
        .route("/api/chat/stream", post(chat_stream_handler))
        .layer(CorsLayer::permissive())
        .layer(Extension(db));

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    println!("RPAI Backend poslouchá na http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
