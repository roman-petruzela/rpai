use axum::{
    extract::Extension,
    response::sse::{Event, KeepAlive, Sse},
    Json,
};
use futures_util::stream::Stream;
use futures_util::StreamExt;
use rusqlite::Connection;
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};

use crate::db::{load_recent_history, save_message};
use crate::search::search_duckduckgo;

#[derive(Deserialize)]
pub struct ChatRequest {
    pub message: String,
    #[serde(default)]
    pub use_search: bool,
}

#[derive(Serialize)]
pub struct ChatResponse {
    pub reply: String,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct OllamaMessage {
    pub role: String,
    pub content: String,
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

#[derive(Deserialize)]
struct OllamaChatResponse {
    message: OllamaMessage,
}

// 1. Ne-streamovaný endpoint (vrací čistý JSON)
pub async fn chat_handler(
    Extension(db): Extension<Arc<Mutex<Connection>>>,
    Json(payload): Json<ChatRequest>,
) -> Result<Json<ChatResponse>, (axum::http::StatusCode, String)> {
    save_message(&db, "user", &payload.message);

    let history = load_recent_history(&db, 10);
    let mut sys_prompt = "Jsi RPAI, pokročilý osobní AI asistent. Odpovídáš věcně, stručně a v češtině.".to_string();

    if payload.use_search {
        let search_results = search_duckduckgo(&payload.message).await;
        sys_prompt.push_str(&format!(
            "\n\n[AKTUÁLNÍ DATA Z INTERNETU]:\n{}\n\nINSTRUKCE: Vycházej výhradně z výše uvedených aktuálních dat.",
            search_results
        ));
    }

    let mut messages = vec![OllamaMessage {
        role: "system".to_string(),
        content: sys_prompt,
    }];
    messages.extend(history);

    let ollama_req = OllamaChatRequest {
        model: "llama3.1:8b".to_string(),
        messages,
        stream: false,
    };

    let client = reqwest::Client::new();
    let res = client
        .post("http://localhost:11434/api/chat")
        .json(&ollama_req)
        .send()
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("Ollama error: {}", e)))?;

    let ollama_res: OllamaChatResponse = res
        .json()
        .await
        .map_err(|e| (axum::http::StatusCode::INTERNAL_SERVER_ERROR, format!("JSON error: {}", e)))?;

    save_message(&db, "assistant", &ollama_res.message.content);

    Ok(Json(ChatResponse {
        reply: ollama_res.message.content,
    }))
}

// 2. Streamovaný SSE endpoint (pro webový frontend)
pub async fn chat_stream_handler(
    Extension(db): Extension<Arc<Mutex<Connection>>>,
    Json(payload): Json<ChatRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    save_message(&db, "user", &payload.message);

    let history = load_recent_history(&db, 10);
    let mut sys_prompt = "Jsi RPAI, pokročilý osobní AI asistent. Odpovídáš věcně, stručně a v češtině.".to_string();

    if payload.use_search {
        let search_results = search_duckduckgo(&payload.message).await;
        sys_prompt.push_str(&format!(
            "\n\n[AKTUÁLNÍ DATA Z INTERNETU]:\n{}\n\nINSTRUKCE: Vycházej výhradně z výše uvedených aktuálních dat.",
            search_results
        ));
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
