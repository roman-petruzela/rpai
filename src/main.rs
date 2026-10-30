mod db;
mod memory;
mod ollama;
mod search;

use axum::{
    response::Html,
    routing::{get, post},
    Extension,
    Router,
};
use rusqlite::Connection;
use std::{
    env,
    error::Error,
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use db::init_db;
use ollama::{chat_handler, chat_stream_handler};

const INDEX_HTML: &str = include_str!("../static/index.html");
const DEFAULT_ADDRESS: &str = "127.0.0.1:3000";

async fn index_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    println!("==================================================");
    println!("                    J.A.R.V.I.S.");
    println!("          PERSONAL AI ASSISTANT SYSTEM");
    println!("==================================================");

    let address = env::var("JARVIS_ADDR")
        .unwrap_or_else(|_| DEFAULT_ADDRESS.to_string());
    let address: SocketAddr = address.parse()?;

    println!("[SYSTEM] Initializing SQLite database...");
    let connection: Connection = init_db()?;

    println!("[SYSTEM] Initializing long-term memory...");
    memory::init_memory(&connection)?;

    let database = Arc::new(Mutex::new(connection));

    let app = Router::new()
        .route("/", get(index_handler))
        .route("/api/chat", post(chat_handler))
        .route("/api/chat/stream", post(chat_stream_handler))
        .layer(Extension(database));

    let listener = tokio::net::TcpListener::bind(address).await?;

    println!("[SYSTEM] HTTP server listening on http://{}", address);
    println!("[SYSTEM] Web interface: http://{}", address);
    println!("[SYSTEM] Chat endpoint: http://{}/api/chat", address);
    println!("[SYSTEM] Stream endpoint: http://{}/api/chat/stream", address);
    println!("[SYSTEM] JARVIS is online.");

    axum::serve(listener, app).await?;

    Ok(())
}
