
mod db;
mod ollama;
mod search;

use axum::{
    response::Html,
    routing::{get, post},
    Extension,
    Router,
};

use std::{
    env,
    error::Error,
    net::SocketAddr,
    sync::{Arc, Mutex},
};

use rusqlite::Connection;

use db::init_db;
use ollama::{chat_handler, chat_stream_handler};

// ============================================================
// R.O.M.A.N. CONFIGURATION
// ============================================================

const DEFAULT_ADDRESS: &str = "127.0.0.1:3000";

const INDEX_HTML: &str = include_str!("../static/index.html");

// ============================================================
// APPLICATION
// ============================================================

async fn index_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}

// ============================================================
// SERVER INITIALIZATION
// ============================================================

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {

    println!("========================================");
    println!("             R.O.M.A.N.");
    println!("    Real-time Operational Mind");
    println!("      for Assistance and Navigation");
    println!("========================================");

    // --------------------------------------------------------
    // CONFIGURATION
    // --------------------------------------------------------

    let address = env::var("ROMAN_ADDR")
        .unwrap_or_else(|_| DEFAULT_ADDRESS.to_string());

    let addr: SocketAddr = address.parse()?;

    println!("[SYSTEM] Loading configuration...");
    println!("[SYSTEM] Server address: {}", addr);

    // --------------------------------------------------------
    // DATABASE INITIALIZATION
    // --------------------------------------------------------

    println!("[DATABASE] Initializing SQLite database...");

    let connection: Connection = init_db()?;

    let db = Arc::new(Mutex::new(connection));

    println!("[DATABASE] SQLite connection established.");

    // --------------------------------------------------------
    // ROUTER INITIALIZATION
    // --------------------------------------------------------

    println!("[ROUTER] Registering application routes...");

    let app = Router::new()

        // Web interface
        .route("/", get(index_handler))

        // Standard chat endpoint
        .route("/api/chat", post(chat_handler))

        // Streaming chat endpoint
        .route("/api/chat/stream", post(chat_stream_handler))

        // Shared application database
        .layer(Extension(db));

    // --------------------------------------------------------
    // SERVER STARTUP
    // --------------------------------------------------------

    let listener = tokio::net::TcpListener::bind(addr).await?;

    println!("[ROUTER] Application routes initialized.");

    println!("[SYSTEM] R.O.M.A.N. is ready.");
    println!("[SYSTEM] Web interface: http://{}", addr);
    println!("[SYSTEM] Chat API: http://{}/api/chat", addr);
    println!("[SYSTEM] Streaming API: http://{}/api/chat/stream", addr);

    println!("========================================");
    println!("             SYSTEM ONLINE");
    println!("========================================");

    // --------------------------------------------------------
    // START SERVER
    // --------------------------------------------------------

    axum::serve(listener, app).await?;

    println!("[SYSTEM] R.O.M.A.N. has been stopped.");

    Ok(())
}
