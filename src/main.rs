mod db;
mod ollama;
mod search;

use axum::{routing::post, Extension, Router};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;

use db::init_db;
use ollama::{chat_handler, chat_stream_handler};

#[tokio::main]
async fn main() {
    let conn = init_db().expect("Chyba při inicializaci SQLite databáze");
    let db = Arc::new(Mutex::new(conn));

    let app = Router::new()
        .route("/api/chat", post(chat_handler))
        .route("/api/chat/stream", post(chat_stream_handler))
        .layer(CorsLayer::permissive())
        .layer(Extension(db));

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    println!("RPAI Backend poslouchá na http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
