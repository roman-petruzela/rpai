mod db;
mod ollama;
mod search;

use axum::{
    response::Html,
    routing::{get, post},
    Extension, Router,
};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use tower_http::cors::CorsLayer;

use db::init_db;
use ollama::{chat_handler, chat_stream_handler};

const INDEX_HTML: &str = include_str!("../static/index.html");

async fn index_handler() -> Html<&'static str> {
    Html(INDEX_HTML)
}

#[tokio::main]
async fn main() {
    let conn = init_db().expect("Chyba při inicializaci SQLite databáze");
    let db = Arc::new(Mutex::new(conn));

    let app = Router::new()
        .route("/", get(index_handler)) // <--- Webový UI klient
        .route("/api/chat", post(chat_handler))
        .route("/api/chat/stream", post(chat_stream_handler))
        .layer(CorsLayer::permissive())
        .layer(Extension(db));

    let addr = SocketAddr::from(([0, 0, 0, 0], 3000));
    println!("RPAI Backend poslouchá na http://{}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
