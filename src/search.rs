use reqwest::header::{ACCEPT, ACCEPT_LANGUAGE};
use scraper::{Html, Selector};

use std::{
    sync::OnceLock,
    time::Duration,
};

// ============================================================
// CONFIGURATION
// ============================================================

const SEARCH_URL: &str = "https://html.duckduckgo.com/html/";

const SEARCH_LOCALE: &str = "cz-cs";

const MAX_RESULTS: usize = 5;
const MAX_SNIPPET_LENGTH: usize = 500;

const REQUEST_TIMEOUT: Duration = Duration::from_secs(20);

const USER_AGENT_VALUE: &str =
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0";

// ============================================================
// HTTP CLIENT
// ============================================================

// Reuse one HTTP client across all search requests.
// This allows connection pooling and avoids rebuilding the client.

static HTTP_CLIENT: OnceLock<Result<reqwest::Client, String>> =
    OnceLock::new();

fn get_http_client() -> Result<&'static reqwest::Client, String> {
    let client = HTTP_CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .user_agent(USER_AGENT_VALUE)
            .timeout(REQUEST_TIMEOUT)
            .build()
            .map_err(|error| {
                format!("Failed to initialize HTTP client: {}", error)
            })
    });

    match client {
        Ok(client) => Ok(client),
        Err(error) => Err(error.clone()),
    }
}

// ============================================================
// HTML PARSING
// ============================================================

// Extract visible text from an HTML element.
// HTML entities are decoded by the HTML parser.

fn extract_text(element: scraper::ElementRef<'_>) -> String {
    element
        .text()
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

// Safely truncate text without breaking UTF-8 characters.

fn truncate_text(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();

    let truncated: String = chars
        .by_ref()
        .take(max_chars)
        .collect();

    if chars.next().is_some() {
        format!("{}...", truncated)
    } else {
        truncated
    }
}

// Normalize DuckDuckGo result URLs.

fn normalize_url(url: &str) -> String {
    if url.starts_with("//") {
        format!("https:{}", url)
    } else if url.starts_with('/') {
        format!("https://html.duckduckgo.com{}", url)
    } else {
        url.to_string()
    }
}

// ============================================================
// SEARCH RESULT EXTRACTION
// ============================================================

fn extract_search_results(html: &str) -> Vec<String> {
    let document = Html::parse_document(html);

    let result_selector =
        Selector::parse(".result").expect("Invalid result selector");

    let title_selector =
        Selector::parse(".result__a").expect("Invalid title selector");

    let snippet_selector =
        Selector::parse(".result__snippet").expect("Invalid snippet selector");

    let mut results = Vec::new();

    for result in document.select(&result_selector) {
        if results.len() >= MAX_RESULTS {
            break;
        }

        // Extract title and URL.

        let title_element = match result.select(&title_selector).next() {
            Some(element) => element,
            None => continue,
        };

        let title = extract_text(title_element);

        if title.is_empty() {
            continue;
        }

        let url = title_element
            .value()
            .attr("href")
            .map(normalize_url);

        // Extract description.

        let snippet = result
            .select(&snippet_selector)
            .next()
            .map(extract_text)
            .unwrap_or_default();

        let snippet = truncate_text(
            &snippet,
            MAX_SNIPPET_LENGTH,
        );

        // Build structured result.

        let mut entry = format!(
            "{}. {}\n",
            results.len() + 1,
            title
        );

        if let Some(url) = url {
            entry.push_str(&format!(
                "   URL: {}\n",
                url
            ));
        }

        if !snippet.is_empty() {
            entry.push_str(&format!(
                "   Description: {}\n",
                snippet
            ));
        }

        results.push(entry);
    }

    results
}

// ============================================================
// DUCKDUCKGO SEARCH
// ============================================================

pub async fn search_duckduckgo(query: &str) -> String {
    let query = query.trim();

    if query.is_empty() {
        return "No search query provided.".to_string();
    }

    println!(
        "[SEARCH] Querying DuckDuckGo: {}",
        query
    );

    // --------------------------------------------------------
    // INITIALIZE HTTP CLIENT
    // --------------------------------------------------------

    let client = match get_http_client() {
        Ok(client) => client,

        Err(error) => {
            eprintln!("[SEARCH] {}", error);
            return "Web search is currently unavailable.".to_string();
        }
    };

    // --------------------------------------------------------
    // BUILD REQUEST
    // --------------------------------------------------------

    let params = [
        ("q", query),
        ("kl", SEARCH_LOCALE),
    ];

    let response = client
        .post(SEARCH_URL)
        .header(
            ACCEPT,
            "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        )
        .header(
            ACCEPT_LANGUAGE,
            "cs-CZ,cs;q=0.9,en;q=0.8",
        )
        .form(&params)
        .send()
        .await;

    // --------------------------------------------------------
    // HANDLE HTTP RESPONSE
    // --------------------------------------------------------

    let response = match response {
        Ok(response) => response,

        Err(error) => {
            eprintln!(
                "[SEARCH] Request failed: {}",
                error
            );

            return "DuckDuckGo search request failed.".to_string();
        }
    };

    let response = match response.error_for_status() {
        Ok(response) => response,

        Err(error) => {
            eprintln!(
                "[SEARCH] HTTP error: {}",
                error
            );

            return "DuckDuckGo returned an HTTP error.".to_string();
        }
    };

    // --------------------------------------------------------
    // READ HTML
    // --------------------------------------------------------

    let html = match response.text().await {
        Ok(html) => html,

        Err(error) => {
            eprintln!(
                "[SEARCH] Failed to read response: {}",
                error
            );

            return "Failed to read DuckDuckGo response.".to_string();
        }
    };

    // --------------------------------------------------------
    // EXTRACT RESULTS
    // --------------------------------------------------------

    let results = extract_search_results(&html);

    if results.is_empty() {
        eprintln!(
            "[SEARCH] No parsable results returned for query: {}",
            query
        );

        return format!(
            "No parsable search results were found for: {}. \
            The search provider may have changed its HTML structure \
            or returned an unexpected page.",
            query
        );
    }

    // --------------------------------------------------------
    // BUILD FINAL RESPONSE
    // --------------------------------------------------------

    println!(
        "[SEARCH] Successfully retrieved {} results.",
        results.len()
    );

    let combined = results.join("\n");

    format!(
        "DuckDuckGo search results for '{}':\n\n{}",
        query,
        combined
    )
}
