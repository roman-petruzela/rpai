use reqwest::header::{ACCEPT, ACCEPT_LANGUAGE};
use scraper::{Html, Selector};
use std::sync::OnceLock;
use std::time::Duration;

const SEARCH_URL: &str = "https://html.duckduckgo.com/html/";
const USER_AGENT: &str = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36";
const MAX_RESULTS: usize = 5;
const MAX_SNIPPET_CHARS: usize = 500;

static HTTP_CLIENT: OnceLock<Result<reqwest::Client, String>> = OnceLock::new();

fn http_client() -> Result<&'static reqwest::Client, String> {
    match HTTP_CLIENT.get_or_init(|| {
        reqwest::Client::builder()
            .user_agent(USER_AGENT)
            .timeout(Duration::from_secs(20))
            .build()
            .map_err(|error| error.to_string())
    }) {
        Ok(client) => Ok(client),
        Err(error) => Err(error.to_string()),
    }
}

fn visible_text(element: scraper::ElementRef<'_>) -> String {
    element
        .text()
        .collect::<Vec<_>>()
        .join(" ")
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn truncate_chars(text: &str, max_chars: usize) -> String {
    let mut chars = text.chars();
    let truncated: String = chars.by_ref().take(max_chars).collect();

    if chars.next().is_some() {
        format!("{}...", truncated)
    } else {
        truncated
    }
}

fn extract_results(html: &str) -> Vec<String> {
    let document = Html::parse_document(html);
    let result_selector = match Selector::parse(".result") {
        Ok(selector) => selector,
        Err(_) => return Vec::new(),
    };
    let title_selector = match Selector::parse(".result__a") {
        Ok(selector) => selector,
        Err(_) => return Vec::new(),
    };
    let snippet_selector = match Selector::parse(".result__snippet") {
        Ok(selector) => selector,
        Err(_) => return Vec::new(),
    };

    let mut results = Vec::new();

    for result in document.select(&result_selector) {
        if results.len() >= MAX_RESULTS {
            break;
        }

        let title_element = match result.select(&title_selector).next() {
            Some(element) => element,
            None => continue,
        };

        let title = visible_text(title_element);
        if title.is_empty() {
            continue;
        }

        let url = title_element.value().attr("href").unwrap_or("");
        let snippet = result
            .select(&snippet_selector)
            .next()
            .map(visible_text)
            .unwrap_or_default();

        let mut entry = format!("{}. {}", results.len() + 1, title);

        if !url.is_empty() {
            entry.push_str(&format!("\n   URL: {}", url));
        }

        if !snippet.is_empty() {
            entry.push_str(&format!(
                "\n   Description: {}",
                truncate_chars(&snippet, MAX_SNIPPET_CHARS)
            ));
        }

        results.push(entry);
    }

    results
}

pub async fn search_duckduckgo(query: &str) -> String {
    let query = query.trim();
    if query.is_empty() {
        return "No search query provided.".to_string();
    }

    println!("[SEARCH] Querying DuckDuckGo: {}", query);

    let client = match http_client() {
        Ok(client) => client,
        Err(error) => {
            eprintln!("[SEARCH] HTTP client initialization failed: {}", error);
            return "Web search is currently unavailable.".to_string();
        }
    };

    let response = match client
        .post(SEARCH_URL)
        .header(ACCEPT, "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8")
        .header(ACCEPT_LANGUAGE, "cs-CZ,cs;q=0.9,en;q=0.8")
        .form(&[("q", query), ("kl", "cz-cs")])
        .send()
        .await
    {
        Ok(response) => response,
        Err(error) => {
            eprintln!("[SEARCH] Request failed: {}", error);
            return "DuckDuckGo search request failed.".to_string();
        }
    };

    let response = match response.error_for_status() {
        Ok(response) => response,
        Err(error) => {
            eprintln!("[SEARCH] HTTP response error: {}", error);
            return "DuckDuckGo returned an HTTP error.".to_string();
        }
    };

    let html = match response.text().await {
        Ok(html) => html,
        Err(error) => {
            eprintln!("[SEARCH] Failed to read response: {}", error);
            return "Failed to read DuckDuckGo response.".to_string();
        }
    };

    let results = extract_results(&html);
    if results.is_empty() {
        eprintln!("[SEARCH] No parsable results found.");
        return "No parsable search results were returned by DuckDuckGo.".to_string();
    }

    println!("[SEARCH] Retrieved {} results.", results.len());
    format!("DuckDuckGo search results for '{}':\n\n{}", query, results.join("\n\n"))
}
