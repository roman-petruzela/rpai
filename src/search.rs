pub async fn search_duckduckgo(query: &str) -> String {
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
