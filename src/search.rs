use reqwest::header::{ACCEPT_LANGUAGE, USER_AGENT};

pub async fn search_duckduckgo(query: &str) -> String {
    println!("[SEARCH] Vyhledávám na DuckDuckGo: {}", query);
    let client = reqwest::Client::new();
    
    // DuckDuckGo spolehlivě odpovídá na POST požadavek s formulářovými daty
    let params = [("q", query), ("b", ""), ("kl", "cz-cs")];
    
    let res = client
        .post("https://html.duckduckgo.com/html/")
        .header(USER_AGENT, "Mozilla/5.0 (X11; Arch Linux; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0")
        .header(ACCEPT_LANGUAGE, "cs,en-US;q=0.7,en;q=0.3")
        .form(&params)
        .send()
        .await;

    if let Ok(response) = res {
        if let Ok(text) = response.text().await {
            let mut results = Vec::new();
            
            for chunk in text.split("class=\"result__snippet\"") {
                if let Some(start) = chunk.find('>') {
                    if let Some(end) = chunk[start..].find("</a>") {
                        let snippet = &chunk[start + 1..start + end];
                        let clean = snippet
                            .replace("<b>", "")
                            .replace("</b>", "")
                            .replace("&quot;", "\"")
                            .replace("&amp;", "&")
                            .replace("&#x27;", "'")
                            .replace("\n", " ")
                            .trim()
                            .to_string();
                        
                        if !clean.is_empty() && clean.len() > 15 {
                            results.push(clean);
                        }
                        if results.len() >= 4 {
                            break;
                        }
                    }
                }
            }

            if !results.is_empty() {
                let combined = results.join("\n- ");
                println!("[SEARCH] Nalezeno:\n- {}", combined);
                return format!("Výsledky z webu:\n- {}", combined);
            }
        }
    }

    println!("[SEARCH] Nepodařilo se získat výsledky z webu.");
    "Nenalezeny žádné aktuální výsledky z vyhledávání.".to_string()
}
