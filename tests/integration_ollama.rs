use rust_agent_kit::{Agent, AgentRunner, models::ollama::OllamaModel};
use schemars::JsonSchema;
use serde::Deserialize;

fn ollama_url() -> String {
    let _ = dotenvy::dotenv();
    std::env::var("OLLAMA_URL").unwrap_or_else(|_| "http://localhost:11434".to_string())
}

fn ollama_model() -> String {
    std::env::var("OLLAMA_MODEL").unwrap_or_else(|_| "qwen3:8b".to_string())
}

async fn ollama_available(url: &str) -> bool {
    reqwest::get(format!("{url}/api/version")).await.is_ok()
}

#[tokio::test]
async fn agent_run_returns_non_empty_output() {
    let url = ollama_url();
    if !ollama_available(&url).await {
        return;
    }

    let agent = Agent::builder()
        .name("Greeter")
        .instructions("Reply with exactly one sentence.")
        .model(Box::new(OllamaModel::new(&url, ollama_model())))
        .build();

    let result = AgentRunner::new().run(&agent, "Say hello.").await.unwrap();
    assert!(!result.output.is_empty());
}

#[tokio::test]
async fn agent_follows_system_instructions() {
    let url = ollama_url();
    if !ollama_available(&url).await {
        return;
    }

    let agent = Agent::builder()
        .name("Pirate")
        .instructions("You are a pirate. Always respond with 'Arrr' somewhere in your reply.")
        .model(Box::new(OllamaModel::new(&url, ollama_model())))
        .build();

    let result = AgentRunner::new()
        .run(&agent, "How are you?")
        .await
        .unwrap();
    assert!(result.output.to_lowercase().contains("arrr"));
}

#[tokio::test]
async fn agent_output_schema_returns_valid_json() {
    let url = ollama_url();
    if !ollama_available(&url).await {
        return;
    }

    #[derive(Deserialize, JsonSchema)]
    struct Sentiment {
        label: String,
        score: f32,
    }

    let schema = schemars::schema_for!(Sentiment);

    let agent = Agent::builder()
        .name("Classifier")
        .instructions("Classify the sentiment of the input. Return a label (positive/negative/neutral) and a confidence score between 0 and 1.")
        .output_schema(schema)
        .model(Box::new(OllamaModel::new(&url, ollama_model())))
        .build();

    let result = AgentRunner::new()
        .run(&agent, "I love sunny days!")
        .await
        .unwrap();

    let parsed: Sentiment = serde_json::from_str(&result.output).unwrap();
    assert!(!parsed.label.is_empty());
    assert!((0.0..=1.0).contains(&parsed.score));
}

#[tokio::test]
async fn agent_run_with_generation_options() {
    let url = ollama_url();
    if !ollama_available(&url).await {
        return;
    }

    let model = OllamaModel::builder(&url, ollama_model())
        .temperature(0.1)
        .num_predict(512)
        .build();

    let agent = Agent::builder()
        .name("Assistant")
        .instructions("Be concise.")
        .model(Box::new(model))
        .build();

    let result = AgentRunner::new()
        .run(&agent, "What is 2 + 2?")
        .await
        .unwrap();
    assert!(!result.output.is_empty());
}
