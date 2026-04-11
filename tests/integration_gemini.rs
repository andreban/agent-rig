use rust_agent_kit::{Agent, AgentRunner, models::gemini::GeminiModel};
use schemars::JsonSchema;
use serde::Deserialize;

const MODEL: &str = "gemini-3.1-flash-lite-preview";

fn api_key() -> Option<String> {
    let _ = dotenvy::dotenv();
    std::env::var("GEMINI_API_KEY").ok()
}

#[tokio::test]
async fn agent_run_returns_non_empty_output() {
    let Some(api_key) = api_key() else { return };

    let agent = Agent::builder()
        .name("Greeter")
        .instructions("Reply with exactly one sentence.")
        .model(Box::new(GeminiModel::new(api_key, MODEL)))
        .build();

    let result = AgentRunner::new().run(&agent, "Say hello.").await.unwrap();
    assert!(!result.output.is_empty());
}

#[tokio::test]
async fn agent_follows_system_instructions() {
    let Some(api_key) = api_key() else { return };

    let agent = Agent::builder()
        .name("Pirate")
        .instructions("You are a pirate. Always respond with 'Arrr' somewhere in your reply.")
        .model(Box::new(GeminiModel::new(api_key, MODEL)))
        .build();

    let result = AgentRunner::new()
        .run(&agent, "How are you?")
        .await
        .unwrap();
    assert!(result.output.to_lowercase().contains("arrr"));
}

#[tokio::test]
async fn agent_output_schema_returns_valid_json() {
    let Some(api_key) = api_key() else { return };

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
        .model(Box::new(GeminiModel::new(api_key, MODEL)))
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
async fn agent_run_with_temperature_setting() {
    let Some(api_key) = api_key() else { return };

    let model = GeminiModel::builder(api_key, MODEL)
        .temperature(0.1)
        .max_output_tokens(64)
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
