use rust_agent_kit::{Agent, AgentRunner, models::gemini::GeminiModel};

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
