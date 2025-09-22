use crate::config::load::Parameters;
use crate::handlers::document::{Document, DocumentformInterface};
use custom_logger as log;
use serde_derive::{Deserialize, Serialize};
use serde_json::Value;
use std::fs;

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LlamaResponse {
    pub index: i64,
    pub content: String,
    pub tokens: Vec<Value>,
    #[serde(rename = "id_slot")]
    pub id_slot: i64,
    pub stop: bool,
    pub model: String,
    #[serde(rename = "tokens_predicted")]
    pub tokens_predicted: i64,
    #[serde(rename = "tokens_evaluated")]
    pub tokens_evaluated: i64,
    #[serde(rename = "generation_settings")]
    pub generation_settings: GenerationSettings,
    pub prompt: String,
    #[serde(rename = "has_new_line")]
    pub has_new_line: bool,
    pub truncated: bool,
    #[serde(rename = "stop_type")]
    pub stop_type: String,
    #[serde(rename = "stopping_word")]
    pub stopping_word: String,
    #[serde(rename = "tokens_cached")]
    pub tokens_cached: i64,
    pub timings: Timings,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct GenerationSettings {
    #[serde(rename = "n_predict")]
    pub n_predict: i64,
    pub seed: i64,
    pub temperature: f64,
    #[serde(rename = "dynatemp_range")]
    pub dynatemp_range: f64,
    #[serde(rename = "dynatemp_exponent")]
    pub dynatemp_exponent: f64,
    #[serde(rename = "top_k")]
    pub top_k: i64,
    #[serde(rename = "top_p")]
    pub top_p: f64,
    #[serde(rename = "min_p")]
    pub min_p: f64,
    #[serde(rename = "top_n_sigma")]
    pub top_n_sigma: f64,
    #[serde(rename = "xtc_probability")]
    pub xtc_probability: f64,
    #[serde(rename = "xtc_threshold")]
    pub xtc_threshold: f64,
    #[serde(rename = "typical_p")]
    pub typical_p: f64,
    #[serde(rename = "repeat_last_n")]
    pub repeat_last_n: i64,
    #[serde(rename = "repeat_penalty")]
    pub repeat_penalty: f64,
    #[serde(rename = "presence_penalty")]
    pub presence_penalty: f64,
    #[serde(rename = "frequency_penalty")]
    pub frequency_penalty: f64,
    #[serde(rename = "dry_multiplier")]
    pub dry_multiplier: f64,
    #[serde(rename = "dry_base")]
    pub dry_base: f64,
    #[serde(rename = "dry_allowed_length")]
    pub dry_allowed_length: i64,
    #[serde(rename = "dry_penalty_last_n")]
    pub dry_penalty_last_n: i64,
    #[serde(rename = "dry_sequence_breakers")]
    pub dry_sequence_breakers: Vec<String>,
    pub mirostat: i64,
    #[serde(rename = "mirostat_tau")]
    pub mirostat_tau: f64,
    #[serde(rename = "mirostat_eta")]
    pub mirostat_eta: f64,
    pub stop: Vec<Value>,
    #[serde(rename = "max_tokens")]
    pub max_tokens: i64,
    #[serde(rename = "n_keep")]
    pub n_keep: i64,
    #[serde(rename = "n_discard")]
    pub n_discard: i64,
    #[serde(rename = "ignore_eos")]
    pub ignore_eos: bool,
    pub stream: bool,
    #[serde(rename = "logit_bias")]
    pub logit_bias: Vec<Value>,
    #[serde(rename = "n_probs")]
    pub n_probs: i64,
    #[serde(rename = "min_keep")]
    pub min_keep: i64,
    pub grammar: String,
    #[serde(rename = "grammar_lazy")]
    pub grammar_lazy: bool,
    #[serde(rename = "grammar_triggers")]
    pub grammar_triggers: Vec<Value>,
    #[serde(rename = "preserved_tokens")]
    pub preserved_tokens: Vec<Value>,
    #[serde(rename = "chat_format")]
    pub chat_format: String,
    #[serde(rename = "reasoning_format")]
    pub reasoning_format: String,
    #[serde(rename = "reasoning_in_content")]
    pub reasoning_in_content: bool,
    #[serde(rename = "thinking_forced_open")]
    pub thinking_forced_open: bool,
    pub samplers: Vec<String>,
    #[serde(rename = "speculative.n_max")]
    pub speculative_n_max: i64,
    #[serde(rename = "speculative.n_min")]
    pub speculative_n_min: i64,
    #[serde(rename = "speculative.p_min")]
    pub speculative_p_min: f64,
    #[serde(rename = "timings_per_token")]
    pub timings_per_token: bool,
    #[serde(rename = "post_sampling_probs")]
    pub post_sampling_probs: bool,
    pub lora: Vec<Value>,
}

#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct Timings {
    #[serde(rename = "prompt_n")]
    pub prompt_n: i64,
    #[serde(rename = "prompt_ms")]
    pub prompt_ms: f64,
    #[serde(rename = "prompt_per_token_ms")]
    pub prompt_per_token_ms: f64,
    #[serde(rename = "prompt_per_second")]
    pub prompt_per_second: f64,
    #[serde(rename = "predicted_n")]
    pub predicted_n: i64,
    #[serde(rename = "predicted_ms")]
    pub predicted_ms: f64,
    #[serde(rename = "predicted_per_token_ms")]
    pub predicted_per_token_ms: f64,
    #[serde(rename = "predicted_per_second")]
    pub predicted_per_second: f64,
}

pub trait AgentInterface {
    async fn execute(params: Parameters, key: String)
    -> Result<String, Box<dyn std::error::Error>>;
}

pub struct Agent {}

impl AgentInterface for Agent {
    async fn execute(
        params: Parameters,
        key: String,
    ) -> Result<String, Box<dyn std::error::Error>> {
        let db_path = params.db_path.clone();
        let fd = Document::get_formdata(format!("{}/queue", db_path.clone()), key.clone()).await?;
        log::debug!("[execute] llama agent {:?}", fd);
        let prompt = fd.prompt;
        let data = match params.test {
            true => {
                log::info!("mode: test");
                fs::read("/home/lzuccarelli/Projects/rust-llama-agent/docs/example-response.json")?
            }
            false => {
                log::info!("mode: execute");
                let llama_url = format!("{}", params.base_url);
                log::debug!("[execute] url {}", llama_url);
                let llama_payload = get_llama_payload(prompt);
                log::debug!("payload {}", llama_payload);
                let client = reqwest::Client::new();
                let res = client.post(llama_url).body(llama_payload).send().await;
                log::debug!("[execute] headers received");
                match res {
                    Ok(data) => {
                        log::debug!("[execute] waiting for body");
                        let data_result = data.bytes().await?;
                        log::debug!(
                            "[execute] body received {}",
                            String::from_utf8(data_result.to_vec()).unwrap(),
                        );
                        data_result.to_vec()
                    }
                    Err(_) => {
                        vec![]
                    }
                }
            }
        };
        let llama: LlamaResponse = serde_json::from_slice(&data)?;
        let llama_document = llama.content.clone();
        log::info!("result from llama\n\n {}", llama_document);
        Document::save_formdata(db_path, key, llama_document).await?;
        Ok("exit => 0".to_string())
    }
}

fn get_llama_payload(prompt: String) -> String {
    let payload = format!(
        r#"
    {{
        "prompt": "{}" , "n_predict": 128 
    }}"#,
        prompt
    );
    payload
}
