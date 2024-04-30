mod utils;
mod nn;


use anyhow::Result;
use std::fs;
use utils::download_gpt2_files;

#[tokio::main]
async fn main() -> Result<()> {
    let model_size = "124M";
    let model_dir = "models";

    fs::create_dir_all(model_dir)?;

    download_gpt2_files(model_size, model_dir).await?;

    println!("All files downloaded successfully!");

    Ok(())
}
