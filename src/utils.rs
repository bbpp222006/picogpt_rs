use std::fs::File;
use std::io::{self, Write};
use std::path::Path;
use std::str::FromStr;
use std::io::prelude::*;
use std::fs;
use indicatif::ProgressStyle;
use anyhow::Result;
use ndarray::{Array, ArrayBase, Data, Dimension, IxDyn, Zip};
use approx::AbsDiffEq;

/// 比较两个数组是否近似相等
pub fn assert_array_abs_diff_eq<A, S, D>(a: &ArrayBase<S, D>, b: &ArrayBase<S, D>, epsilon: A)
where
    A: AbsDiffEq<Epsilon = A> + std::fmt::Display +Copy,
    S: Data<Elem = A>,
    D: Dimension,
{
    assert_eq!(a.shape(), b.shape(), "数组形状不同");

    // 并行遍历并逐元素比较两个数组
    Zip::from(a)
        .and(b)
        .for_each(|ai, bi| {
            assert!(
                A::abs_diff_eq(&ai, &bi, epsilon),
                "元素不同：{} vs {}，误差超出范围 {}",
                ai,
                bi,
                epsilon
            );
        });
}

pub async fn download_gpt2_files(model_size: &str, model_dir: &str) -> Result<()> {
    assert!(["124M", "355M", "774M", "1558M"].contains(&model_size));

    let base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models";

    let filenames = vec![
        "checkpoint",
        "encoder.json",
        "hparams.json",
        "model.ckpt.data-00000-of-00001",
        "model.ckpt.index",
        "model.ckpt.meta",
        "vocab.bpe",
    ];

    for filename in filenames {
        let url = format!("{}/{}/{}", base_url, model_size, &filename);
        let mut response = reqwest::get(&url).await?;
        let content_length = response.content_length().unwrap_or(0);

        let path = Path::new(model_dir).join(&filename);
        let mut file = File::create(&path)?;
        let mut downloaded = 0;

        let pb = indicatif::ProgressBar::new(content_length);
        pb.set_style(ProgressStyle::default_bar()
            .template("{msg} [{bar:40.cyan/blue}] {bytes}/{total_bytes} ({eta})")?
            .progress_chars("#>-"));
        let msg = format!("Fetching {}", filename);
        pb.set_message(msg);

        while let Some(chunk) = response.chunk().await? {
            file.write_all(&chunk)?;
            downloaded += chunk.len() as u64;
            pb.set_position(downloaded);
        }

        pb.finish_with_message("Downloaded");
    }

    Ok(())
}

#[tokio::test]
async fn test_download_gpt2_files() {
    let model_size = "124M";
    let model_dir = "models";

    fs::create_dir_all(model_dir).unwrap();

    download_gpt2_files(model_size, model_dir).await.unwrap();
}