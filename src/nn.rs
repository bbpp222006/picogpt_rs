use ndarray::{s, Array, ArrayD, Axis, Ix2, Ix3, IxDyn, Slice, SliceInfo};

pub fn gelu(x: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
    // √(2 / π)
    let sqrt_2_pi = (2.0 / std::f64::consts::PI).sqrt();
    let coeff = 0.044715;

    // 对输入数组的每个元素应用 GELU 函数
    x.mapv(|xi| 0.5 * xi * (1.0 + (sqrt_2_pi * (xi + coeff * xi.powi(3))).tanh()))
}

/// 生成具有因果遮罩的矩阵（类似 NumPy 的 tri 函数）
/// 参数：
/// - `size`: 矩阵的大小 [n_seq, n_seq]
/// - `mask_value`: 掩码值（填充负无穷大的值）
/// 返回一个 [size, size] 大小的因果遮罩矩阵
fn causal_mask(size: usize, mask_value: f64) -> Array<f64,Ix2> {
    // 创建一个全 0 的矩阵
    let mut mask = Array::<f64,Ix2>::zeros((size, size));

    // 设置为上三角（排除对角线），值为 mask_value
    for i in 0..size {
        for j in (i + 1)..size {
            mask[[i, j]] = mask_value;
        }
    }

    mask
}

pub fn softmax(x: &Array<f64, IxDyn>) -> Array<f64, IxDyn> {
    // Find the maximum value along the last axis for numerical stability
    let max = x.map_axis(Axis(x.ndim() - 1), |row| {
        *row.iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap()
    });

    // Subtract max from each element to prevent overflow
    let shifted = x - &max.insert_axis(Axis(x.ndim() - 1));

    // Calculate the exponentials and their sum
    let exp_x = shifted.mapv(f64::exp);
    let sum_exp = exp_x
        .sum_axis(Axis(x.ndim() - 1))
        .insert_axis(Axis(x.ndim() - 1));

    // Divide each element by the sum to normalize
    exp_x / sum_exp
}

pub fn layer_norm(
    x: &Array<f64, IxDyn>,
    g: &Array<f64, IxDyn>,
    b: &Array<f64, IxDyn>,
    eps: f64,
) -> Array<f64, IxDyn> {
    // 确保输入形状一致
    assert_eq!(x.shape(), g.shape(), "输入数组和 g (gamma) 形状不匹配");
    assert_eq!(x.shape(), b.shape(), "输入数组和 b (beta) 形状不匹配");

    // 计算最后一个轴的均值
    let mean = x
        .mean_axis(Axis(x.ndim() - 1))
        .unwrap()
        .insert_axis(Axis(x.ndim() - 1));

    // 计算最后一个轴的方差
    let variance = x
        .var_axis(Axis(x.ndim() - 1), 0.0)
        .insert_axis(Axis(x.ndim() - 1));

    // 归一化操作：mean=0，var=1
    let normalized = (x - &mean) / (variance + eps).mapv(f64::sqrt);

    // 使用 g 和 b 对归一化后的数据进行缩放和偏移
    g * &normalized + b
}

/// 执行线性层的计算：x * w + b
/// 参数：
/// - `x`: 输入矩阵，形状为 [m, in]
/// - `w`: 权重矩阵，形状为 [in, out]
/// - `b`: 偏移向量，形状为 [out]
/// 返回：形状为 [m, out] 的结果矩阵
fn linear(x: &Array<f64, Ix2>, w: &Array<f64, Ix2>, b: &Array<f64, Ix2>) -> Array<f64, Ix2> {
    assert_eq!(x.shape()[1], w.shape()[0], "输入矩阵和权重矩阵的形状不匹配");
    assert_eq!(w.shape()[1], b.shape()[1], "权重矩阵和偏移向量的形状不匹配");

    // 执行矩阵乘法：x @ w
    let result = x.dot(w);

    // 将偏移量 b 添加到每个结果向量
    result + b
}

/// FFN (前馈神经网络) 实现
fn ffn(
    x: &Array<f64, Ix2>,
    c_fc_w: &Array<f64, Ix2>,
    c_fc_b: &Array<f64, Ix2>,
    c_proj_w: &Array<f64, Ix2>,
    c_proj_b: &Array<f64, Ix2>,
) -> Array<f64, Ix2> {
    // 使用 c_fc 进行扩展，并通过 GELU 激活
    let a = gelu(&linear(x, c_fc_w, c_fc_b).into_dyn())
        .into_dimensionality::<Ix2>()
        .unwrap();

    // 使用 c_proj 进行还原
    linear(&a, c_proj_w, c_proj_b)
}

/// Attention 实现
fn attention(q: &Array<f64, Ix2>, k: &Array<f64, Ix2>, v: &Array<f64, Ix2>,mask: &Array<f64, Ix2>) -> Array<f64, Ix2> {
    // 确保输入的维度一致
    assert_eq!(q.shape()[1], k.shape()[1], "查询矩阵和键矩阵的特征维度不同");
    assert_eq!(k.shape()[0], v.shape()[0], "键矩阵和值矩阵的数量不同");

    // 计算 q 与 k.T 的点积，并缩放
    let scale_factor = (q.shape()[1] as f64).sqrt();
    let scores = q.dot(&k.t()) / scale_factor;

    // 对点积结果应用 softmax
    let softmax_scores = softmax(&(scores.into_dyn()+mask));
    // 乘以值矩阵 v
    softmax_scores.into_dimensionality::<Ix2>().unwrap().dot(v)
}


/// 按照指定轴分割任意维度数组
fn split_array(x: &Array<f64, IxDyn>, n_splits: usize, axis: usize) -> Array<f64, IxDyn>
{
    // 确保指定轴的大小可以被均匀分割
    let axis_size = x.shape()[axis];
    assert!(
        axis_size % n_splits == 0,
        "指定轴上的大小不能被均匀分割"
    );

    // 每个分割后的块的大小
    let split_size = axis_size / n_splits;

    // 创建输出数组的形状
    let mut result_shape = x.shape().to_vec();
    result_shape[axis] = split_size;
    result_shape.insert(0, n_splits);

    // 创建一个动态维度的输出数组
    let mut result = Array::<f64, IxDyn>::zeros(IxDyn(&result_shape));

    // 使用切片将原始数组分成多个子块，并将它们放入 `result` 中
    for i in 0..n_splits {
        // 获取当前块的范围
        let start = i * split_size;
        let end = start + split_size;
        // 构建切片，指定沿指定轴的范围
        let mut slice_info = vec![s![..]; x.ndim()];
        slice_info[axis] = s![start..end];

        // 获取子块并插入到 `result` 中
        let sub_array = x.slice(slice_info[..]);
        let mut result_view = result.index_axis_mut(Axis(0), i);
        result_view.assign(&sub_array);
    }

    


    result
}

/// [n_seq, n_embd] -> [n_seq, n_embd]
fn self_attention(
    x: &Array<f64, Ix2>, //[n_seq, n_embd]
    c_attn_w: &Array<f64, Ix2>,
    c_attn_b: &Array<f64, Ix2>,
    c_proj_w: &Array<f64, Ix2>,
    c_proj_b: &Array<f64, Ix2>,
) -> Array<f64, Ix2> {
    // 分别计算查询、键和值
    // [n_seq, n_embd] -> [n_seq, 3*n_embd]
    let qkv = linear(x, c_attn_w, c_attn_b);
    let q = qkv.slice(s![.., 0..x.shape()[1]]).to_owned();
    let k = qkv.slice(s![.., x.shape()[1]..2 * x.shape()[1]]).to_owned();
    let v = qkv.slice(s![.., 2 * x.shape()[1]..]).to_owned();

    let mask = causal_mask(x.shape()[0], -1e10);   

    // 使用 attention 函数计算自注意力
    let attention_output = attention(&q, &k, &v,&mask);

    // 使用 c_proj 进行还原
    linear(&attention_output, c_proj_w, c_proj_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::assert_array_abs_diff_eq;
    use ndarray::array;
    use rand::thread_rng;
    use rand::Rng;
    use rand_distr::StandardNormal;

    #[tokio::test]
    async fn test_gelu() {
        // 创建测试数据
        // 示例输入矩阵，动态维度
        let input = array![[1.0, 2.0], [3.0, 4.0]].into_dyn();

        // 调用 gelu 函数
        let result = gelu(&input);

        // 预期输出
        let expected = array![[0.841192, 1.954597], [2.995732, 3.999999]].into_dyn();

        println!("{:?}", result);
        // 使用近似比较宏进行验证
        assert_array_abs_diff_eq(&result, &expected, 1e-2);
    }

    #[tokio::test]
    async fn test_softmax() {
        // 创建测试数据
        // 示例输入矩阵，动态维度
        let input = array![[2.0, 100.0], [-5.0, 0.0]].into_dyn();

        // 调用 softmax 函数
        let result = softmax(&input);

        // 预期输出
        let expected = array![
            [2.74878501e-43, 1.00000000e+00],
            [6.69285092e-03, 9.93307149e-01]
        ]
        .into_dyn();
        println!("{:?}", result);

        // 使用近似比较宏进行验证
        assert_array_abs_diff_eq(&result, &expected, 1e-5);
    }

    #[tokio::test]
    async fn test_layer_norm() {
        // 创建测试数据
        let input = array![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]].into_dyn();
        let g = array![[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]].into_dyn(); // gamma
        let b = array![[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]].into_dyn(); // beta
        let eps = 1e-5;

        // 调用 layer_norm 函数
        let result = layer_norm(&input, &g, &b, eps);

        // 预期输出
        let expected =
            array![[-1.2247449, 0.0, 1.2247449], [-1.2247449, 0.0, 1.2247449]].into_dyn();
        println!("{:?}", result);

        // 使用近似比较宏进行验证
        assert_array_abs_diff_eq(&result, &expected, 1e-5);
    }

    #[tokio::test]
    async fn test_linear() {
        /// 生成具有正态分布的随机矩阵
        fn random_normal_array(rows: usize, cols: usize) -> Array<f64,Ix2> {
            let mut rng = thread_rng();
            let distribution = StandardNormal;
            let data: Vec<f64> = (0..(rows * cols))
                .map(|_| rng.sample(distribution))
                .collect();
            Array::<f64,Ix2>::from_shape_vec((rows, cols), data).unwrap()
        }
        // 生成随机输入矩阵，形状为 (64, 784)
        let x = random_normal_array(64, 784);

        // 生成随机权重矩阵，形状为 (784, 10)
        let w = random_normal_array(784, 10);

        // 生成随机偏移矩阵，形状为 (1, 10)
        let b = random_normal_array(1, 10);

        // 调用线性层函数
        let result = linear(&x, &w, &b);

        // 检查输出的形状是否符合预期
        assert_eq!(result.shape(), &[64, 10]);

        // 打印形状以验证
        println!("输入形状: {:?}", x.shape());
        println!("输出形状: {:?}", result.shape());
    }

    #[tokio::test]
    async fn test_ffn() {
        /// 生成具有正态分布的随机矩阵
        fn random_normal_array(rows: usize, cols: usize) -> Array<f64,Ix2> {
            let mut rng = thread_rng();
            let distribution = StandardNormal;
            let data: Vec<f64> = (0..(rows * cols))
                .map(|_| rng.sample(distribution))
                .collect();
            Array::<f64,Ix2>::from_shape_vec((rows, cols), data).unwrap()
        }

        // 输入维度
        let n_seq = 64;
        let n_embd = 784;
        let n_proj = 10;

        // 生成随机参数
        let c_fc_w = random_normal_array(n_embd, 4 * n_embd);
        let c_fc_b = random_normal_array(1, 4 * n_embd);
        let c_proj_w = random_normal_array(4 * n_embd, n_proj);
        let c_proj_b = random_normal_array(1, n_proj);

        // 生成随机输入矩阵
        let x = random_normal_array(n_seq, n_embd);

        // 调用 FFN 函数
        let result = ffn(&x, &c_fc_w, &c_fc_b, &c_proj_w, &c_proj_b);

        // 检查输出的形状是否符合预期
        assert_eq!(result.shape(), &[n_seq, n_proj]);

        // 打印形状以验证
        println!("输入形状: {:?}", x.shape());
        println!("输出形状: {:?}", result.shape());
    }

    #[tokio::test]
    async fn test_attention() {
        // 示例输入矩阵
        let q = array![[1.0, 0.0], [0.0, 1.0]];
        let k = array![[1.0, 0.0], [0.0, 1.0]];
        let v = array![[1.0, 2.0], [3.0, 4.0]];

        // 调用 attention 函数
        let result = attention(&q, &k, &v);

        // 打印结果
        println!("{:?}", result);
    }
}
