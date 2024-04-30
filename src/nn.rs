use ndarray::{Array1, ArrayD, IxDyn};
use num_traits::Float;

fn gelu<T>(x: &ArrayD<T>) -> ArrayD<T>
where
    T: Float,
{
    let sqrt_2_pi = T::from(2.0 / std::f64::consts::PI).unwrap().sqrt();
    let coeff = T::from(0.044715).unwrap();

    x.mapv(|xi| {
        T::from(0.5).unwrap() * xi * (T::one() + (sqrt_2_pi * (xi + coeff * xi.powi(3))).tanh())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    use ndarray::array; // 需要在Cargo.toml中添加approx库作为依赖

    #[tokio::test]
    async fn test_gelu() {
        // 创建测试数据
        let x = ArrayD::from_shape_vec(IxDyn(&[2, 2]), vec![1.0, 2.0, -2.0, 0.5]).unwrap();
        let expected = array![
            [0.845397, 1.964027],
            [-0.045402, 0.345731]
        ].into_dyn(); // expected values based on GELU formula application

        // 调用gelu函数
        let result = gelu(&x);

        // 检查结果是否与预期相符
        assert_relative_eq!(result, expected, epsilon = 1e-6);
    }
}
