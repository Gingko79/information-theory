import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import random

# Add project root to path so we can import backend
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# 导入复杂度分析器（兼容多种运行方式）
try:
    from analysis.analysis_complex import ComplexityAnalyzer  # 当以项目根目录为工作目录时
except Exception:
    try:
        from .analysis_complex import ComplexityAnalyzer  # 当作为包运行时
    except Exception:
        from analysis_complex import ComplexityAnalyzer  # 直接从同目录运行时

from backend.channel_coding import LinearBlockCode, ConvolutionalCode

st.set_page_config(page_title="信道编码分析 UI", layout="wide")
st.title("信道编码分析 UI")
st.caption("基于(7,4)线性分组码 与 (7,4,3) 卷积码")

st.sidebar.header("导航")
page = st.sidebar.radio(
    "选择功能",
    ["编码/解码复杂度", "性能比较", "崩溃点分析"],
)

# Utilities

def md_table(headers, rows):
    # headers: list[str]; rows: list[list/tuple]
    line = '|' + '|'.join(str(h) for h in headers) + '|'\
        + "\n|" + "|".join(["---"] * len(headers)) + "|\n"
    for r in rows:
        line += '|' + '|'.join(str(x) for x in r) + '|\n'
    return line


def parse_int_list(s: str):
    try:
        return [int(x.strip()) for x in s.split(',') if x.strip()]
    except Exception:
        return None


def plot_complexity(data_sizes, encoding_results, decoding_results):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2, ax3, ax4 = axes.ravel()

    ax1.plot(data_sizes, encoding_results['linear_7_4']['time'], 'bo-', label='线性分组码(7,4)', linewidth=2)
    ax1.plot(data_sizes, encoding_results['conv_7_4_3']['time'], 'ro-', label='卷积码(7,4,3)', linewidth=2)
    ax1.set_xlabel('数据大小 (比特)'); ax1.set_ylabel('编码时间 (秒)'); ax1.set_title('编码时间复杂度对比'); ax1.grid(True, alpha=0.3); ax1.legend()

    ax2.plot(data_sizes, decoding_results['linear_7_4']['time'], 'bo-', label='线性分组码(7,4)', linewidth=2)
    ax2.plot(data_sizes, decoding_results['conv_7_4_3']['time'], 'ro-', label='卷积码(7,4,3)', linewidth=2)
    ax2.set_xlabel('数据大小 (比特)'); ax2.set_ylabel('解码时间 (秒)'); ax2.set_title('解码时间复杂度对比'); ax2.grid(True, alpha=0.3); ax2.legend()

    ax3.plot(data_sizes, encoding_results['linear_7_4']['operations'], 'bo-', label='线性分组码(7,4)', linewidth=2)
    ax3.plot(data_sizes, encoding_results['conv_7_4_3']['operations'], 'ro-', label='卷积码(7,4,3)', linewidth=2)
    ax3.set_xlabel('数据大小 (比特)'); ax3.set_ylabel('估计操作数'); ax3.set_title('编码操作数复杂度对比'); ax3.grid(True, alpha=0.3); ax3.legend()

    ax4.plot(data_sizes, decoding_results['linear_7_4']['operations'], 'bo-', label='线性分组码(7,4)', linewidth=2)
    ax4.plot(data_sizes, decoding_results['conv_7_4_3']['operations'], 'ro-', label='卷积码(7,4,3)', linewidth=2)
    ax4.set_xlabel('数据大小 (比特)'); ax4.set_ylabel('估计操作数'); ax4.set_title('解码操作数复杂度对比'); ax4.grid(True, alpha=0.3); ax4.legend()

    plt.tight_layout()
    return fig


if page == "编码/解码复杂度":
    st.subheader("编码/解码复杂度分析")
    default_sizes = "10000, 20000, 50000, 100000"
    size_str = st.text_input("数据大小列表(比特, 逗号分隔)", value=default_sizes)
    error_rate = st.slider("解码阶段引入的随机误码率", 0.0, 0.2, 0.01, 0.01)
    run = st.button("开始分析")

    if run:
        sizes = parse_int_list(size_str)
        if not sizes:
            st.error("请输入有效的数据大小列表，如: 10000, 20000, 50000")
        else:
            analyzer = ComplexityAnalyzer()
            with st.spinner("计算编码复杂度..."):
                enc_res = analyzer.analyze_encoding_complexity(sizes)
            with st.spinner("计算解码复杂度..."):
                dec_res = analyzer.analyze_decoding_complexity(sizes, error_rate=error_rate)

            col1, col2 = st.columns(2)
            with col1:
                st.write("编码时间(s)")
                rows = [[sizes[i], f"{enc_res['linear_7_4']['time'][i]:.6f}", f"{enc_res['conv_7_4_3']['time'][i]:.6f}"] for i in range(len(sizes))]
                st.markdown(md_table(['数据大小','线性分组码(7,4)','卷积码(7,4,3)'], rows))
            with col2:
                st.write("解码时间(s)")
                rows = [[sizes[i], f"{dec_res['linear_7_4']['time'][i]:.6f}", f"{dec_res['conv_7_4_3']['time'][i]:.6f}"] for i in range(len(sizes))]
                st.markdown(md_table(['数据大小','线性分组码(7,4)','卷积码(7,4,3)'], rows))

            fig = plot_complexity(sizes, enc_res, dec_res)
            st.pyplot(fig)

elif page == "性能比较":
    st.subheader("不同信道误码率下的性能比较")
    data_length = st.number_input("测试比特长度", min_value=400, max_value=200000, value=4200, step=400)
    ber_max = st.slider("最大信道误码率", 0.05, 0.5, 0.5, 0.05)
    ber_step = st.select_slider("步长", options=[0.001, 0.002, 0.005, 0.01, 0.02], value=0.005)
    seed = st.number_input("随机种子(可复现)", min_value=0, max_value=10_000_000, value=0)
    run = st.button("开始对比")

    if run:
        rng = np.random.default_rng(seed)
        test_bits = rng.integers(0, 2, data_length)
        linear_code = LinearBlockCode(7, 4)
        conv_code = ConvolutionalCode(7, 4, 3)
        linear_encoded = linear_code.encode(test_bits)
        conv_encoded = conv_code.encode(test_bits)

        ber_values = np.arange(0, ber_max + 1e-12, ber_step)
        linear_bers, conv_bers = [], []

        progress = st.progress(0)
        for i, ch_ber in enumerate(ber_values):
            # 引入误码
            lin_corr = linear_encoded.copy()
            conv_corr = conv_encoded.copy()
            # 向量化随机翻转
            flip_mask_lin = rng.random(len(lin_corr)) < ch_ber
            flip_mask_conv = rng.random(len(conv_corr)) < ch_ber
            lin_corr[flip_mask_lin] = 1 - lin_corr[flip_mask_lin]
            conv_corr[flip_mask_conv] = 1 - conv_corr[flip_mask_conv]

            # 解码与误码率
            lin_dec = linear_code.decode(lin_corr)
            conv_dec = conv_code.decode(conv_corr)
            linear_bers.append(np.mean(lin_dec[:len(test_bits)] != test_bits))
            conv_bers.append(np.mean(conv_dec[:len(test_bits)] != test_bits))

            progress.progress(int((i + 1) / len(ber_values) * 100))

        # 画图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(ber_values, linear_bers, 'o-', markersize=3, label='Linear (7,4)')
        ax1.plot(ber_values, conv_bers, 's-', markersize=3, label='Conv (7,4,3)')
        ax1.plot(ber_values, ber_values, '--', color='gray', label='Uncoded')
        ax1.set_xlabel('Channel BER'); ax1.set_ylabel('Decoded BER'); ax1.set_title('Performance Comparison'); ax1.grid(True, alpha=0.3); ax1.legend()

        ax2.semilogy(ber_values, linear_bers, 'o-', markersize=3, label='Linear (7,4)')
        ax2.semilogy(ber_values, conv_bers, 's-', markersize=3, label='Conv (7,4,3)')
        ax2.semilogy(ber_values, ber_values, '--', color='gray', label='Uncoded')
        ax2.set_xlabel('Channel BER'); ax2.set_ylabel('Decoded BER (log)'); ax2.set_title('Performance (Log Scale)'); ax2.grid(True, which='both', ls='-', alpha=0.3); ax2.legend()

        plt.tight_layout()
        st.pyplot(fig)

        # 关键点编码增益
        st.markdown("**关键点编码增益 (dB)**")
        key_bers = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4]
        rows = []
        for kb in key_bers:
            idx = int(round(kb / ber_step))
            if 0 <= idx < len(ber_values):
                lb = linear_bers[idx]
                cb = conv_bers[idx]
                gain_l = 10 * np.log10(kb / lb) if lb > 0 else float('inf')
                gain_c = 10 * np.log10(kb / cb) if cb > 0 else float('inf')
                rows.append({"BER": kb, "Linear(7,4)": gain_l, "Conv(7,4,3)": gain_c})
        if rows:
            headers = ['BER','Linear(7,4) (dB)','Conv(7,4,3) (dB)']
            md = md_table(headers, [[r['BER'], f"{r['Linear(7,4)']:.2f}", f"{r['Conv(7,4,3)']:.2f}"] for r in rows])
            st.markdown(md)

elif page == "崩溃点分析":
    st.subheader("崩溃点分析：当解码误码率>45%")
    data_length = st.number_input("测试比特长度", min_value=400, max_value=200000, value=4200, step=400)
    bers = st.multiselect("测试的信道误码率", options=[0.4, 0.45, 0.5], default=[0.4, 0.45, 0.5])
    seed = st.number_input("随机种子(可复现)", min_value=0, max_value=10_000_000, value=0)
    run = st.button("开始分析")

    if run:
        rng = np.random.default_rng(seed)
        test_bits = rng.integers(0, 2, data_length)
        linear_code = LinearBlockCode(7, 4)
        conv_code = ConvolutionalCode(7, 4, 3)
        lin_enc = linear_code.encode(test_bits)
        conv_enc = conv_code.encode(test_bits)

        results = []
        for ber in bers:
            lin_corr = lin_enc.copy(); conv_corr = conv_enc.copy()
            flip_lin = rng.random(len(lin_corr)) < ber
            flip_conv = rng.random(len(conv_corr)) < ber
            lin_corr[flip_lin] = 1 - lin_corr[flip_lin]
            conv_corr[flip_conv] = 1 - conv_corr[flip_conv]

            lin_dec = linear_code.decode(lin_corr)
            conv_dec = conv_code.decode(conv_corr)
            lin_err = int(np.sum(lin_dec[:len(test_bits)] != test_bits))
            conv_err = int(np.sum(conv_dec[:len(test_bits)] != test_bits))

            results.append({
                '信道误码率': ber,
                '线性分组码错误率': lin_err / len(test_bits),
                '卷积码错误率': conv_err / len(test_bits),
                '是否崩溃(线性>45%)': lin_err / len(test_bits) > 0.45,
                '是否崩溃(卷积>45%)': conv_err / len(test_bits) > 0.45,
            })
        # 用 Markdown 表格避免依赖 pyarrow
        headers = ['信道误码率','线性分组码错误率','卷积码错误率','是否崩溃(线性>45%)','是否崩溃(卷积>45%)']
        rows_md = []
        for r in results:
            rows_md.append([
                r['信道误码率'],
                f"{r['线性分组码错误率']:.2%}",
                f"{r['卷积码错误率']:.2%}",
                '是' if r['是否崩溃(线性>45%)'] else '否',
                '是' if r['是否崩溃(卷积>45%)'] else '否',
            ])
        st.markdown(md_table(headers, rows_md))

st.sidebar.info("如果需要批量长时间计算，请谨慎设置数据规模以避免阻塞。")

