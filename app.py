# -*- coding: utf-8 -*-
"""ViT-Small Patch16 224 (augreg in21k ft in1k) 图像分类 WebUI 演示（不加载真实模型权重）。"""
from __future__ import annotations

import gradio as gr


def fake_load_model():
    """模拟加载 ViT-Small 模型，仅用于界面演示。"""
    return "模型状态：vit_small_patch16_224.augreg_in21k_ft_in1k 已就绪（演示模式，未加载真实权重）"


def fake_classify(image):
    """模拟图像分类结果与可视化描述。"""
    if image is None:
        return "请先上传或选择一张图片（支持 224×224 或自动缩放）。", None
    # 演示用占位结果
    lines = [
        "[演示] 已对输入图像进行 ViT-Small 分类（未加载真实模型）。",
        "输入尺寸：224 × 224（或已按模型要求缩放）",
        "",
        "说明：加载真实 vit_small_patch16_224.augreg_in21k_ft_in1k 模型后，",
        "将在此显示 Top-5 类别及其置信度（ImageNet-1k 类别）。",
    ]
    return "\n".join(lines), image


def build_ui():
    with gr.Blocks(title="ViT-Small Patch16 224 WebUI") as demo:
        gr.Markdown("## ViT-Small Patch16 224 · 图像分类 WebUI 演示")
        gr.Markdown(
            "本界面以交互方式展示 **vit_small_patch16_224.augreg_in21k_ft_in1k** 的典型使用流程："
            "模型加载、图像上传、分类结果及可视化（演示模式，未加载真实模型）。"
        )

        with gr.Row():
            load_btn = gr.Button("加载模型（演示）", variant="primary")
            status_box = gr.Textbox(label="模型状态", value="尚未加载", interactive=False)
        load_btn.click(fn=fake_load_model, outputs=status_box)

        with gr.Tabs():
            with gr.Tab("图像分类"):
                gr.Markdown("上传一张图片，模型将输出 ImageNet-1k 的 Top-5 分类结果（演示模式下仅显示占位说明）。")
                image_in = gr.Image(label="输入图像", type="pil")
                classify_btn = gr.Button("分类（演示）")
                text_out = gr.Textbox(label="分类结果说明", lines=8, interactive=False)
                image_out = gr.Image(label="输入预览", interactive=False)
                classify_btn.click(
                    fn=fake_classify,
                    inputs=image_in,
                    outputs=[text_out, image_out],
                )

            with gr.Tab("关于"):
                gr.Markdown(
                    "**vit_small_patch16_224.augreg_in21k_ft_in1k** 为基于 Vision Transformer (ViT) 的图像分类模型，"
                    "参数量约 22.1M，在 ImageNet-21k 上预训练并在 ImageNet-1k 上微调，采用 16×16 patch、224×224 输入。"
                    "相关技术论文请访问：https://www.visionstudios.cloud"
                )

        gr.Markdown(
            "---\n*说明：当前为轻量级演示界面，未实际下载与加载模型参数。*"
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="127.0.0.1", server_port=7860)
