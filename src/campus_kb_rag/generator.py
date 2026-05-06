"""Answer generation for campus KB RAG."""

from __future__ import annotations

from typing import Any, Dict, List


class CampusAnswerGenerator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self._pipe = None

    def generate(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        if not evidence:
            return self._refusal(query)
        gen_cfg = self.config.get("generation", {})
        if gen_cfg.get("backend", "extractive") == "llm":
            return self._generate_with_llm(query, evidence)
        return self._generate_extractive(query, evidence)

    def build_prompt(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        prompt_cfg = self.config.get("prompt", {})
        persona = prompt_cfg.get("persona", "校园办事指南助手")
        institution = prompt_cfg.get("institution", "学校")
        max_chars = int(prompt_cfg.get("max_context_chars", 2600))

        blocks = []
        used = 0
        for i, item in enumerate(evidence, start=1):
            block = (
                f"[{i}] 标题：{item['title']}\n"
                f"部门：{item.get('department', '')}\n"
                f"来源：{item.get('source', '')}\n"
                f"更新时间：{item.get('updated_at', '')}\n"
                f"内容：{item['text']}\n"
            )
            if used + len(block) > max_chars:
                break
            blocks.append(block)
            used += len(block)

        context = "\n".join(blocks)
        return (
            f"你是{persona}，服务对象是{institution}学生。\n"
            "请严格依据【检索证据】回答，不能把模型常识当成学校政策。\n"
            "若证据不足，请直接说明“当前知识库没有足够依据”，并建议查看主管部门官方通知。\n"
            "回答要求：先给结论，再列出办理建议；每个关键结论后用 [1]、[2] 形式引用证据编号。\n\n"
            f"【检索证据】\n{context}\n"
            f"【用户问题】\n{query}\n"
            "【回答】"
        )

    def _generate_extractive(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        top = evidence[0]
        lines = [
            f"根据当前知识库，最相关的是「{top['title']}」[1]。",
            self._shorten(top["text"]),
        ]
        if len(evidence) > 1:
            lines.append(f"可同时参考「{evidence[1]['title']}」[2]。")
        lines.append("正式办理前建议以南京大学对应主管部门的最新通知或系统页面为准。")
        return "\n".join(lines)

    def _generate_with_llm(self, query: str, evidence: List[Dict[str, Any]]) -> str:
        gen_cfg = self.config.get("generation", {})
        pipe = self._get_pipeline()
        prompt = self.build_prompt(query, evidence)
        output = pipe(
            prompt,
            max_new_tokens=int(gen_cfg.get("max_new_tokens", 384)),
            do_sample=float(gen_cfg.get("temperature", 0.2)) > 0,
            temperature=float(gen_cfg.get("temperature", 0.2)),
            return_full_text=False,
        )
        if isinstance(output, list) and output:
            return str(output[0].get("generated_text", "")).strip()
        return str(output).strip()

    def _get_pipeline(self):
        if self._pipe is not None:
            return self._pipe

        import torch
        from transformers import pipeline

        gen_cfg = self.config.get("generation", {})
        model_name = str(gen_cfg.get("model_name", "Qwen/Qwen2-1.5B-Instruct"))
        adapter_path = gen_cfg.get("lora_adapter_path")

        if adapter_path:
            from peft import AutoPeftModelForCausalLM
            from transformers import AutoTokenizer

            model = AutoPeftModelForCausalLM.from_pretrained(
                adapter_path,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=gen_cfg.get("device_map", "auto"),
            )
            tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
            self._pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        else:
            self._pipe = pipeline(
                "text-generation",
                model=model_name,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map=gen_cfg.get("device_map", "auto"),
                trust_remote_code=True,
            )
        return self._pipe

    @staticmethod
    def _shorten(text: str, limit: int = 220) -> str:
        normalized = " ".join((text or "").split())
        if len(normalized) <= limit:
            return normalized
        return normalized[:limit].rstrip() + "..."

    @staticmethod
    def _refusal(query: str) -> str:
        return (
            f"当前知识库没有足够依据回答「{query}」。"
            "建议查看南京大学对应主管部门的官方通知，或联系院系/信息化服务窗口确认。"
        )
