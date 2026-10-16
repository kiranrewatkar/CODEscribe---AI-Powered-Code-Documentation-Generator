"""
app/generator.py

Inference and Markdown documentation synthesis engine for CODEscribe.
Connects the local fine-tuned CodeT5 model, AST docstring stripping,
metadata extraction, and AST-to-Mermaid flowchart generation.

Key Architecture:
- Single-instance model loading at application startup.
- Local execution only: strictly loads from local model checkpoints.
- Hardware-adaptive: leverages CUDA with PyTorch inference mode; falls back to CPU.
- Never executes user source code: uses AST parsing and feature extraction only.
- Assembles structured production-grade Markdown API documentation.
"""

from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple

import torch

from app.ast_mermaid import generate_mermaid_flowchart
from training.remove_docstrings import remove_docstrings_from_source

logger = logging.getLogger("CODEscribe.Generator")


class ModelNotLoadedError(RuntimeError):
    """Raised when an inference request is made before the model is loaded."""
    pass


class CodeSyntaxError(ValueError):
    """Raised when submitted Python code has syntax errors."""
    pass


@dataclass
class TimingMetadata:
    preprocessing_ms: float = 0.0
    inference_ms: float = 0.0
    mermaid_ms: float = 0.0
    total_ms: float = 0.0


@dataclass
class ASTMetadata:
    function_names: List[str] = field(default_factory=list)
    parameters: List[str] = field(default_factory=list)
    return_annotation: Optional[str] = None
    docstring_stripped: bool = False
    syntax_valid: bool = True


@dataclass
class GenerationResult:
    success: bool
    markdown: str
    docstring: str
    mermaid_code: Optional[str] = None
    ast_metadata: ASTMetadata = field(default_factory=ASTMetadata)
    timing_ms: TimingMetadata = field(default_factory=TimingMetadata)
    device: str = "cpu"
    status: str = "SUCCESS"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "markdown": self.markdown,
            "docstring": self.docstring,
            "mermaid_code": self.mermaid_code,
            "ast_metadata": asdict(self.ast_metadata),
            "timing_ms": asdict(self.timing_ms),
            "device": self.device,
            "status": self.status,
            "error_message": self.error_message,
        }


def extract_function_ast_info(tree: ast.AST) -> ASTMetadata:
    """
    Extract function metadata (names, parameters, return annotations)
    safely from a parsed Python AST.
    """
    functions: List[ast.FunctionDef | ast.AsyncFunctionDef] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node)

    if not functions:
        return ASTMetadata(
            function_names=["<module>"],
            parameters=[],
            return_annotation=None,
            docstring_stripped=False,
            syntax_valid=True,
        )

    primary_fn = functions[0]
    func_names = [fn.name for fn in functions]

    params: List[str] = []
    for arg in primary_fn.args.args:
        arg_str = arg.arg
        if arg.annotation:
            try:
                arg_str += f": {ast.unparse(arg.annotation)}"
            except Exception:
                pass
        params.append(arg_str)

    ret_annot: Optional[str] = None
    if primary_fn.returns:
        try:
            ret_annot = ast.unparse(primary_fn.returns)
        except Exception:
            ret_annot = None

    return ASTMetadata(
        function_names=func_names,
        parameters=params,
        return_annotation=ret_annot,
        docstring_stripped=False,
        syntax_valid=True,
    )


def assemble_markdown_document(
    function_name: str,
    docstring: str,
    parameters: List[str],
    return_annotation: Optional[str],
    source_code: str,
    mermaid_code: Optional[str] = None,
) -> str:
    """
    Assemble generated docstring, AST parameters, and Mermaid flowchart
    into a structured Markdown API documentation document.
    """
    lines: List[str] = []

    # Title
    lines.append(f"# `{function_name}`")
    lines.append("")

    # Signature
    param_str = ", ".join(parameters)
    ret_str = f" -> {return_annotation}" if return_annotation else ""
    lines.append("```python")
    lines.append(f"def {function_name}({param_str}){ret_str}")
    lines.append("```")
    lines.append("")

    # Overview / Generated Description
    lines.append("## Description")
    lines.append("")
    clean_doc = docstring.strip()
    if clean_doc:
        lines.append(clean_doc)
    else:
        lines.append("No documentation summary generated.")
    lines.append("")

    # Parameters Section
    if parameters:
        lines.append("## Parameters")
        lines.append("")
        for param in parameters:
            parts = param.split(":", 1)
            p_name = parts[0].strip()
            p_type = f" (`{parts[1].strip()}`)" if len(parts) > 1 else ""
            lines.append(f"- **`{p_name}`**{p_type}: Parameter input for `{function_name}`.")
        lines.append("")

    # Return Type Section
    if return_annotation:
        lines.append("## Returns")
        lines.append("")
        lines.append(f"- **`{return_annotation}`**: Computed result returned by the function.")
        lines.append("")

    # Control Flow Diagram (Mermaid)
    if mermaid_code:
        lines.append("## Control Flow Diagram")
        lines.append("")
        lines.append("```mermaid")
        lines.append(mermaid_code.strip())
        lines.append("```")
        lines.append("")

    return "\n".join(lines)


class CodeScribeGenerator:
    """
    Singleton service managing local model inference and documentation assembly.
    """

    def __init__(
        self,
        model_path: str | Path = "model/codescribe",
        device_override: Optional[str] = None,
        task_prefix: str = "document python code: ",
        max_source_length: int = 512,
        max_target_length: int = 128,
    ) -> None:
        self.model_path = Path(model_path)
        self.task_prefix = task_prefix
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.device_override = device_override

        self.model: Optional[Any] = None
        self.tokenizer: Optional[Any] = None
        self.device: str = "cpu"
        self.is_loaded: bool = False

    def load(self) -> None:
        """
        Load model and tokenizer strictly from local files.
        """
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint directory not found at: {self.model_path.resolve()}"
            )

        config_file = self.model_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(
                f"Missing config.json in model directory: {config_file.resolve()}"
            )

        # Determine target device
        if self.device_override:
            self.device = self.device_override
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        logger.info("Loading tokenizer from: %s", self.model_path.resolve())
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path), local_files_only=True)

        logger.info("Loading fine-tuned model onto %s from: %s", self.device, self.model_path.resolve())
        self.model = AutoModelForSeq2SeqLM.from_pretrained(str(self.model_path), local_files_only=True)
        self.model.to(self.device)
        self.model.eval()
        self.is_loaded = True

        logger.info("CodeScribeGenerator successfully loaded on %s.", self.device)

    def unload(self) -> None:
        """Unload model and free GPU resources."""
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("CodeScribeGenerator unloaded.")

    def generate_documentation(
        self,
        source_code: str,
        num_beams: int = 1,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        do_sample: bool = False,
        include_mermaid: bool = True,
    ) -> GenerationResult:
        """
        Synthesize natural-language documentation and Mermaid control flow
        from raw Python source code.
        """
        if not self.is_loaded or self.model is None or self.tokenizer is None:
            raise ModelNotLoadedError("Model is not loaded. Call load() first.")

        if not source_code or not source_code.strip():
            raise CodeSyntaxError("Submitted source code is empty.")

        t_start = time.perf_counter()

        # 1. Preprocessing: AST verification & docstring removal
        t0 = time.perf_counter()
        clean_code, status, meta = remove_docstrings_from_source(source_code)
        if status not in ("SUCCESS", "NO_DOCSTRING_PRESENT") or clean_code is None:
            err_msg = meta.get("error", f"Preprocessing rejected input with status '{status}'.")
            raise CodeSyntaxError(f"Invalid Python source code: {err_msg}")

        docstring_stripped = (status == "SUCCESS")

        # Parse AST for signature and parameters
        try:
            tree = ast.parse(source_code)
            ast_info = extract_function_ast_info(tree)
            ast_info.docstring_stripped = docstring_stripped
        except SyntaxError as e:
            raise CodeSyntaxError(f"Syntax error at line {e.lineno}: {e.msg}") from e

        t_pre = (time.perf_counter() - t0) * 1000.0

        # 2. Mermaid Control Flowchart Generation
        t0 = time.perf_counter()
        mermaid_code: Optional[str] = None
        if include_mermaid:
            m_res = generate_mermaid_flowchart(source_code)
            if m_res.success:
                mermaid_code = m_res.mermaid_code
            else:
                mermaid_code = f"graph TD\n    Error[\"Flowchart unavailable: {m_res.error_message}\"]"
        t_mermaid = (time.perf_counter() - t0) * 1000.0

        # 3. Model Inference (CodeT5 Documentation Generation)
        t0 = time.perf_counter()
        input_text = f"{self.task_prefix}{clean_code}"

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_source_length,
            truncation=True,
            return_tensors="pt",
        )
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)
        else:
            inputs = {k: v.to(self.device) if hasattr(v, "to") else v for k, v in inputs.items()}

        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        eos_token_id = self.tokenizer.eos_token_id if self.tokenizer.eos_token_id is not None else 2
        decoder_start_token_id = getattr(self.model.config, "decoder_start_token_id", pad_token_id)
        if decoder_start_token_id is None:
            decoder_start_token_id = pad_token_id

        gen_kwargs: Dict[str, Any] = {
            "max_new_tokens": min(max_new_tokens, self.max_target_length),
            "num_beams": num_beams,
            "do_sample": do_sample,
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "decoder_start_token_id": decoder_start_token_id,
        }
        if do_sample:
            gen_kwargs["temperature"] = temperature

        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )

        decoded = self.tokenizer.decode(output_ids[0], skip_special_tokens=True).strip()
        t_inf = (time.perf_counter() - t0) * 1000.0

        # 4. Markdown Document Assembly
        primary_name = ast_info.function_names[0] if ast_info.function_names else "function"
        markdown_doc = assemble_markdown_document(
            function_name=primary_name,
            docstring=decoded,
            parameters=ast_info.parameters,
            return_annotation=ast_info.return_annotation,
            source_code=clean_code,
            mermaid_code=mermaid_code,
        )

        t_total = (time.perf_counter() - t_start) * 1000.0

        timing = TimingMetadata(
            preprocessing_ms=round(t_pre, 2),
            inference_ms=round(t_inf, 2),
            mermaid_ms=round(t_mermaid, 2),
            total_ms=round(t_total, 2),
        )

        return GenerationResult(
            success=True,
            markdown=markdown_doc,
            docstring=decoded,
            mermaid_code=mermaid_code,
            ast_metadata=ast_info,
            timing_ms=timing,
            device=self.device,
            status="SUCCESS",
        )
