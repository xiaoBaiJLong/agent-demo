# prompt_manager.py

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PyYAML import yaml


@dataclass
class PromptTemplate:
    prompt_id: str
    version: str
    scene: str | None
    system_prompt: str
    user_template: str

    @property
    def template(self) -> str:
        if not self.system_prompt:
            return self.user_template
        return f"{self.system_prompt}\n\n{self.user_template}"


class PromptManager:
    def __init__(self, prompt_dir: str | Path | None = None):
        self.prompt_dir = Path(prompt_dir) if prompt_dir else Path(__file__).resolve().parent / "prompts"
        self.templates = self._load_templates()

    def _load_templates(self) -> dict[str, PromptTemplate]:
        if not self.prompt_dir.exists():
            raise ValueError(f"Prompt directory not found: {self.prompt_dir}")

        templates: dict[str, PromptTemplate] = {}
        prompt_files = sorted([*self.prompt_dir.glob("*.yaml"), *self.prompt_dir.glob("*.yml")])

        for prompt_file in prompt_files:
            template = self._load_template_file(prompt_file)
            key = f"{template.prompt_id}:{template.version}"

            if key in templates:
                raise ValueError(f"Duplicate prompt template key: {key} in {prompt_file}")

            templates[key] = template

        return templates

    def _load_template_file(self, prompt_file: Path) -> PromptTemplate:
        data = yaml.safe_load(prompt_file.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Prompt YAML must be a mapping: {prompt_file}")

        prompt_id = self._require_string(data, "prompt_id", prompt_file)
        version = self._require_string(data, "version", prompt_file)
        user_template = self._require_string(data, "user_template", prompt_file)
        system_prompt = self._optional_string(data, "system_prompt", prompt_file)
        scene = self._optional_string(data, "scene", prompt_file) or None

        return PromptTemplate(
            prompt_id=prompt_id,
            version=version,
            scene=scene,
            system_prompt=system_prompt,
            user_template=user_template,
        )

    @staticmethod
    def _require_string(data: dict[str, Any], field_name: str, prompt_file: Path) -> str:
        value = data.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Missing required prompt field '{field_name}' in {prompt_file}")
        return value

    @staticmethod
    def _optional_string(data: dict[str, Any], field_name: str, prompt_file: Path) -> str:
        value = data.get(field_name, "")
        if value is None:
            return ""
        if not isinstance(value, str):
            raise ValueError(f"Prompt field '{field_name}' must be a string in {prompt_file}")
        return value

    def render(self, prompt_id: str, version: str, variables: dict) -> str:
        key = f"{prompt_id}:{version}"

        if key not in self.templates:
            raise ValueError(f"Prompt template not found: {key}")

        template = self.templates[key].template
        return template.format(**variables)
