# Copyright 2026 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import inspect
import subprocess
import sys
import textwrap
from importlib import import_module

import pytest


class TestDependencies:
    def test_diffusers_import(self):
        import diffusers  # noqa: F401

    def test_backend_registration(self):
        import diffusers
        from diffusers.dependency_versions_table import deps

        all_classes = inspect.getmembers(diffusers, inspect.isclass)

        for cls_name, cls_module in all_classes:
            if "dummy_" in cls_module.__module__:
                for backend in cls_module._backends:
                    if backend == "k_diffusion":
                        backend = "k-diffusion"
                    elif backend == "invisible_watermark":
                        backend = "invisible-watermark"
                    elif backend == "opencv":
                        backend = "opencv-python"
                    elif backend == "nvidia_modelopt":
                        backend = "nvidia_modelopt[hf]"
                    elif backend == "auto_round":
                        backend = "auto-round"
                    assert backend in deps, f"{backend} is not in the deps table!"

    def test_pipeline_imports(self):
        import diffusers
        import diffusers.pipelines

        all_classes = inspect.getmembers(diffusers, inspect.isclass)
        for cls_name, cls_module in all_classes:
            if hasattr(diffusers.pipelines, cls_name):
                pipeline_folder_module = ".".join(str(cls_module.__module__).split(".")[:3])
                _ = import_module(pipeline_folder_module, str(cls_name))

    def test_pipeline_module_imports(self):
        """Import every pipeline submodule whose dependencies are satisfied,
        to catch unguarded optional-dep imports (e.g., torchvision).

        Uses inspect.getmembers to discover classes that the lazy loader can
        actually resolve (same self-filtering as test_pipeline_imports), then
        imports the full module path instead of truncating to the folder level.
        """
        import diffusers
        import diffusers.pipelines

        failures = []
        all_classes = inspect.getmembers(diffusers, inspect.isclass)

        for cls_name, cls_module in all_classes:
            if not hasattr(diffusers.pipelines, cls_name):
                continue
            if "dummy_" in cls_module.__module__:
                continue

            full_module_path = cls_module.__module__
            try:
                import_module(full_module_path)
            except ImportError as e:
                failures.append(f"{full_module_path}: {e}")
            except Exception:
                # Non-import errors (e.g., missing config) are fine; we only
                # care about unguarded import statements.
                pass

        if failures:
            pytest.fail("Unguarded optional-dependency imports found:\n" + "\n".join(failures))

    def test_ltx2_pipelines_import_on_older_transformers(self):
        """The LTX2 pipelines annotate `text_encoder` and `prompt_enhancer` with
        `Gemma4UnifiedForConditionalGeneration` and `Gemma4ForConditionalGeneration`, which transformers
        added in 5.10.0 and 5.5.0. Importing them must not require either symbol.

        Runs in a subprocess because the symbols have to be hidden before `diffusers` is imported.
        """
        script = textwrap.dedent(
            """
            import importlib.metadata
            import sys
            import types

            symbols = {"Gemma4ForConditionalGeneration", "Gemma4UnifiedForConditionalGeneration"}
            _version = importlib.metadata.version
            importlib.metadata.version = lambda name, *args, **kwargs: (
                "5.4.0" if name == "transformers" else _version(name, *args, **kwargs)
            )

            import transformers

            class _OlderTransformers(types.ModuleType):
                def __getattr__(self, name):
                    if name in symbols:
                        raise AttributeError(name)
                    return getattr(transformers, name)

            shim = _OlderTransformers("transformers")
            shim.__dict__.update({k: v for k, v in vars(transformers).items() if k not in symbols})
            sys.modules["transformers"] = shim

            import diffusers

            for name in [
                "LTX2Pipeline",
                "LTX2ImageToVideoPipeline",
                "LTX2ConditionPipeline",
                "LTX2InContextPipeline",
                "LTX2HDRPipeline",
                "LTX2DFRPipeline",
                "LTX2DFRTemporalRefinePipeline",
                "LTX2AutoBlocks",
                "LTX25AutoBlocks",
            ]:
                getattr(diffusers, name)
            """
        )
        result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
