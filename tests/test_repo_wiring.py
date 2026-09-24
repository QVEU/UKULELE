import re

import pytest
import yaml

from conftest import ROOT

WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))


@pytest.mark.parametrize("workflow", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_parses_and_referenced_scripts_exist(workflow):
    spec = yaml.safe_load(workflow.read_text())
    assert spec["jobs"]
    for script in re.findall(r"python ((?:tools|retrieval|intake)/\S+\.py)", workflow.read_text()):
        assert (ROOT / script).exists(), f"{workflow.name} calls missing {script}"


def _package_names(lines):
    names = set()
    for line in lines:
        line = line.split("#")[0].strip().lstrip("- ").strip()
        if line and not line.endswith(":"):
            names.add(re.split(r"[<>=!\s]", line, maxsplit=1)[0].lower())
    return names


def test_environment_yml_covers_requirements_txt():
    required = _package_names((ROOT / "requirements.txt").read_text().splitlines())
    env = yaml.safe_load((ROOT / "environment.yml").read_text())
    listed = []
    for dep in env["dependencies"]:
        listed.extend(dep["pip"] if isinstance(dep, dict) else [dep])
    missing = required - _package_names(listed)
    assert not missing, f"environment.yml is missing {sorted(missing)}"
