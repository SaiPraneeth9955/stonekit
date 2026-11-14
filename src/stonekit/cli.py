"""The "stonekit <command>" command line (Section 8).

Implemented so far: `stonekit run-case`, `stonekit app` and the `data` commands.
TODO(later stages): the other Section 8 commands, each in the stage that needs it.
"""

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from stonekit.data.registry import dataset_statuses, find_project_root, load_raw_dataset_folders

app = typer.Typer(no_args_is_help=True, help="StoneKit. Research software. Not for clinical use.")
data_app = typer.Typer(no_args_is_help=True, help="Dataset commands.")
app.add_typer(data_app, name="data")


_CT_OPTION = typer.Option(None, "--ct", help="Path to a NIfTI CT, instead of --case.")
_OUT_OPTION = typer.Option(Path("data/interim/stonekit_runs"), "--out", help="Output folder.")
_CONFIG_OPTION = typer.Option(
    Path("configs/system_pipeline_v1.yaml"),
    "--config",
    help="Pipeline config. configs/baseline_pipeline.yaml is the frozen 100-epoch baseline without the kidney-distance filter.",
)


@app.command("run-case")
def run_case_command(
    case: str | None = typer.Option(None, "--case", help="A Li development scan ID, such as wd0254."),
    ct: Path | None = _CT_OPTION,
    out: Path = _OUT_OPTION,
    reuse_kidney: bool = typer.Option(False, "--reuse-kidney", help="Reuse a saved kidney segmentation."),
    device: str = typer.Option("gpu", "--device", help="gpu or cpu."),
    config_file: Path = _CONFIG_OPTION,
) -> None:
    """Process one CT from start to finish. Research software. Not for clinical use."""
    import re

    import yaml

    from stonekit.config import load_yaml
    from stonekit.data.li2022 import LiDevelopmentData
    from stonekit.data.li_split import ProtectedCaseError, require_development
    from stonekit.pipeline.config_check import check_config
    from stonekit.pipeline.run_case import run_case

    console = Console()
    if (case is None) == (ct is None):
        console.print("Error: give exactly one of --case or --ct", markup=False)
        raise typer.Exit(code=1)
    root = find_project_root()
    data = LiDevelopmentData(root)
    annotation = None
    try:
        if case is not None:
            ct_path, annotation = data.paths(case)
            case_id = case
        else:
            ct_path = ct
            case_id = ct.name.split(".")[0]
            # A Li scan given by path passes the same guard as one given by ID.
            match = re.search(r"wd\d{4}", ct.name, flags=re.IGNORECASE)
            if match:
                require_development([match.group().lower()], data.manifest)
    except ProtectedCaseError as error:
        console.print(f"Refused: {error}", markup=False)
        raise typer.Exit(code=2) from error
    if not Path(ct_path).exists():
        console.print(f"Error: {ct_path} does not exist", markup=False)
        raise typer.Exit(code=1)

    saved = root / "data" / "interim" / "li2022_kidney_ts" / f"{case_id}_ts.nii.gz"
    config_path = config_file if config_file.is_absolute() else root / config_file
    if not config_path.exists():
        console.print(f"Error: config {config_path} does not exist", markup=False)
        raise typer.Exit(code=1)
    try:
        config = load_yaml(config_path)
        check_config(config)
    except (ValueError, OSError, yaml.YAMLError) as error:
        console.print(f"Error: {error}", markup=False)
        raise typer.Exit(code=1) from error
    folder = run_case(
        Path(ct_path), root / out, config, root, case_id, annotation,
        saved if reuse_kidney and saved.exists() else None, device, config_path,
    )  # fmt: skip
    print(f"Output: {folder}")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    console.print(f"Total seconds: {report['total_seconds']}", markup=False)
    for name, method in report["methods"].items():
        count = len(method.get("stones", [])) if method.get("available") else "not available"
        console.print(f"{name}: stones {count}", markup=False)
    console.print(f"Case flags: {report['case_qc_flags']}", markup=False)
    console.print(f"Result status: {report.get('result_status')}", markup=False)
    if report.get("failure"):
        console.print(f"Failure: {report['failure']}", markup=False)
        raise typer.Exit(code=3)
    if report.get("result_status") == "no_result":
        console.print("No stone result: see the case flags.", markup=False)
        raise typer.Exit(code=4)


@app.command("app")
def app_command(
    port: int = typer.Option(8501, "--port", help="Local port."),
) -> None:
    """Open the local viewer for processed cases. Research software. Not for clinical use."""
    import subprocess
    import sys

    target = Path(__file__).parent / "app" / "streamlit_app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(target), "--server.port", str(port),
         "--server.headless", "true", "--browser.gatherUsageStats", "false",
         "--client.toolbarMode", "minimal", "--theme.base", "dark",
         "--theme.primaryColor", "#5b8fb9", "--theme.backgroundColor", "#121417",
         "--theme.secondaryBackgroundColor", "#1b1f24", "--theme.textColor", "#d8dde3"],
        check=True,
    )  # fmt: skip


@data_app.command("status")
def data_status() -> None:
    """Show which datasets are present, licences recorded and missing items."""
    console = Console()
    try:
        statuses = dataset_statuses(find_project_root())
    except (FileNotFoundError, ValueError) as error:
        console.print(f"Error: {error}", markup=False)
        raise typer.Exit(code=1) from error

    table = Table(title="StoneKit datasets")
    table.add_column("ID", no_wrap=True)
    table.add_column("Name")
    table.add_column("Access")
    table.add_column("Licence", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Location")
    for s in statuses:
        table.add_row(s.id, s.name, s.access_mode, s.licence, s.status, s.location)
    console.print(table)

    missing = [s.id for s in statuses if s.status != "present"]
    unverified = [s.id for s in statuses if s.licence == "VERIFY"]
    console.print(f"Missing: {', '.join(missing) or 'none'}", markup=False)
    console.print(f"Licence still to verify: {', '.join(unverified) or 'none'}", markup=False)


@data_app.command("download")
def data_download(
    dataset_id: str = typer.Option(..., "--id", help="Dataset ID. Only D1 is available so far."),
) -> None:
    """Verify an open source, then fetch any missing files with checksum checks."""
    from stonekit.data.download import D1, DownloadError, download_dataset

    console = Console()
    if dataset_id != "D1":
        console.print(f"Error: download is not available for {dataset_id}", markup=False)
        raise typer.Exit(code=1)
    try:
        root = find_project_root()
        outcomes = download_dataset(D1, root / load_raw_dataset_folders(root)["D1"])
    except (FileNotFoundError, ValueError, DownloadError, OSError) as error:
        console.print(f"Error: {error}", markup=False)
        raise typer.Exit(code=1) from error
    for name, outcome in outcomes.items():
        console.print(f"{name}: {outcome}", markup=False)


@data_app.command("build-freiburg")
def data_build_freiburg() -> None:
    """Build data/processed/freiburg_stones.parquet from the two D1 source files."""
    from stonekit.data.freiburg import FreiburgDataError, build_table, write_table

    console = Console()
    try:
        root = find_project_root()
        table = build_table(root / load_raw_dataset_folders(root)["D1"])
        output = root / "data" / "processed" / "freiburg_stones.parquet"
        write_table(table, output)
    except (FileNotFoundError, ValueError, FreiburgDataError, OSError) as error:
        console.print(f"Error: {error}", markup=False)
        raise typer.Exit(code=1) from error
    console.print(f"{len(table)} stones written to {output}", markup=False)
    disagree = int(table["spacing_fields_disagree"].sum())
    console.print(f"voxsiz and voxvol_mL disagree for {disagree} stones", markup=False)


@data_app.command("li-working-split")
def data_li_working_split() -> None:
    """Create the one DEVELOPMENT-ONLY Li split. It prints counts, never reserved case IDs."""
    import csv
    from collections import Counter

    from stonekit.config import load_yaml
    from stonekit.data import li_split

    console = Console()
    try:
        root = find_project_root()
        config_path = root / "configs" / "li_working_split.yaml"
        config = load_yaml(config_path)
        cases_table = root / config["cases_table"]
        review_sheet = root / config["review_sheet"]
        with cases_table.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        result = li_split.build_assignments(
            rows, config["fractions"], config["excluded"], config["seed"]
        )
        if len(result) != config["expected_cohort_size"]:
            raise ValueError(f"Cohort has {len(result)} scans, not {config['expected_cohort_size']}")
        digest = li_split.write_manifest(
            result,
            root / config["manifest"],
            {
                "seed": config["seed"],
                "config_sha256": li_split.sha256_file(config_path),
                "cases_table_sha256": li_split.sha256_file(cases_table),
                "review_sheet_sha256": li_split.sha256_file(review_sheet),
                "excluded": config["excluded"],
            },
        )
    except (FileNotFoundError, FileExistsError, ValueError, OSError) as error:
        console.print(f"Error: {error}", markup=False)
        raise typer.Exit(code=1) from error
    console.print("DEVELOPMENT-ONLY working split. Not the final scientific split.", markup=False)
    console.print(f"Counts: {dict(Counter(row['assignment'] for row in result))}", markup=False)
    console.print(f"SHA-256: {digest}", markup=False)
