from pathlib import Path


def test_reference_modules_are_not_imported_by_controller_paths() -> None:
    roots = [
        Path("src/cage_pinn/audit"),
        Path("src/cage_pinn/controller"),
        Path("src/cage_pinn/enforcement"),
        Path("src/cage_pinn/sampling"),
    ]
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            assert "cage_pinn.references" not in source
            assert ".reference(" not in source

