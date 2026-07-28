from pathlib import Path


def test_src_files_do_not_reference_removed_module_paths():
    repo_root = Path(__file__).resolve().parents[1]
    files_and_expected_strings = {
        repo_root / "src/empo/__init__.py": [
            "from multigrid_worlds.one_or_three_chambers import SmallOneOrTwoChambersMapEnv",
        ],
        repo_root / "src/empo/learning_based/__init__.py": [
            "from empo.learning_based.multigrid import (",
            "from empo.learning_based import (",
        ],
        repo_root / "src/empo/learning_based/transport/__init__.py": [
            "from empo.learning_based.transport import (",
        ],
        repo_root / "src/empo/human_policy_prior.py": [
            "PathDistanceCalculator from empo.learning_based.multigrid",
        ],
    }

    for path, expected_strings in files_and_expected_strings.items():
        text = path.read_text()
        assert "empo.nn_based" not in text
        assert "src.envs" not in text
        for expected_string in expected_strings:
            assert expected_string in text
