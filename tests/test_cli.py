from __future__ import annotations

from click.testing import CliRunner

from speechsum.cli import main


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(main, ["--help"])
    assert result.exit_code == 0
    assert "speechsum" in result.output


def test_cli_transcribe_help():
    runner = CliRunner()
    result = runner.invoke(main, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "Transcribe" in result.output


def test_cli_record_help():
    runner = CliRunner()
    result = runner.invoke(main, ["record", "--help"])
    assert result.exit_code == 0
    assert "Record" in result.output


def test_cli_models():
    runner = CliRunner()
    result = runner.invoke(main, ["models"])
    assert result.exit_code == 0


def test_cli_config_show():
    runner = CliRunner()
    result = runner.invoke(main, ["config", "show"])
    assert result.exit_code == 0
    assert "sample_rate" in result.output


def test_cli_download_help():
    runner = CliRunner()
    result = runner.invoke(main, ["download", "--help"])
    assert result.exit_code == 0
    assert "Download" in result.output


def test_cli_serve_help():
    runner = CliRunner()
    result = runner.invoke(main, ["serve", "--help"])
    assert result.exit_code == 0
    assert "Launch" in result.output


def test_cli_transcribe_missing_file():
    runner = CliRunner()
    result = runner.invoke(main, ["transcribe", "/nonexistent/file.wav"])
    assert result.exit_code != 0
