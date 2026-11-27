from asrs_rag.cli import main


def test_cli_without_command_prints_help(capsys):
    assert main([]) == 0
    assert "usage" in capsys.readouterr().out


def test_common_options_work_before_and_after_subcommand():
    from asrs_rag.cli import build_parser

    parser = build_parser()
    a = parser.parse_args(["--llm", "none", "summarise", "--phase", "approach"])
    b = parser.parse_args(["summarise", "--phase", "approach", "--llm", "none"])
    c = parser.parse_args(["summarise"])
    assert a.llm == b.llm == "none" and c.llm is None
    assert b.phase == "approach"
