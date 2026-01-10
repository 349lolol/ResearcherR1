def main():
    import typer
    from ingest.cli import app as ingest_app
    from eval.cli import app as eval_app

    app = typer.Typer()
    app.add_typer(ingest_app, name="ingest")
    app.add_typer(eval_app, name="eval")
    app()


if __name__ == "__main__":
    main()
