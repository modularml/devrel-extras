import click
from . import run_eval


@click.command()
@click.option('-s',
              '--server',
              default='http://0.0.0.0:8000',
              help='URL from "Uvicorn running on <URL>"')
@click.option('-m',
              '--model',
              default='meta-llama/Llama-Guard-3-8B',
              help='Model to evaluate')
@click.option('-k',
              '--keyword',
              default='Unsafe',
              help='Response model gives when content is unsafe')
@click.option('-n',
              '--num_examples',
              default=1000,
              help='Number of examples to evaluate')
def main(server: str, model: str, keyword: str, num_examples: int):
    run_eval(server, model, keyword, num_examples)


main()
