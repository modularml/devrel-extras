import pandas as pd
import requests
import sys
import time
from io import StringIO
from openai import OpenAI
from rich.progress import Progress
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from urllib.parse import urljoin


def run_eval(server: str, model: str, keyword: str, n: int):
    client = setup_client(server)
    dataset = download_dataset()

    if dataset is not None and not dataset.empty:
        evaluation = evaluate_llm(dataset[:n], client, model, keyword)
        if not evaluation.empty:
            calculate_metrics(evaluation)
        else:
            print("Evaluation data contains no results.\nTry using/changing the --model flag.")


def setup_client(server: str) -> OpenAI:
    base_url = urljoin(server, "/v1")
    client = OpenAI(
        api_key='123',  # Use any value here; can't be blank or absent
        base_url=base_url,
    )
    return client


def download_dataset() -> pd.DataFrame:
    url = "https://raw.githubusercontent.com/surge-ai/toxicity/refs/heads/main/toxicity_en.csv"

    try:
        response = requests.get(url)
        response.raise_for_status()
        df = pd.read_csv(StringIO(response.text))        
        df['is_toxic'] = df['is_toxic'].apply(lambda x: 1 if x == 'Toxic' else 0)
        return df

    except Exception as e:
        print("Problem downloading dataset:", e)
        sys.exit(1)


def get_llm_prediction(client: OpenAI, input: str, model: str) -> str:
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": input}]
        )
        content = response.choices[0].message.content
        return content
    
    except Exception as _:
        return None


def evaluate_llm(dataset: pd.DataFrame, 
                 client: OpenAI,
                 model: str,
                 keyword: str) -> pd.DataFrame:
    results = pd.DataFrame(columns=["content", "y_true", "y_pred"])
    size = len(dataset)
    start_time = time.time()

    with Progress() as progress:       
        task = progress.add_task(
            f"[cyan]💬 Evaluating {model}[/cyan]",
            total=size
        )

        for i, row in dataset.iterrows():
            content = row['text']
            y_true = row['is_toxic']
            response = get_llm_prediction(client, content, model)

            if response:
                y_pred = 1 if keyword.lower() in response.lower() else 0
                results.loc[i] = [content, y_true, y_pred]
            else:
                size -= 1

            progress.update(task, advance=1)

        progress.update(
            task, 
            description=f"[green]✅ Evaluated {model} [/green]"
        )

    elapsed_time = time.time() - start_time
    elapsed_time_formatted = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    
    print(f"Time Elapsed: {elapsed_time_formatted}")
    print("Sample Size:", size)

    return results


def calculate_metrics(results: pd.DataFrame):
    accuracy = accuracy_score(results["y_true"], results["y_pred"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        results["y_true"], 
        results["y_pred"], 
        average="binary"
    )

    # How often do the model’s predictions match the ground-truth labels
    print(f"Accuracy: {accuracy:.2f}")

    # Of all model's "unsafe" predictions, how many did humans label as toxic? 
    print(f"Precision: {precision:.2f}")

    # Of all the human-labeled toxic items, how many did the model identify as unsafe?
    print(f"Recall: {recall:.2f}")

    # Harmonic mean of precision and recall, a single score that balances both metrics.
    print(f"F1 Score: {f1:.2f}")
