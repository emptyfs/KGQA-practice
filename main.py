import os
import torch
from pykeen.pipeline import pipeline


device = 'cuda' if torch.cuda.is_available() else 'cpu'

models = [
    'TransE',
    'DistMult',
    'ComplEx',
    'TransH',
    'TransR',
    'RotatE',
    'RESCAL',
    'HolE',
]

datasets = ['FB15k-237', 'WN18RR', 'YAGO3-10']

os.makedirs('results', exist_ok=True)

for dataset in datasets:
    num_epochs = 100 if dataset != 'YAGO3-10' else 10
    for model_name in models:
        print(f"\nTraining the model: {model_name} on dataset: {dataset}")
        result = pipeline(
            model=model_name,
            dataset=dataset,
            training_kwargs=dict(num_epochs=num_epochs, batch_size=128, stopper='early'),
            stopper_kwargs=dict(frequency=10, patience=5),
            device=device,
            random_seed=42,
        )

        metrics = result.metric_results.to_dict()
        mrr = round(metrics['both']['realistic']['inverse_harmonic_mean_rank'], 4)
        hits1 = round(metrics['both']['realistic']['hits_at_1'], 4)
        hits10 = round(metrics['both']['realistic']['hits_at_10'], 4)

        filename = f'results/results_{dataset}_{model_name}.txt'

        with open(filename, 'w') as f:
            f.write(f"Dataset: {dataset}\n")
            f.write(f"Model: {model_name}\n")
            f.write(f"MRR: {mrr}\n")
            f.write(f"Hits@1: {hits1}\n")
            f.write(f"Hits@10: {hits10}\n")

        print(f"Saved to {filename}")
