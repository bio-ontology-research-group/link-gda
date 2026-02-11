import pandas as pd
import random
import os
import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)

random.seed(42)

def split_pairs_kfold(pairs, k=10, split_by="tail"):
    # This codes assumes gene--disease pairs in that order (i.e., disease--gene pairs not allowed)
    tail_to_heads = dict()
    for head, tail in pairs:
        if tail not in tail_to_heads:
            tail_to_heads[tail] = []
        tail_to_heads[tail].append(head)

    tails = list(tail_to_heads.keys())
    random.shuffle(tails)

    # Calculate fold sizes
    fold_size = len(tails) // k
    remainder = len(tails) % k

    folds = []
    start_idx = 0

    for fold_idx in range(k):
        # Some folds get one extra tail if there's a remainder
        current_fold_size = fold_size + (1 if fold_idx < remainder else 0)
        fold_tails = tails[start_idx:start_idx + current_fold_size]

        # Convert tails back to pairs
        fold_pairs = [(head, tail) for tail in fold_tails for head in tail_to_heads[tail]]
        folds.append(fold_pairs)

        start_idx += current_fold_size

    # Verify all pairs are included exactly once across all folds
    total_pairs = sum(len(fold) for fold in folds)
    assert total_pairs == len(pairs)

    # Verify no overlap between folds
    all_fold_pairs = set()
    for fold in folds:
        fold_set = set(fold)
        assert len(fold_set & all_fold_pairs) == 0, "Overlapping pairs between folds"
        all_fold_pairs.update(fold_set)

    logger.info(f"Created {k} folds with sizes: {[len(fold) for fold in folds]}")

    return folds

def get_fold_splits(folds, test_fold_idx):
    """Get training and testing sets for a specific fold"""
    test = folds[test_fold_idx]
    train = []
    for i, fold in enumerate(folds):
        if i != test_fold_idx:
            train.extend(fold)
    
    return train, test

def main():
    association_file = "data/gene_diseases.csv"
    df = pd.read_csv(association_file)
    gene_disease_pairs = list(zip(df['Gene'], df['Disease']))
    assert len(gene_disease_pairs) == len(set(gene_disease_pairs)), "Duplicate gene-disease pairs found"

    folds = split_pairs_kfold(gene_disease_pairs, k=10)

    os.makedirs("data/folds", exist_ok=True)
    for i, fold in enumerate(folds):
        train, test = get_fold_splits(folds, i)

        train_pairs = set(train)
        test_pairs = set(test)
        
        train_diseases = set(tail for head, tail in train_pairs)
        test_diseases = set(tail for head, tail in test_pairs)
        assert len(train_pairs & test_pairs) == 0, "Overlap between train and test pairs"

        os.makedirs(f"data/folds/fold_{i}", exist_ok=True)

        with open(f"data/folds/fold_{i}/train.csv", "w") as f:
            f.write(f"Gene\tDisease\n")
            for head, tail in train:
                f.write(f"{head}\t{tail}\n")

        with open(f"data/folds/fold_{i}/test.csv", "w") as f:
            f.write(f"Gene\tDisease\n")
            for head, tail in test:
                f.write(f"{head}\t{tail}\n")

        logger.info(f"Saved fold {i}: {len(train)} train pairs, {len(test)} test pairs")
                
                
    
    
if __name__ == '__main__':
    main()
