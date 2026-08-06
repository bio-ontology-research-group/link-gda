from pykeen.stoppers import Stopper
import torch as th
from evaluation import evaluate_by_similarity, evaluate_by_graph

import logging
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ValidationStopper(Stopper):
    def __init__(self,
                 model,
                 triples_factory,
                 file_identifier,
                 val_disease_genes,
                 gene2pheno,
                 disease2pheno,
                 eval_genes,
                 tolerance,
                 model_out_filename,
                 use_graph=False,
                 calibrate=False,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model = model
        self.triples_factory = triples_factory
        self.file_identifier = file_identifier
        self.val_disease_genes = val_disease_genes
        self.gene2pheno = gene2pheno
        self.disease2pheno = disease2pheno
        self.eval_genes = eval_genes
        self.tolerance = tolerance
        self.curr_tolerance = tolerance
        self.model_out_filename = model_out_filename
        self.use_graph = use_graph
        self.calibrate = calibrate
        self.best_val_mr = float('inf')
        
    def get_summary_dict(self, *args, **kwargs):
        return dict()

    def should_stop(self, epoch):
        if self.curr_tolerance <= 0:
            logger.info(f"Early stopping at epoch {epoch} due to no improvement in validation MR for {self.tolerance} evaluations.")
            return True
        else:
            return False

    def should_evaluate(self, epoch):
        if epoch % 20 == 0:
            self.model.eval()
            # No output_file_prefix: the stopper needs only the mean rank, and the
            # per-instance score files it used to write were never read by anything
            # (diagnose_early_stopping.py reads the validation curve from the training
            # log). Writing them cost ~100 MB of I/O every twenty epochs.

            if self.use_graph:
                (val_inductive_bma_macro_metrics,
                 val_inductive_bmm_macro_metrics) = evaluate_by_graph(
                     model=self.model,
                     test_disease_genes=self.val_disease_genes,
                     disease2pheno=self.disease2pheno,
                     eval_genes=self.eval_genes,
                     triples_factory=self.triples_factory,
                     calibrate=self.calibrate,
                )
            else:
                (val_inductive_bma_macro_metrics,
                 val_inductive_bmm_macro_metrics) = evaluate_by_similarity(
                     model=self.model,
                     test_disease_genes=self.val_disease_genes,
                     gene2pheno=self.gene2pheno,
                     disease2pheno=self.disease2pheno,
                     eval_genes=self.eval_genes,
                     triples_factory=self.triples_factory,
                )

            
            val_mr = val_inductive_bma_macro_metrics['mr']
            metric_type = "inductive"

            if val_mr < self.best_val_mr:
                self.best_val_mr = val_mr
                self.curr_tolerance = self.tolerance
                th.save(self.model.state_dict(), self.model_out_filename)
                logger.info(f"\nEpoch {epoch} - New best validation {metric_type} MR: {val_mr:.4f}. Model saved.")
            else:
                self.curr_tolerance -= 1

            logger.info(f"Epoch {epoch}, Val {metric_type} MR: {val_mr:.4f}, Best Val MR: {self.best_val_mr:.4f}, Tolerance left: {self.curr_tolerance}")

            return True
        else:
            return False
