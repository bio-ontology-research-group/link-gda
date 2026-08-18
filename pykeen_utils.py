from pykeen.stoppers import Stopper
import json
import os
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
                 dual=False,
                 model_out_filename_cal=None,
                 eval_chunk_size=100_000,
                 state_path=None,
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
        self.dual = dual
        self.model_out_filename_cal = model_out_filename_cal
        self.best_val_mr = float('inf')
        self.best_raw_mr = float('inf')
        self.best_cal_mr = float('inf')
        self.tol_raw = tolerance
        self.tol_cal = tolerance
        self.eval_chunk_size = eval_chunk_size
        self.state_path = state_path
        if state_path is not None and os.path.exists(state_path):
            self.load_state()

    _STATE_FIELDS = ("best_val_mr", "best_raw_mr", "best_cal_mr",
                     "curr_tolerance", "tol_raw", "tol_cal", "last_epoch")

    def save_state(self, epoch):
        """Persist the early-stopping counters so a resumed run keeps its patience."""
        if self.state_path is None:
            return
        self.last_epoch = epoch
        state = {k: getattr(self, k, None) for k in self._STATE_FIELDS}
        tmp = f"{self.state_path}.tmp"
        with open(tmp, "w") as fh:
            json.dump(state, fh)
        os.replace(tmp, self.state_path)

    def load_state(self):
        """Restore early-stopping counters written by a previous run of this configuration."""
        with open(self.state_path) as fh:
            state = json.load(fh)
        for k in self._STATE_FIELDS:
            if state.get(k) is not None:
                setattr(self, k, state[k])
        logger.info(
            f"resumed stopper state from {self.state_path}: epoch {getattr(self, 'last_epoch', None)}, "
            f"best raw MR {self.best_raw_mr:.4f}, best cal MR {self.best_cal_mr:.4f}, "
            f"tol raw {self.tol_raw}, tol cal {self.tol_cal}"
        )

    def get_summary_dict(self, *args, **kwargs):
        return dict()

    def should_stop(self, epoch):
        if self.dual:
            if self.tol_raw <= 0 and self.tol_cal <= 0:
                logger.info(f"Early stopping at epoch {epoch}: both arms exhausted patience.")
                return True
            return False
        if self.curr_tolerance <= 0:
            logger.info(f"Early stopping at epoch {epoch} due to no improvement in validation MR for {self.tolerance} evaluations.")
            return True
        else:
            return False

    def should_evaluate(self, epoch):
        if epoch % 20 != 0:
            return False
        self.model.eval()

        if self.dual:
            if self.use_graph:
                both = evaluate_by_graph(
                    model=self.model,
                    test_disease_genes=self.val_disease_genes,
                    disease2pheno=self.disease2pheno,
                    eval_genes=self.eval_genes,
                    triples_factory=self.triples_factory,
                    dual_metrics=True,
                    eval_chunk_size=self.eval_chunk_size,
                )
            else:
                both = evaluate_by_similarity(
                    model=self.model,
                    test_disease_genes=self.val_disease_genes,
                    gene2pheno=self.gene2pheno,
                    disease2pheno=self.disease2pheno,
                    eval_genes=self.eval_genes,
                    triples_factory=self.triples_factory,
                    dual_metrics=True,
                )
            raw_mr = both["raw"][0]["mr"]
            cal_mr = both["calibrated"][0]["mr"]

            if raw_mr < self.best_raw_mr:
                self.best_raw_mr = raw_mr
                self.tol_raw = self.tolerance
                th.save(self.model.state_dict(), self.model_out_filename)
            else:
                self.tol_raw -= 1

            if cal_mr < self.best_cal_mr:
                self.best_cal_mr = cal_mr
                self.tol_cal = self.tolerance
                th.save(self.model.state_dict(), self.model_out_filename_cal)
            else:
                self.tol_cal -= 1

            logger.info(
                f"Epoch {epoch}, Val raw MR: {raw_mr:.4f} (best {self.best_raw_mr:.4f}, "
                f"tol {self.tol_raw}) | Val calibrated MR: {cal_mr:.4f} "
                f"(best {self.best_cal_mr:.4f}, tol {self.tol_cal})"
            )
            self.save_state(epoch)
            return True

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
                     eval_chunk_size=self.eval_chunk_size,
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
                     calibrate=self.calibrate,
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

            self.save_state(epoch)
            return True
        else:
            return False
