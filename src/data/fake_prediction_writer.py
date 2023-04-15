from lightning.pytorch.callbacks import BasePredictionWriter


class FakePredictionWriter(BasePredictionWriter):

    def __init__(self, submission_save_path):
        super().__init__(write_interval='epoch')
        self.submission_save_path = submission_save_path

    def write_on_epoch_end(
        self,
        trainer,
        pl_module,
        predictions,
        batch_indices,
    ) -> None:
        """Override with the logic to write all batches."""
        with open(self.submission_save_path, 'w') as f:
            print(predictions, file=f)
