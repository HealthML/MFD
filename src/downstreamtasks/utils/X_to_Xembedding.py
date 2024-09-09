import torch
import torch.utils.data as torch_data
import pytorch_lightning as pl

from utils.ShiftWrapper import ShiftWrapper


def X_to_Xembedding(Xnormal, Xreverse, model, shift_len, log_func=print):
    class EnhancerPreprocessed(torch_data.Dataset):
        def __init__(self, X):
            self.X = X

        def __getitem__(self, index):
            x = self.X[index].transpose(0, 1)
            x = x.float()
            return x

        def __len__(self):
            return len(self.X)

    enhancer_preprocessed_data = EnhancerPreprocessed(Xnormal)
    enhancer_preprocessed_dataloader = torch.utils.data.DataLoader(
        dataset=enhancer_preprocessed_data,
        batch_size=30,
        pin_memory=False
    )

    enhancer_preprocessed_data_complement = EnhancerPreprocessed(Xreverse)
    enhancer_preprocessed_dataloader_complement = torch.utils.data.DataLoader(
        dataset=enhancer_preprocessed_data_complement,
        batch_size=1000,
        pin_memory=False
    )

    shift_model = ShiftWrapper(model, shift_len)
    trainer = pl.Trainer(
        enable_checkpointing=False,
        accelerator='gpu' if int(torch.cuda.is_available()) else 'cpu',
        devices=1 if int(torch.cuda.is_available()) else 'auto',
    )

    def get_embeddings(enhancer_preprocessed_d):
        ret = trainer.predict(
            model=shift_model,
            dataloaders=enhancer_preprocessed_d
        )

        # Extract and concatenate features
        predictions, features_biological, features_technical = (
            torch.cat([batch[output_idx] for batch in ret], dim=0)
            for output_idx in range(3)
        )

        return predictions, features_biological, features_technical

    predictionsNormal, XembeddingsNormal_biologicial, XembeddingsNormal_technical = get_embeddings(
        enhancer_preprocessed_dataloader)
    predictionsReverse, XembeddingsReverse_biologicial, XembeddingsReverse_technical = get_embeddings(
        enhancer_preprocessed_dataloader_complement)
    log_func(f"predictionsNormal.shape {tuple(predictionsNormal.shape)}")
    log_func(f"predictionsReverse.shape {tuple(predictionsReverse.shape)}")
    log_func(f"XembeddingsNormal_biologicial.shape {tuple(XembeddingsNormal_biologicial.shape)}")
    log_func(f"XembeddingsNormal_technical.shape {tuple(XembeddingsNormal_technical.shape)}")
    log_func(f"XembeddingsReverse_biologicial.shape {tuple(XembeddingsReverse_biologicial.shape)}")
    log_func(f"XembeddingsReverse_technical.shape {tuple(XembeddingsReverse_technical.shape)}")

    XembeddingsNormal = torch.cat([XembeddingsNormal_biologicial, XembeddingsNormal_technical], dim=1)
    XembeddingsReverse = torch.cat([XembeddingsReverse_biologicial, XembeddingsReverse_technical], dim=1)
    log_func(f"XembeddingsNormal.shape {tuple(XembeddingsNormal.shape)}")
    log_func(f"XembeddingsReverse.shape {tuple(XembeddingsReverse.shape)}")

    Xembeddings_biological = (XembeddingsNormal_biologicial + XembeddingsReverse_biologicial) / 2
    Xembeddings_technical = (XembeddingsNormal_technical + XembeddingsReverse_technical) / 2
    log_func(f"Xembeddings_biological.shape {tuple(Xembeddings_biological.shape)}")
    log_func(f"Xembeddings_technical.shape {tuple(Xembeddings_technical.shape)}")

    # average the embeddings
    Xembeddings = (XembeddingsNormal + XembeddingsReverse) / 2
    log_func(f"Xembeddings.shape {tuple(Xembeddings.shape)}")

    return Xembeddings, Xembeddings_biological, Xembeddings_technical
