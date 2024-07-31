import network
from dataset import read_dataset
import torch
import numpy as np  
from diffusers.training_utils import EMAModel
import wandb
from tqdm.auto import tqdm
import time
import sys
from torchmetrics.functional import precision, recall
from loss import PrecisionRecallLoss, get_recall, get_accuracy, get_sdf_loss
from utils import get_keys_sdf

device = torch.device('cuda')

if __name__ == '__main__':
    obs_horizon = 1
    obs_dim = 88

    dataset_path = "dataset_h0_midi_mini.zarr"

    device = torch.device('cuda')
    precision_recall_loss = PrecisionRecallLoss()
    # create dataloader
    train_loader, test_loader = read_dataset(dataset_path=dataset_path, train_split=0.2)

    # create network object
    ae = network.Autoencoder(
        latent_dim=16,
        cond_dim=64,
    ).to(device)

    ckpt_path = "vae/ckpts/checkpoint_AE-dataset_h0_midi_mini-1712505102.7050881.ckpt"
    state_dict = torch.load(ckpt_path, map_location='cuda')
    ae.load_state_dict(state_dict)

    num_epochs = 20

    # Exponential Moving Average
    # accelerates training and improves stability
    # holds a copy of the model weights
    # ema = torch.optim.swa_utils.AveragedModel(vae, 
                                            # multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999))

    # Standard ADAM optimizer
    # Note that EMA parametesr are not optimized
    optimizer = torch.optim.AdamW(
        params=ae.parameters(),
        lr=1e-3, weight_decay=1e-6)

    # linear learning rate scheduler
    # lr_scheduler = torch.optim.lr_scheduler.LinearLR(
    #     optimizer=optimizer,
    #     end_factor=1e-6,
    #     start_factor=1e-4,
    #     total_steps=len(train_loader) * num_epochs
    # )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)
    # lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0)

    run_name = f"AE-{dataset_path.split('.')[0]}-{time.time()}"
    # wandb.login()
    # wandb.init(
    #     project="robopianist",
    #     name=run_name,
    #     config={},
    #     sync_tensorboard=True,
    # )
    min_loss = 1000000
    with tqdm(range(num_epochs), desc='Epoch') as tglobal:
        # epoch loop
        for epoch_idx in tglobal:
            epoch_loss = list()
            # batch loop
            # with tqdm(train_loader, desc='Batch', leave=False) as tepoch:
            #     for x in tepoch:
            #         # data normalized in dataset
            #         # device transfer
            #         x = x.to(device)
            #         # Randomize int q of shape (B,) in [0, 87]  
            #         q = torch.randint(0, 88, (x.shape[0], 1)).to(device)
            #         x_hat, sdf_hat = ae(x=x, q=q)

            #         # loss = ((x - x_hat)**2).sum() + vae.encoder.kl 
            #         # loss = ae.encoder.kl
            #         loss = get_sdf_loss(sdf_hat, x, q)
            #         loss += ae.encoder.reg * 1e-3

            #         # optimize
            #         loss.backward()
            #         optimizer.step()
            #         optimizer.zero_grad()
            #         # step lr scheduler every batch
            #         # this is different from standard pytorch behavior

            #         # update Exponential Moving Average of the model weights
            #         # ema.update_parameters(vae)

            #         # logging
            #         loss_cpu = loss.item()
            #         epoch_loss.append(loss_cpu)
            #         tepoch.set_postfix(loss=loss_cpu)
            # tglobal.set_postfix(loss=np.mean(epoch_loss))
            # # wandb.log({"loss": np.mean(epoch_loss)})
            # # wandb.log({"learning rate": lr_scheduler.get_last_lr()[0]})
            # # wandb.log({"epoch": epoch_idx})
            # lr_scheduler.step()
            # # Test every 10 epochs
            if epoch_idx % 1 == 0:
                # Test
                with torch.no_grad():
                    test_loss = list()
                    for x in test_loader:
                        x = x.to(device)
                        # Randomize int q of shape (B,) in [0, 87]  
                        q = torch.randint(0, 88, (x.shape[0], 1)).to(device)
                        x_hat, sdf_hat = ae.forward_without_sampling(x=x, q=q)
                        # loss = ae.encoder.kl
                        # loss = ((x - x_hat)**2).sum() + vae.encoder.kl 
                        loss = get_sdf_loss(sdf_hat, x, q)
                        test_loss.append(loss.item())
                        # wandb.log({"test loss": np.mean(test_loss)})
                        # wandb.log({"test accuracy": np.mean(test_acc)})
                        # wandb.log({"test recall": np.mean(test_rec)})
                    print("Test loss:", np.mean(test_loss))
                if np.mean(test_loss) < min_loss:
                    min_loss = np.mean(test_loss)
                    # torch.optim.swa_utils.update_bn(train_loader, ema)
                    # Assuming ema_model is your EMA model
                    ema_model_state_dict = ae.state_dict()

                    # Specify the path to save the EMA model's weights
                    ema_model_weights_path = 'vae/ckpts/checkpoint_{}.ckpt'.format(run_name)

                    # Save the EMA model's weights to the specified path
                    # torch.save(ema_model_state_dict, ema_model_weights_path)
                    print("Saved checkpoint at epoch {}".format(epoch_idx))

    # # Assuming ema_model is your EMA model
    # ema_model_state_dict = vae.state_dict()

    # # Specify the path to save the EMA model's weights
    # ema_model_weights_path = 'vae/ckpts/checkpoint_{}.ckpt'.format(run_name)

    # # Save the EMA model's weights to the specified path
    # torch.save(ema_model_state_dict, ema_model_weights_path)

    print("Done!")